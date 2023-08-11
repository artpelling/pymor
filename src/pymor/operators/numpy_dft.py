# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Matrix-free |Operators| represented by |NumPy| arrays with DFT-based matrix-vector multiplicaion.

This module provides the following |NumPy|-based |Operators|:

- |NumpyCirculantOperator| matrix-free operator of a circulant matrix.
- |NumpyToeplitzOperator| matrix-free operator of a Toeplitz matrix.
- |NumpyHankelOperator| matrix-free operator of a Hankel matrix.
"""

import numpy as np
from scipy.fft import fft, get_workers, ifft, irfft, rfft

from pymor.core.cache import CacheableObject, cached
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NumpyCirculantOperator(Operator, CacheableObject):
    r"""Matrix-free representation of a circulant matrix by a |NumPy Array|.

    A circulant matrix is a special kind of Toeplitz matrix which is square and completely
    determined by its first column via

    .. math::
        C =
            \begin{bmatrix}
                c_1    & c_n    & c_{n-1} & \cdots & \cdots & c_2 \\
                c_2    & c_1    & c_n     &        &        & \vdots \\
                c_3    & c_2    & \ddots  &        &        & \vdots \\
                \vdots &        &         & \ddots & c_n    & c_{n-1} \\
                \vdots &        &         & c_2    & c_1    & c_n \\
                c_n    & \cdots & \cdots  & c_3    & c_2    & c_1
            \end{bmatrix} \in \mathbb{C}^{n \times n},

    where the so-called circulant vector :math:`c \in \mathbb{C}^n` denotes the first column of the
    matrix. The matrix :math:`C` as seen above is not explicitly constructed, only `c` is stored.
    Efficient matrix-vector multiplications are realized with DFT in the class' `apply` method.
    See :cite:`GVL13` Chapter 4.8.2. for details.

    Parameters
    ----------
    c
        The |NumPy array| that defines the circulant vector.
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    cache_region = 'memory'

    def __init__(self, c, source_id=None, range_id=None, name=None, scipy=True):
        assert isinstance(c, np.ndarray)
        if c.ndim == 1:
            c = c.reshape(-1, 1, 1)
        assert c.ndim == 3
        c.setflags(write=False)  # make numpy arrays read-only
        self.__auto_init(locals())
        n, p, m = c.shape
        self._arr = c
        self.linear = True
        self.source = NumpyVectorSpace(n*m, source_id)
        self.range = NumpyVectorSpace(n*p, range_id)

    @cached
    def _circulant(self):
        return (rfft(self._arr, axis=0) if np.isrealobj(self._arr) else fft(self._arr, axis=0))

    def _circular_matvec(self, vec):
        s, k = vec.shape
        # use real arithmetic if possible
        isreal = np.isrealobj(self._arr) and np.isrealobj(vec)
        ismixed = np.isrealobj(self._arr) and np.iscomplexobj(vec)

        C = self._circulant()
        if ismixed:
            C = np.concatenate([C, C[1:(None if s % 2 else -1)].conj()[::-1]])

        dtype = float if isreal else complex
        y = np.zeros((self.range.dim, k), dtype=dtype)
        n, p, m = self._arr.shape
        for j in range(m):
            x = vec[j::m]
            X = rfft(x, axis=0) if isreal else fft(x, axis=0)
            for i in range(p):
                Y = X*C[:, i, j].reshape(-1, 1)
                # setting n=n below is necessary to allow uneven lengths but considerably slower
                # Hankel operator will always pad to even length to avoid that
                Y = irfft(Y, n=n, axis=0) if isreal else ifft(Y, axis=0)
                y[i::p] += Y[:self.range.dim // p]
        return y.T

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy().T
        return self.range.make_array(self._circular_matvec(U))

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @property
    def H(self):
        return self.with_(c=np.roll(self._arr.conj(), -1, axis=0)[::-1].transpose(0, 2, 1),
                          source_id=self.range_id, range_id=self.source_id, name=self.name + '_adjoint')


class NumpyToeplitzOperator(NumpyCirculantOperator):
    r"""Matrix-free representation of a Toeplitz matrix by a |NumPy Array|.

    A Toeplitz matrix is a matrix with constant diagonals, i.e.:

    .. math::
        T =
            \begin{bmatrix}
                c_1    & r_2    & r_3    & \cdots & \cdots & r_n \\
                c_2    & c_1    & r_2    &        &        & \vdots \\
                c_3    & c_2    & \ddots &        &        & \vdots \\
                \vdots &        &        & \ddots & r_2    & r_3 \\
                \vdots &        &        & c_2    & c_1    & r_2 \\
                c_m    & \cdots & \cdots & c_3    & c_2    & c_1
            \end{bmatrix} \in \mathbb{C}^{m \times n},

    where :math:`c \in \mathbb{C}^m` and :math:`r \in \mathbb{C}^n` denote the first column and first
    row of the Toeplitz matrix, respectively. The matrix :math:`T` as seen above is not explicitly
    constructed, only the arrays `c` and `r` are stored. The operator's `apply` method takes
    advantage of the fact that any Toeplitz matrix can be embedded in a larger circulant matrix to
    leverage efficient matrix-vector multiplications with DFT.

    Parameters
    ----------
    c
        The |NumPy array| that defines the first column of the Toeplitz matrix.
    r
        The |NumPy array| that defines the first row of the Toeplitz matrix. If supplied, its first
        entry `r[0]` has to equal to `c[0]`. Defaults to `None`. If `r` is `None`, the behavior
        of :func:`scipy.linalg.toeplitz` is mimicked which sets `r = c.conj()` (except for the first entry).
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    def __init__(self, c, r=None, source_id=None, range_id=None, name=None):
        assert isinstance(c, np.ndarray)
        c = c.reshape(-1, 1, 1) if c.ndim == 1 else c
        assert c.ndim == 3
        if r is None:
            r = np.zeros_like(c)
            r[0] = c[-1]
        else:
            assert isinstance(r, np.ndarray)
            r = r.reshape(-1, 1, 1) if r.ndim == 1 else r
            assert r.ndim == 3
            assert c.shape[1:] == r.shape[1:]
            assert np.allclose(c[0], r[0])
        c.setflags(write=False)
        r.setflags(write=False)
        super().__init__(np.concatenate([c, r[:0:-1]]), source_id=source_id, range_id=range_id, name=name)
        n, p, m = self._arr.shape
        self.source = NumpyVectorSpace(m*r.shape[0], source_id)
        self.range = NumpyVectorSpace(p*c.shape[0], range_id)
        self.c = c
        self.r = r

    def apply(self, U, mu=None):
        assert U in self.source
        U = np.concatenate([U.to_numpy().T, np.zeros((self._arr.shape[0] - U.dim, len(U)))])
        return self.range.make_array(self._circular_matvec(U)[:self.range.dim])

    @property
    def H(self):
        return self.with_(c=self.r.conj().transpose(0, 2, 1), r=self.c.conj().transpose(0, 2, 1), source_id=self.range_id, range_id=self.source_id,
                          name=self.name + '_adjoint')


class NumpyHankelOperator(NumpyCirculantOperator):
    r"""Matrix-free representation of a Hankel matrix by a |NumPy Array|.

    A Hankel matrix is a matrix with constant anti-diagonals, i.e.:

    .. math::
        H =
            \begin{bmatrix}
                c_1    & c_2    & c_3    & \cdots  & \cdots  & r_1 \\
                c_2    & c_3    &        &         &         & \vdots \\
                c_3    &        &        &         &         & \vdots \\
                \vdots &        &        &         &         & r_{n-2} \\
                \vdots &        &        &         & r_{n-2} & r_{n-1} \\
                c_m    & \cdots & \cdots & r_{n-2} & r_{n-1} & r_n
            \end{bmatrix} \in \mathbb{C}^{m \times n},

    where :math:`c \in \mathbb{C}^m` and :math:`r \in \mathbb{C}^n` denote the first column and last
    row of the Hankel matrix, respectively.
    The matrix :math:`H` as seen above is not explicitly constructed, only the arrays `c` and `r`
    are stored. Efficient matrix-vector multiplications are realized with DFT in the class' `apply`
    method (see :cite:`MSKC21` Algorithm 3.1. for details).

    Parameters
    ----------
    c
        The |NumPy array| that defines the first column of the Hankel matrix.
    r
        The |NumPy array| that defines the last row of the Hankel matrix. If supplied, its first
        entry `r[0]` has to equal to `c[-1]`. Defaults to `None`. If `r` is `None`, the behavior
        of :func:`scipy.linalg.hankel` is mimicked which sets `r` to zero (except for the first entry).
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    def __init__(self, c, r=None, source_id=None, range_id=None, name=None):
        assert isinstance(c, np.ndarray)
        c = c.reshape(-1, 1, 1) if c.ndim == 1 else c
        assert c.ndim == 3
        if r is None:
            r = np.zeros_like(c)
            r[0] = c[-1]
        else:
            assert isinstance(r, np.ndarray)
            r = r.reshape(-1, 1, 1) if r.ndim == 1 else r
            assert r.ndim == 3
            assert c.shape[1:] == r.shape[1:]
            assert np.allclose(r[0], c[-1])
        c.setflags(write=False)
        r.setflags(write=False)
        k, l = c.shape[0], r.shape[0]
        n = k + l - 1
        # zero pad to even length if real to avoid slow irfft
        z = int(np.isrealobj(c) and np.isrealobj(r) and n % 2)
        h = np.concatenate((c, r[1:], np.zeros([z, *c.shape[1:]])))
        shift = n // 2 + int(np.ceil((k - l) / 2)) + (n % 2) + z # this works
        super().__init__(np.roll(h, shift, axis=0), source_id=source_id, range_id=range_id, name=name)
        p, m = self._arr.shape[1:]
        self.source = NumpyVectorSpace(l*m, source_id)
        self.range = NumpyVectorSpace(k*p, range_id)
        self.c = c
        self.r = r

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy().T
        n, p, m = self._arr.shape
        x = np.zeros((n*m, U.shape[1]), dtype=U.dtype)
        for j in range(m):
            x[:self.source.dim][j::m] = np.flip(U[j::m], axis=0)
        return self.range.make_array(self._circular_matvec(x)[:self.range.dim])

    @property
    def H(self):
        h = np.concatenate([self.c, self.r[1:]], axis=0).conj().transpose(0, 2, 1)
        return self.with_(c=h[:self.r.shape[0]], r=h[self.r.shape[0]-1:],
                          source_id=self.range_id, range_id=self.source_id, name=self.name+'_adjoint')
