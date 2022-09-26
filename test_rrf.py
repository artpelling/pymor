#!/usr/bin/env python3
import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.cache import CacheableObject, cached
from pymor.operators.interface import Operator
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.logger import set_log_levels
from pymor.tools.random import get_seed_seq, new_rng

from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

set_log_levels({'pymor.algorithms.gram_schmidt': 'ERROR'})


class RandomizedRangeFinder(CacheableObject):
    def __init__(self, A, subspace_iterations=0, range_product=None, source_product=None, lambda_min=None,
                 complex=False):
        assert isinstance(A, Operator)
        if range_product is None:
            range_product = IdentityOperator(A.range)
        else:
            assert isinstance(range_product, Operator)

        if source_product is None:
            source_product = IdentityOperator(A.source)
        else:
            assert isinstance(source_product, Operator)

        assert 0 <= subspace_iterations and isinstance(subspace_iterations, int)
        assert isinstance(complex, bool)

        self.__auto_init(locals())
        self._l = 0
        self._Q = [self.A.range.empty()]
        for _ in range(subspace_iterations):
            self._Q.append(self.A.source.empty())
            self._Q.append(self.A.range.empty())
        self._Q = tuple(self._Q)
        self.testvecs = self.A.source.empty()
        self._basis_rng_real = new_rng(get_seed_seq().spawn(1)[0])
        self._test_rng_real = new_rng(get_seed_seq().spawn(1)[0])
        if complex:
            self._basis_rng_imag = new_rng(get_seed_seq().spawn(1)[0])
            self._test_rng_imag = new_rng(get_seed_seq().spawn(1)[0])

    @cached
    def _lambda_min(self):
        if isinstance(self.source_product, IdentityOperator):
            return 1
        elif self.lambda_min is None:
            with self.logger.block('Estimating minimum singular value of source_product...'):
                def mv(v):
                    return self.source_product.apply(self.source_product.source.from_numpy(v)).to_numpy()

                def mvinv(v):
                    return self.source_product.apply_inverse(self.source_product.range.from_numpy(v)).to_numpy()
                L = LinearOperator((self.source_product.source.dim, self.source_product.range.dim), matvec=mv)
                Linv = LinearOperator((self.source_product.range.dim, self.source_product.source.dim), matvec=mvinv)
                return eigsh(L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv)[0]
        else:
            return self.lambda_min

    def _draw_test_vector(self, n):
        with self._test_rng_real:
            W = self.A.source.random(n, distribution='normal')
            if self.complex:
                with self._test_rng_imag:
                    W += 1j * self.A.source.random(n, distribution='normal')
        self.testvecs.append(self.A.apply(W))

    def _maxnorm(self, basis_size, num_testvecs):
        if len(self.testvecs) < num_testvecs:
            self._draw_test_vector(num_testvecs - len(self.testvecs))

        W, Q = self.testvecs[:num_testvecs], self.find_range(basis_size=basis_size, tol=None)
        W -= Q.lincomb(Q.inner(W, self.range_product).T)
        return np.max(W.norm(self.range_product))

    @cached
    def _c_est(self, num_testvecs, p_fail):
        c = np.sqrt(2 * self._lambda_min()) \
            * erfinv((p_fail / min(self.A.source.dim, self.A.range.dim)) ** (1 / num_testvecs))
        return 1 / c

    def estimate_error(self, basis_size, num_testvecs=20, p_fail=1e-14):
        assert 0 < num_testvecs and isinstance(num_testvecs, int)
        assert 0 < p_fail

        err = self._c_est(num_testvecs, p_fail) * self._maxnorm(basis_size, num_testvecs)
        self.logger.info(f'estimated error: {err:.2f}')

        return err

    def _extend_basis(self, n=1):
        self.logger.info(f'Appending {n} basis vector{"s" if n > 1 else ""}...')

        with self._basis_rng_real:
            W = self.A.source.random(n, distribution='normal')
        if self.complex:
            with self._basis_rng_imag:
                W += 1j * self.A.source.random(n, distribution='normal')

        self._Q[0].append(self.A.apply(W))
        gram_schmidt(self._Q[0], self.range_product, offset=self._l, copy=False)

        for i in range(self.subspace_iterations):
            i = 2*i + 1
            self._Q[i].append(self.source_product.apply_inverse(
                (self.A.apply_adjoint(self.range_product.apply(self._Q[i-1][-n:])))))
            gram_schmidt(self._Q[i], self.source_product, offset=self._l, copy=False)
            self._Q[i+1].append(self.A.apply(self._Q[i][-n:]))
            gram_schmidt(self._Q[i+1], self.range_product, offset=self._l, copy=False)

        self._l += n

    def find_range(self, basis_size=8, tol=None, num_testvecs=20, p_fail=1e-14, block_size=8, increase_block=True):
        assert isinstance(basis_size, int) and basis_size > 0
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Basis is larger than the rank of the operator!')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert tol is None or tol > 0
        assert isinstance(num_testvecs, int) and num_testvecs > 0
        assert p_fail > 0
        assert isinstance(block_size, int) and block_size > 0
        assert isinstance(increase_block, bool)

        if basis_size > self._l:
            self._extend_basis(basis_size - self._l)

        if tol is None or self.estimate_error(basis_size, num_testvecs, p_fail) <= tol:
            return self._Q[-1][:basis_size]
        else:
            with self.logger.block('Extending range basis adaptively...'):
                while self._l < min(self.A.source.dim, self.A.range.dim):
                    basis_size += block_size
                    if self.estimate_error(basis_size, num_testvecs, p_fail) <= tol:
                        break
                    if increase_block:
                        block_size *= 2
            return self.find_range(basis_size=basis_size, tol=None)


n = 1000
with new_rng(0) as RNG:
    B = RNG.random((n, n))
V = NumpyVectorSpace.from_numpy(B)
X = NumpyMatrixOperator(B)

q = 0
l = 1
tol = 32
p_fail = 1e-12
with new_rng(0):
    RSI = RandomizedRangeFinder(X, subspace_iterations=q)
    Q1 = RSI.find_range(basis_size=l, tol=tol, p_fail=p_fail)
