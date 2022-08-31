#!/usr/bin/env python3
import numpy as np
from pymor.core.base import BasicObject
import scipy.linalg as spla
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.interface import Operator
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.logger import set_log_levels
set_log_levels({'pymor.algorithms.gram_schmidt': 'ERROR'})

A = np.random.rand(5, 3)
u = np.random.rand(5, 1)
V = NumpyVectorSpace.from_numpy(A.T)
X = NumpyMatrixOperator(A)

Qp, Rp = gram_schmidt(V, return_R=True)
Qp.append(NumpyVectorSpace.from_numpy(u.T))
Qp, Rp = gram_schmidt(Qp, offset=3, return_R=True)
Qp = Qp.to_numpy().T
q1, r1 = spla.qr(A, mode='economic')
Q, R = spla.qr_insert(q1, r1, u, 3, 'col')


class RandomizedSubspaceIterator(BasicObject):
    def __init__(self, A, q, source_product=None, range_product=None, complex=False):
        assert isinstance(A, Operator)
        if range_product is None:
            range_product = IdentityOperator(A.range)
        else:
            assert isinstance(range_product, Operator)

        if source_product is None:
            source_product = IdentityOperator(A.source)
        else:
            assert isinstance(source_product, Operator)

        assert 0 <= q and isinstance(q, int)
        assert isinstance(complex, bool)

        self.__auto_init(locals())
        self.l = 0
        self.Q = [A.range.empty() for _ in range(2*q+1)]

    def sample(self, n):
        assert 0 <= n and isinstance(n, int)
        if (n + self.l) > min(self.A.source.dim, self.A.range.dim):
            print('warn')

        W = self.A.source.random(n, distribution='normal')
        if self.complex:
            W += 1j*self.A.source.random(n, distribution='normal')

        self.Q[0].append(self.A.apply(W))
        gram_schmidt(self.Q[0], self.range_product, offset=self.l, copy=False)
        for i in range(self.q):
            i = 2*i + 1
            self.Q[i].append(self.source_product.apply_inverse(
                (self.A.apply_adjoint(self.range_product.apply(self.Q[0][-n:])))))
            gram_schmidt(self.Q[i], self.source_product, offset=self.l, copy=False)
            self.Q[i+1].append(self.A.apply(self.Q[i][-n:]))
            gram_schmidt(Q[i+1], self.range_product, offset=self.l, copy=False)

        self.l += n
        return self.Q[-1].to_numpy().T
