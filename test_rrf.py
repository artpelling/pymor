#!/usr/bin/env python3
import numpy as np
from pymor.core.base import BasicObject
import scipy.linalg as spla
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.rand_la import rrf
from pymor.operators.interface import Operator
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.logger import set_log_levels
from src.pymor.tools.random import new_rng
set_log_levels({'pymor.algorithms.gram_schmidt': 'ERROR'})


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
        self.Q = [self.A.range.empty()]
        for _ in range(q):
            self.Q.append(self.A.source.empty())
            self.Q.append(self.A.range.empty())
        self.Q = tuple(self.Q)

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
            gram_schmidt(self.Q[i+1], self.range_product, offset=self.l, copy=False)

        self.l += n
        return self.Q[-1].to_numpy().T

from pymor.tools.random import new_rng
from pymor.core.defaults import print_defaults, set_defaults
set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.atol": 0, "pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 0})

A = np.random.rand(5, 3)
u = np.random.rand(5, 1)
V = NumpyVectorSpace.from_numpy(A.T)
X = NumpyMatrixOperator(A)


RSI = RandomizedSubspaceIterator(X, 2)



with new_rng(0):
    Q = RSI.sample(3)
    print(Q.T)

with new_rng(0):
    Q = rrf(X, q=2, l=3)
    print(Q)
