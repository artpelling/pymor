#!/usr/bin/env python3
import numpy as np
from pymor.core.base import BasicObject
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.rand_la import InverseOperator, rrf
from pymor.core.cache import cached
from pymor.operators.interface import Operator
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.core.logger import set_log_levels
from src.pymor.tools.random import get_seed_seq, new_rng

from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

set_log_levels({'pymor.algorithms.gram_schmidt': 'ERROR'})


class RandomizedSubspaceIterator(BasicObject):
    def __init__(self, A, subspace_iterations=0, source_product=None, range_product=None, complex=False):
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
        # TODO        assert isinstance(complex, bool)

        self.__auto_init(locals())
        self._l = 0
        self._Q = [self.A.range.empty()]
        for _ in range(subspace_iterations):
            self._Q.append(self.A.source.empty())
            self._Q.append(self.A.range.empty())
        self._Q = tuple(self._Q)
        self._test_seed_seq, sample_seed_seq = get_seed_seq().spawn(1)[0]
        self._sample_rng = new_rng(sample_seed_seq)

    @cached
    def _lambda_min(self):
        if self.source_product is None:
            return 1
        elif self.lambda_min is None:
            def mv(v):
                return self.source_product.apply(self.source_product.source.from_numpy(v)).to_numpy()

            def mvinv(v):
                return self.source_product.apply_inverse(self.source_product.range.from_numpy(v)).to_numpy()
            L = LinearOperator((self.source_product.source.dim, self.source_product.range.dim), matvec=mv)
            Linv = LinearOperator((self.source_product.range.dim, self.source_product.source.dim), matvec=mvinv)
            return eigsh(L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv)[0]
        else:
            return self.lambda_min

    @cached
    def _maxnorm(self, n):
        with new_rng(self._test_seed_seq):
            W = self.A.source.random(n)
        Q = self.find_range(n, tol=None)
        W -= Q.lincomb(Q.inner(W, self.range_product).T)
        return np.max(W.norm(self.range_product))

    @cached
    def estimate_error(self, n, p_fail):
        c_est = np.sqrt(2 * self._lambda_min) * erfinv((p_fail / min(self.A.source.dim, self.A.range.dim)) ** (1 / n))
        return 1 / c_est * self._maxnorm(n)

    def _extend_basis(self, n=1):
        assert 0 <= n and isinstance(n, int)
        if (n + self._l) > min(self.A.source.dim, self.A.range.dim):
            print('warn')

        with self._sample_rng:
            W = self.A.source.random(n, distribution='normal')

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

    def find_range(self, n, p_fail=1e-14, tol=None):
        if tol is None:
            if n < self._l:
                self._extend_basis(n - self._l)
            return self._Q[-1][:n]
        else:
            while(self._estimate_error(n, p_fail) > tol):
                self._extend_basis()
            return self.Q[-1]


A = np.random.rand(10, 10)
W = np.diag(np.abs(np.random.rand(5)))
W = NumpyMatrixOperator(W)
V = NumpyVectorSpace.from_numpy(A)
X = NumpyMatrixOperator(A)

q = 1
l = 3
RSI = RandomizedSubspaceIterator(X, q)
W1 = InverseOperator(W)

Q = rrf(X)
W = X.source.random(10)
