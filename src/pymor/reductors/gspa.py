# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.models.transforms import MoebiusTransformation
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import LTIPGReductor


class GSPAReductor(BasicObject):
    """Generalized Singular Perturbation reductor.

    Parameters
    ----------
    fom
        The full-order |LTIModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, LTIModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self.W = None
        self._pg_reductor = None

    def _gramians(self):
        return self.fom.gramian('c_lrcf', mu=self.mu), self.fom.gramian('o_lrcf', mu=self.mu)

    def _sv_U_V(self):
        return self.fom._sv_U_V(mu=self.mu)

    def error_bounds(self):
        sv = self._sv_U_V()[0]
        return 2 * sv[:0:-1].cumsum()[::-1]

    def reduce(self, r=None, tol=None, s0=None, projection='bfsr', MT=None, old=True):
        """Generalized Singular Perturbation Approximation.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`, maximum order if `tol` is specified.
        tol
            Tolerance for the error bound if `r` is `None`.
        s0
            The interpolation frequency. `s0=0` is equivalent to regular Singular Perturbation
            Approximation, `s0=np.inf` is equivalent to Balanced Truncation. Defaults to `np.inf`.
        projection
            Projection method used:

            - `'sr'`: square root method
            - `'bfsr'`: balancing-free square root method (default, since it avoids scaling by
              singular values and orthogonalizes the projection matrices, which might make it more
              accurate than the square root method)

        Returns
        -------
        rom
            Reduced-order model.
        """
        assert r is not None or tol is not None
        assert r is None or 0 < r < self.fom.order
        assert projection in ('sr', 'bfsr')

        cf, of = self._gramians()
        sv, sU, sV = self._sv_U_V()

        # find reduced order if tol is specified
        if tol is not None:
            error_bounds = self.error_bounds()
            r_tol = np.argmax(error_bounds <= tol) + 1
            r = r_tol if r is None else min(r, r_tol)
        if r > min(len(cf), len(of)):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.')

        # compute projection matrices
        self.V = cf.lincomb(sV[:r])
        self.W = of.lincomb(sU[:r])
        if projection == 'sr':
            alpha = 1 / np.sqrt(sv[:r])
            self.V.scal(alpha)
            self.W.scal(alpha)
        elif projection == 'bfsr':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)
            gram_schmidt(self.W, atol=0, rtol=0, copy=False)

        # find reduced-order model
        if self.fom.parametric:
            fom_mu = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                       for op in ['A', 'B', 'C', 'D', 'E']})
        else:
            fom_mu = self.fom

        if s0 == np.inf:
            self._pg_reductor = LTIPGReductor(fom_mu, self.W, self.V, projection == 'sr')
            rom = self._pg_reductor.reduce()
        else:
            M = MoebiusTransformation(np.array([0, 1j, 1j, -s0]), normalize=True) if MT is None else MT
            a, b, c, d = M.coefficients

            # E = project(fom_mu.E, self.W, self.V)
            # A = project(fom_mu.A, self.W, self.V)
            # B = project(fom_mu.B, self.W, None)

            # G = to_matrix(d*fom_mu.E + c*fom_mu.A, format='dense')
            # LU = spla.lu_factor(G)
            # M = to_matrix(fom_mu.E @ LowRankOperator(self.V, E.matrix,
            # self.W, inverted=True), format='dense')

            # Bv = to_matrix(fom_mu.B, format='dense')
            # Cv = to_matrix(fom_mu.C, format='dense')
            # Dv = to_matrix(fom_mu.D, format='dense')

            # R1 = M @ G @ self.V.to_numpy().T
            # C = NumpyMatrixOperator(Cv @ spla.lu_solve(LU, R1))

            # R2 = M @ Bv - Bv
            # D = NumpyMatrixOperator(Dv + c*Cv @ spla.lu_solve(LU, R2))

            # rom = LTIModel(A, B, C, D=D, E=E, name=fom_mu.name+'_reduced')


            kappa = np.sqrt(complex(d*a-b*c))
            As, Bs, Cs, *_ = fom_mu.to_matrices(format='dense')
            Es = to_matrix(fom_mu.E, format='dense')

            self.logger.info('Solving LU ..')
            LU = spla.lu_factor(d*Es+c*As)
            EG = spla.lu_solve(LU, Es.T.conj(), trans=2).T.conj()
            CG = spla.lu_solve(LU, Cs.T.conj(), trans=2).T.conj()

            self.logger.info('Constructing model ...')
            E = project(fom_mu.E, self.W, self.V)
            A = project(NumpyMatrixOperator(EG @ (a*As + b*Es)), self.W, self.V)
            B = project(NumpyMatrixOperator(kappa * EG @ Bs), self.W, None)
            C = project(NumpyMatrixOperator(kappa * CG @ Es), None, self.V)
            D = fom_mu.D - NumpyMatrixOperator(c * CG @ Bs)

            self.logger.info('Backtransforming ROM ...')
            rom = LTIModel(A, B, C, D=D, E=E, name=fom_mu.name+'_reduced').moebius_substitution(M.inverse(), sampling_time=self.fom.sampling_time)

        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)
