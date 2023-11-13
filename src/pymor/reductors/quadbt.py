#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.core.cache import cached
from pymor.models.iosys import LTIModel
from pymor.reductors.loewner import GenericLoewnerReductor


class QuadBTReductor(GenericLoewnerReductor):
    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', conjugate=True,
                 mimo_handling='full', quadrature_rule='trapezoidal', MT=None, name=None):
        assert quadrature_rule in ('trapezoidal', )
        self.quadrature_rule = quadrature_rule
        self.MT = MT
        if MT is not None:
            s = MT.inverse()(s)
        super().__init__(s, Hs, partitioning=partitioning, ordering=ordering, conjugate=conjugate,
                         mimo_handling=mimo_handling, name=name)

    @cached
    def weights(self):
        ip, jp = self._partition_frequencies() if isinstance(self.partitioning, str) else self.partitioning
        nodes_l, nodes_r = self.s[ip], self.s[jp]

        if self.quadrature_rule == 'trapezoidal':
            weights_l = np.r_[nodes_l[1] - nodes_l[0],
                              nodes_l[2:] - nodes_l[0:-2],
                              nodes_l[-1] - nodes_l[-2]
                              ] / 2
            weights_r = np.r_[nodes_r[1] - nodes_r[0],
                              nodes_r[2:] - nodes_r[0:-2],
                              nodes_r[-1] - nodes_r[-2]
                              ] / 2
            weights_l = np.sqrt(np.abs(weights_l) / 2 / np.pi)
            weights_r = np.sqrt(np.abs(weights_r) / 2 / np.pi)
        else:
            raise NotImplementedError
        return weights_l.reshape(-1, 1), weights_r.reshape(-1, 1)

    def reduce(self, r):
        L, M, H, G = self.loewner_quadruple()
        U, S, Vh = self._loewner_svds(L)
        S1 = 1/np.sqrt(S[:r]).reshape(-1, 1)
        Z1, Y1 = S1*U[:, :r].conj().T, Vh[:r].conj().T*S1.T
        A = Z1 @ M @ Y1
        B = Z1 @ H
        C = G @ Y1
        return LTIModel.from_matrices(A, B, C)
        if self.MT is None:
            pass
        else:
            return LTIModel.from_matrices(A, B, C, D=None, E=None).moebius_substitution(self.MT.inverse())

    @cached
    def loewner_quadruple(self):
        L, Ls, V, W = super().loewner_quadruple()
        wl, wr = self.weights()
        p, m = self.Hs.shape[1:]
        wl, wr = np.repeat(wl, p, axis=0), np.repeat(wr, m, axis=0)
        return -wl*L*wr.T, -wl*Ls*wr.T, wl*V, W*wr.T

    @cached
    def _loewner_svds(self, L):
        return spla.svd(L, full_matrices=False)
