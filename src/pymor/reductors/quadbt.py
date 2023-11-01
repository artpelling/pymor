#!/usr/bin/env python3

import numpy as np
import scipy.linalg as spla

from pymor.core.base import BasicObject
from pymor.core.cache import CacheableObject, cached
from pymor.models.iosys import LTIModel
from pymor.reductors.loewner import GenericLoewnerReductor, LoewnerReductor


class QuadBTReductor(GenericLoewnerReductor):
    def __init__(self, s, Hs, partitioning='even-odd', ordering='regular', conjugate=True,
                 mimo_handling='full', quadrature_rule='trapezoidal', MT=None, name=None):
        assert quadrature_rule in ('trapezoidal', )
        self.quadrature_rule = quadrature_rule
        self.MT = MT
        if MT is not None:
            s = MT(s)
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    from pymor.core.logger import set_log_levels
    from pymor.models.transforms import MoebiusTransformation
    from pymor.operators.numpy import NumpyMatrixOperator


    set_log_levels({'pymor.operators.constructions.LincombOperator': 'ERROR'})
    plt.close('all')
    n = 25
    r = 25
    lti = LTIModel.from_mat_file('/Users/pelling/data/slicot/iss.mat')

    ss = np.logspace(-1, 2, n)
    l = LoewnerReductor(ss, lti, mimo_handling='full')
    q = QuadBTReductor(ss, lti)
    rom = q.reduce(r)
    err = lti - rom

    MT = MoebiusTransformation((0,1,1,0))
    MT = MoebiusTransformation.from_points((0, 29, np.inf))
    qspa = QuadBTReductor(ss, lti, MT=MT)
    rom_spa = qspa.reduce(r)
    err_spa = lti - rom_spa

    wlim = np.array((np.min(ss), np.max(ss)))
    fig, ax = plt.subplots(figsize=(8,6))
    lti.transfer_function.mag_plot(wlim, ax=ax, dB=True)
    ax.scatter(ss, 20*np.log10(spla.norm(lti.transfer_function.bode(ss)[0], axis=(1,2))), marker='s', fc='none', ec='b')
    rom.transfer_function.mag_plot(wlim, ax=ax, dB=True, linestyle='-.', c='g')
    err.transfer_function.mag_plot(wlim, ax=ax, dB=True, linestyle=':', c='g')
    rom_spa.transfer_function.mag_plot(wlim, ax=ax, dB=True, linestyle='-.', c='r')
    err_spa.transfer_function.mag_plot(wlim, ax=ax, dB=True, linestyle=':', c='r')

    #ax.set_xlim(wlim)
    ax.set_xlim((10,100))
    ax.set_ylim((-100, -12))
    plt.savefig('/Users/pelling/projects/pymor/plot.png')

    fig, ax = plt.subplots(figsize=(8,6))
    L = l.loewner_quadruple()[0]
    ax.semilogy(spla.svdvals(L), c='b', label=f'Loewner: {np.linalg.cond(L)}')
    L = q.loewner_quadruple()[0]
    ax.semilogy(spla.svdvals(L), c='g', label=f'QuadBT: {np.linalg.cond(L)}')
    L = qspa.loewner_quadruple()[0]
    ax.semilogy(spla.svdvals(L), c='r', label=f'QuadSPA: {np.linalg.cond(L)}')
    plt.legend()
    plt.savefig('/Users/pelling/projects/pymor/svds.png')
