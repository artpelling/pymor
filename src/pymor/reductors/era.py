# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyHankelOperator, NumpyMatrixOperator


class ERAReductor(BasicObject):
    def __init__(self, data, sampling_time):
        assert sampling_time > 0

        self.sampling_time = sampling_time
        self.H = NumpyHankelOperator(data)
        self._input_projector_cache = None
        self._output_projector_cache = None
        self._SVD_cache = {}

    def _output_projector(self, l1):
        if self._output_projector_cache is None:
            W1, s1, _ = spla.svd(np.hstack(self.H.markov_parameters), full_matrices=False)
            self._output_projector_cache = (W1, s1)
        W1, _ = self._output_projector_cache
        return W1[:, :l1]

    def _input_projector(self, l2):
        if self._input_projector_cache is None:
            _, s2, W2 = spla.svd(np.vstack(self.H.markov_parameters), full_matrices=False)
            self._input_projector_cache = (W2.conj().T, s2)
        W2, _ = self._input_projector_cache
        return W2[:, :l2]

    def _SVD(self, r, l1=None, l2=None):
        key = f'{(l1, l2)}'
        if key in self._SVD_cache.keys():
            U, sv, Vh = self._SVD_cache[key]
        else:
            # project Markov parameters
            H = self.H
            if l1:
                W1 = self._output_projector(l1)
                H = NumpyHankelOperator(W1.conj().T @ H.markov_parameters)
            if l2:
                W2 = self._input_projector(l2)
                H = NumpyHankelOperator(H.markov_parameters @ W2)

            # compute SVD
            U, sv, Vh = spla.svd(to_matrix(H), full_matrices=False)

            # cache results
            self._SVD_cache[key] = (U, sv, Vh)

        return U[:, :r], sv[:r], Vh[:r]

    def reduce(self, r=None, l1=None, l2=None, tol=None):
        assert r is not None or tol is not None
        _, p, m = self.H.markov_parameters.shape
        assert l1 is None or isinstance(l1, int) and l1 <= p
        assert l2 is None or isinstance(l2, int) and l2 <= m
        assert r is None or 0 < r <= min(self.H.range.dim * (l1 or p) / p, self.H.source.dim * (l2 or m) / m)

        U, hsv, Vh = self._SVD(r, l1=l1, l2=l2)

        # construct realization
        sqs = np.sqrt(hsv)
        Zo = U * sqs.reshape(1, -1)
        A = NumpyMatrixOperator(spla.pinv(Zo[: -(l1 or p)]) @ Zo[(l1 or p):])
        B = NumpyMatrixOperator(sqs.reshape(-1, 1) * Vh[:, :(l2 or m)])
        C = NumpyMatrixOperator(Zo[:(l1 or p)])

        # tangential backprojection
        if l1:
            C = project(C, source_basis=None, range_basis=C.range.from_numpy(self._output_projector(l1)))
        if l2:
            B = project(B, source_basis=B.source.from_numpy(self._input_projector(l2)), range_basis=None)

        presets = {'hsv': hsv, 'o_dense': np.diag(hsv), 'c_dense': np.diag(hsv)}
        return LTIModel(A, B, C, sampling_time=self.sampling_time, presets=presets)
