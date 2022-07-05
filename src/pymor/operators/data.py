# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy.ma as ma

from pymor.operators.interface import Operator
from pymor.data.base import FrequencyData, NumpySamples
from pymor.vectorarrays.numpy import NumpyVectorSpace


class LoewnerOperator(Operator):
    def __init__(self, data1, data2, source_id=None, range_id=None, solver_options=None, name=None):
        for s in (data1, data2):
            assert isinstance(s, FrequencyData)
            assert isinstance(s.x, NumpySamples) and isinstance(s.y, NumpySamples)

        assert data1.shape == data2.shape

        self.__auto_init(locals())
        self.p, self.m = data1.shape
        self.range = NumpyVectorSpace(data1.y.shape[0]*data1.n, range_id)
        self.source = NumpyVectorSpace(data1.y.shape[1]*data1.n, source_id)

    @classmethod
    def from_partitioning(cls, data, split="even-odd", source_id=None, range_id=None, solver_options=None, name=None):
        assert isinstance(data, FrequencyData)
        assert split in ("even-odd", "half-half", "random")

        mask = ma.make_mask_none(data.samples)
        if split == "even-odd":
            pass

        cls(ma.array(data.samples, mask=mask), ma.array(data.samples, mask=~mask), source_id=source_id,
            range_id=range_id, solver_options=solver_options, name=name)

    def apply(self, U, mu=None):
        assert U in self.source
