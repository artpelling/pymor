import numpy as np
import numpy.ma as ma

from pymor.core.base import ImmutableObject
from pymor.models.transfer_function import TransferFunction


class GenericSamples(ImmutableObject):
    samples = None
    mask = ma.nomask
    n = None
    shape = None

    def __call__(self, i):
        if self.mask is ma.nomask:
            return self.samples[i]
        else:
            return self.samples[~self.mask][i]

    def __iter__(self):
        for i in range(self.n):
            yield self(i)


class NumpySamples(GenericSamples):
    def __init__(self, samples, mask=ma.nomask):
        assert isinstance(samples, np.ndarray)
        assert samples.ndim <= 3

        if mask is not ma.nomask:
            mask = np.asarray(mask, dtype=bool)
            assert mask.ndim == 1 and mask.shape[0] == samples.shape[0]
            mask.setflags(write=False)

        if samples.ndim < 3:
            samples = np.expand_dims(samples, tuple(range(2, samples.ndim - 1, -1)))
        samples.setflags(write=False)

        self.__auto_init(locals())
        self.n = samples[~mask].shape[0]
        self.shape = samples.shape[1:]


class TransferFunctionSampler(GenericSamples):
    def __init__(self, w, tf):
        assert isinstance(w, GenericSamples)
        assert isinstance(tf, TransferFunction)
        assert w.shape == (1, 1)

        self.n = w.n
        self.shape = (tf.dim_output, tf.dim_input)
        self.__auto_init(locals())

    def __call__(self, i):
        return self.tf.eval_tf(self.w(i))


class GenericData(NumpySamples):
    def __init__(self, x, y):
        assert isinstance(x, GenericSamples)
        assert isinstance(y, GenericSamples)
        assert y.mask is ma.nomask
        assert x.shape == (1, 1)
        assert x.n == y.n

        self.__auto_init(locals())
        super().__init__(y.samples)

    def __call__(self, i):
        return ma.masked if (ma.is_masked(self.x(i)) or ma.is_masked(self.y(i))) else (self.x(i), self.y(i))


class TimeData(GenericData):
    def __init__(self, x, y):
        assert np.isrealobj(x)
        super().__init__(x, y)


class FrequencyData(GenericData):
    @classmethod
    def from_transfer_function(cls, w, tf):
        assert isinstance(w, GenericSamples)
        assert isinstance(tf, TransferFunction)
        return cls(w, TransferFunctionSampler(w, tf))
