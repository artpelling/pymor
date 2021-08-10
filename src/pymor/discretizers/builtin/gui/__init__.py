# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.vectorarrays.interface import VectorArray


def vmin_vmax_numpy(U, separate_colorbars, rescale_colorbars):
    if separate_colorbars:
        if rescale_colorbars:
            vmins = tuple(np.min(u[0]) for u in U)
            vmaxs = tuple(np.max(u[0]) for u in U)
        else:
            vmins = tuple(np.min(u) for u in U)
            vmaxs = tuple(np.max(u) for u in U)
    else:
        if rescale_colorbars:
            vmins = (min(np.min(u[0]) for u in U),) * len(U)
            vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
        else:
            vmins = (min(np.min(u) for u in U),) * len(U)
            vmaxs = (max(np.max(u) for u in U),) * len(U)
    return vmins, vmaxs


def vmin_vmax_vectorarray(U, separate_colorbars, rescale_colorbars):
    """
    Parameters
    ----------
    separate_colorbars
        iff True, min/max are taken per element of the U tuple

    rescale_colorbars
        iff False, min/max are the same for all indices for all elements of the U tuple
    """
    assert isinstance(U, tuple)
    limits = {}
    ind_count = len(U[0])
    tuple_size = len(U)
    mins, maxs = [None] * ind_count
    for ind in range(ind_count):
        mins[ind] = (np.min(u[ind]) for u in U)
        maxs[ind] = (np.max(u[ind]) for u in U)

    for ind in range(ind_count):
        if rescale_colorbars:
            if separate_colorbars:
                limits[ind] = tuple(mins[ind]), tuple(maxs[ind])
            else:
                limits[ind] = min(mins) * tuple_size, max(maxs) * tuple_size
        else:
            if separate_colorbars:
                limits[ind] = (tuple(),
                               tuple(np.max(u[ind]) for u in U))
            else:
                limits[ind] = ((min(np.min(u) for u in U),) * len(U),
                               (max(np.max(u[ind]) for u in U),) * len(U))
    return limits