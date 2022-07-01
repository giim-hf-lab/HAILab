# -*- coding: utf-8 -*-
__all__ = ["scale_coords"]

import numpy as np
cimport numpy as np


cpdef void scale_coords(
    (size_t, size_t) reshaped_size,
    np.ndarray[np.float32_t, ndim = 2] coords,
    (size_t, size_t) original_size
):
    cdef double r0, r1, o0, o1
    r0, r1 = reshaped_size
    o0, o1 = original_size

    cdef double gain = min(r0 / o0, r1 / o1), p0 = (r0 - gain * o0) / 2, p1 = (r1 - gain * o1) / 2

    coords[:, [0, 2]] -= p0
    coords[:, [1, 3]] -= p1
    coords /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, o1)
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, o0)
