# -*- coding: utf-8 -*-
# cython: language_level = 3, c_string_type = str, c_string_encoding = utf-8

import numpy as np
cimport numpy as np


cpdef void scale_coords(
    (size_t, size_t) reshaped_size,
    np.ndarray[np.float32_t, ndim = 2] coords,
    (size_t, size_t) original_size
)
