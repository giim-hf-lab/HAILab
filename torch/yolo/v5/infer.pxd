# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np


cdef class Engine:
    cpdef list infer(self, np.ndarray[np.uint8_t, ndim = 3] image)
