# -*- coding: utf-8 -*-
__all__ = ["scale_coords"]

from typing import Tuple

import numpy as np
import numpy.typing as npt


def scale_coords(
    reshaped_size: Tuple[int, int],
    coords: npt.NDArray[np.float32],
    original_size: Tuple[int, int]
) -> None: ...
