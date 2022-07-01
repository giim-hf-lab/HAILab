# -*- coding: utf-8 -*-
__all__ = ["Engine"]

from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import numpy.typing as npt


class Engine:
    def __init__(self,
        model: Any,
        conf_threshold: float,
        iou_threshold: float,
        classes: Iterable[str],
        exclusion: bool = False
    ) -> None: ...

    def infer(self, image: npt.NDArray[np.uint8]) -> Sequence[Tuple[str, float, Sequence[Tuple[int, int]]]]: ...
