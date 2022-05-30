# -*- coding: utf-8 -*-

from numbers import Real
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from paddleocr import PaddleOCR


class Inference:
    __slots__ = "_exclude", "_keywords", "_min_length", "_ocr", "_top_k"

    def __init__(
        self,
        min_length: int = 1,
        top_k: int = None,
        keywords: Iterable[str] = (),
        exclude: bool = True
    ) -> None:
        self._exclude = exclude
        self._keywords = frozenset(keywords)
        self._min_length = min_length
        self._top_k = top_k

        self._ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    def __call__(self, image: npt.NDArray[np.uint8]) -> Sequence[Tuple[str, Real, List[List[Real]]]]:
        results: List[Tuple[List[List[Real]], Tuple[str, Real]]] = self._ocr.ocr(image)

        if self._top_k is not None:
            scores = frozenset(sorted((r[1][1] for r in results), reverse=True)[:self._top_k])
            results = [
                (box, (txt, score))
                for box, (txt, score) in results
                if score in scores
            ]

        return [
            (txt, score, box)
            for box, (txt, score) in results
            if len(txt) >= self._min_length and (self._exclude and txt not in self._keywords or txt in self._keywords)
        ]
