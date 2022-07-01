# -*- coding: utf-8 -*-
# cython: language_level = 3, c_string_type = str, c_string_encoding = utf-8

import numpy as np
cimport numpy as np
import torch.nn

from utility.processing cimport scale_coords

from ._impl.models.yolo import Detect, Model
from ._impl.utils.augmentations import letterbox
from ._impl.utils.general import non_max_suppression

cdef object _CPU = torch.device("cpu"), _CUDA = torch.device("cuda:0")

cdef frozenset _YOLO_INPLACE_LAYERS = frozenset({
    torch.nn.Hardswish,
    torch.nn.LeakyReLU,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.SiLU,
    Model
})


cdef class Engine:
    cdef readonly double _conf_threshold, _iou_threshold
    cdef readonly object _model
    cdef readonly np.ndarray[np.int64_t, ndim = 1] _classes
    cdef readonly list _name_mappings

    def __cinit__(self,
        object model,
        double conf_threshold,
        double iou_threshold,
        object classes,
        bint exclusion = False
    ) -> None:
        cdef dict checkpoints = torch.load(model, map_location=_CPU)
        model = (checkpoints.get("ema", None) or checkpoints["model"]).float().fuse().eval()

        cdef type t
        for m in model.modules():
            t = type(m)
            if t in _YOLO_INPLACE_LAYERS:
                m.inplace = True
            elif t is Detect:
                m.inplace = True
                if not isinstance(m.anchor_grid, list):
                    m.anchor_grid = [torch.zeros(1)] * m.nl
            elif t is torch.nn.Upsample and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None

        model = self._model = model.to(_CUDA).half()
        model(torch.zeros(1, 3, 640, 640, dtype=torch.half, device=_CUDA))

        cdef list name_mappings

        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        name_mappings = self._name_mappings = getattr(model, "module", model).names

        cdef dict index_mapping = {
            name: i
            for i, name in enumerate(name_mappings)
        }
        self._classes = np.ascontiguousarray(sorted({
            index_mapping[name]
            for name in (set(name_mappings).difference(classes) if exclusion else classes)
            if name in index_mapping
        }), dtype=np.int64)

    cpdef list infer(self, np.ndarray[np.uint8_t, ndim = 3] image):
        cdef list retval = []

        cdef np.ndarray[np.uint8_t, ndim = 3] reshaped = np.ascontiguousarray(letterbox(
            image,
            (640, 640),
            stride=32,
            auto=True
        )[0].transpose(2, 0, 1)[::-1])

        cdef np.ndarray[np.float32_t, ndim = 2] result
        cdef np.ndarray[np.float32_t, ndim = 1] det
        cdef size_t o0 = image.shape[0], o1 = image.shape[1], r0 = reshaped.shape[1], r1 = reshaped.shape[2]
        for tensor in non_max_suppression(
            self._model((torch.from_numpy(reshaped).to(_CUDA).half() / 255).unsqueeze(0))[0],
            self._conf_threshold,
            self._iou_threshold,
            self._classes
        ):
            result = tensor.float().to(_CPU).numpy()
            scale_coords((r0, r1), result[:, :4], (o0, o1))
            for det in result:
                retval.append((
                    self._name_mappings[int(det[5])],
                    det[4],
                    ((int(det[0]), int(det[1])), (int(det[2]), int(det[3])))
                ))

        return retval
