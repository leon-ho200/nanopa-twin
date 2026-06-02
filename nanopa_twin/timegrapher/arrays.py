from __future__ import annotations

import numpy as np
import numpy.typing as npt
from torch import Tensor

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def to_numpy(value: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def as_float(value: Tensor | np.ndarray) -> FloatArray:
    return to_numpy(value).astype(np.float64)


def as_int(value: Tensor | np.ndarray) -> IntArray:
    return to_numpy(value).astype(np.int64)
