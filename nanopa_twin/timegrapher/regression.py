from __future__ import annotations

import numpy as np
from torch import Tensor

from nanopa_twin.timegrapher.arrays import as_float


def regression_report(
    predicted: Tensor | np.ndarray, reference: Tensor | np.ndarray
) -> dict[str, float]:
    pred = as_float(predicted)
    truth = as_float(reference)
    error = pred - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    return {"rmse": rmse, "mae": mae}
