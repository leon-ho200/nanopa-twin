from __future__ import annotations

import numpy as np
from torch import Tensor

from nanopa_twin.timegrapher.arrays import as_float, as_int


def expected_calibration_error(
    probabilities: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    n_bins: int = 15,
) -> float:
    probs = as_float(probabilities)
    truth = as_int(labels)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == truth).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(confidence.size)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi)
        count = float(mask.sum())
        if count == 0.0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        error += (count / total) * abs(avg_conf - avg_acc)
    return error
