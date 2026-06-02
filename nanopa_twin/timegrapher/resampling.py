from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch import Tensor

from nanopa_twin.timegrapher.arrays import FloatArray, IntArray, as_float, as_int

Metric = Callable[[FloatArray, IntArray], float]


def bootstrap_ci(
    probabilities: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    metric: Metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    probs = as_float(probabilities)
    truth = as_int(labels)
    rng = np.random.default_rng(seed)
    n = truth.size
    samples: list[float] = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        value = metric(probs[index], truth[index])
        if not np.isnan(value):
            samples.append(value)
    if not samples:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(samples)
    return (
        float(array.mean()),
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )
