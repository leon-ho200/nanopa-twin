from __future__ import annotations

import numpy as np
from scipy.stats import norm, ttest_rel
from torch import Tensor

from nanopa_twin.timegrapher.arrays import FloatArray, as_float, as_int


def _midrank(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = values.size
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _fast_delong(stacked: FloatArray, n_positive: int) -> tuple[FloatArray, FloatArray]:
    m = n_positive
    n = stacked.shape[1] - m
    positive = stacked[:, :m]
    negative = stacked[:, m:]
    k = stacked.shape[0]
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(stacked[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    labels: Tensor | np.ndarray,
    scores_a: Tensor | np.ndarray,
    scores_b: Tensor | np.ndarray,
) -> tuple[float, float, float]:
    truth = as_int(labels)
    a = as_float(scores_a)
    b = as_float(scores_b)
    n_positive = int(truth.sum())
    if n_positive == 0 or n_positive == truth.size:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-truth)
    stacked = np.vstack((a, b))[:, order]
    aucs, cov = _fast_delong(stacked, n_positive)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T).item())
    if variance <= 0.0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    p_value = float(2.0 * norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p_value


def paired_pvalue(scores_a: Tensor | np.ndarray, scores_b: Tensor | np.ndarray) -> float:
    a = as_float(scores_a)
    b = as_float(scores_b)
    result = ttest_rel(a, b)
    return float(result.pvalue)
