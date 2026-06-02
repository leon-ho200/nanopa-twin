from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch import Tensor

from nanopa_twin.timegrapher.arrays import FloatArray, IntArray, as_float, as_int


def binary_auc(scores: FloatArray, positive: IntArray) -> float:
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum = float(ranks[positive == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auroc(probabilities: FloatArray, labels: IntArray, n_class: int) -> float:
    per_class: list[float] = []
    for c in range(n_class):
        positive = (labels == c).astype(np.int64)
        value = binary_auc(probabilities[:, c], positive)
        if not np.isnan(value):
            per_class.append(value)
    if not per_class:
        return float("nan")
    return float(np.mean(per_class))


def staging_report(
    probabilities: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    n_class: int = 3,
) -> dict[str, float]:
    probs = as_float(probabilities)
    truth = as_int(labels)
    prediction = probs.argmax(axis=1)
    classes = list(range(n_class))
    return {
        "auroc": macro_auroc(probs, truth, n_class),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(truth, prediction, labels=classes)),
    }


def per_class_report(
    probabilities: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    n_class: int = 3,
) -> dict[int, dict[str, float]]:
    probs = as_float(probabilities)
    truth = as_int(labels)
    prediction = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=list(range(n_class)), zero_division=0
    )
    report: dict[int, dict[str, float]] = {}
    for c in range(n_class):
        report[c] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
        }
    return report
