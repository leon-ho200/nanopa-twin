from __future__ import annotations

import numpy as np
from torch import Tensor

from nanopa_twin.timegrapher.arrays import as_float, as_int
from nanopa_twin.timegrapher.classification import binary_auc


def concordance_index(risk: Tensor | np.ndarray, outcome: Tensor | np.ndarray) -> float:
    risk_values = as_float(risk)
    outcome_values = as_float(outcome)
    higher = outcome_values[:, None] > outcome_values[None, :]
    comparable = float(higher.sum())
    if comparable == 0.0:
        return float("nan")
    diff = risk_values[:, None] - risk_values[None, :]
    concordant = float((higher & (diff > 0)).sum())
    ties = float((higher & (diff == 0)).sum())
    return (concordant + 0.5 * ties) / comparable


def forecast_report(
    progression_probabilities: Tensor | np.ndarray,
    progression_labels: Tensor | np.ndarray,
    horizons: tuple[int, ...],
    n_class: int = 3,
) -> dict[int, dict[str, float]]:
    probs = as_float(progression_probabilities)
    labels = as_int(progression_labels)
    ordinal = np.arange(n_class, dtype=np.float64)
    report: dict[int, dict[str, float]] = {}
    for h, months in enumerate(horizons):
        probs_h = probs[:, h, :]
        labels_h = labels[:, h]
        positive = (labels_h >= 1).astype(np.int64)
        score = 1.0 - probs_h[:, 0]
        risk = probs_h @ ordinal
        report[months] = {
            "auroc": binary_auc(score, positive),
            "c_index": concordance_index(risk, labels_h.astype(np.float64)),
        }
    return report
