from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from nanopa_twin.timegrapher.classification import binary_auc, macro_auroc, staging_report
from nanopa_twin.timegrapher.forecasting import concordance_index
from nanopa_twin.timegrapher.regression import regression_report
from nanopa_twin.timegrapher.resampling import bootstrap_ci
from nanopa_twin.timegrapher.significance import delong_test, paired_pvalue


def test_binary_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.4).astype(np.int64)
    mine = binary_auc(scores, labels)
    reference = roc_auc_score(labels, scores)
    assert abs(mine - reference) < 1e-9


def test_macro_auroc_perfect_separation() -> None:
    probs = np.eye(3)[np.array([0, 1, 2, 0, 1, 2])]
    labels = np.array([0, 1, 2, 0, 1, 2])
    assert abs(macro_auroc(probs, labels, 3) - 1.0) < 1e-9


def test_concordance_index_perfect_and_random() -> None:
    risk = np.array([0.1, 0.4, 0.9])
    outcome = np.array([0.0, 1.0, 2.0])
    assert concordance_index(risk, outcome) == 1.0
    assert concordance_index(-risk, outcome) == 0.0


def test_regression_report_known_values() -> None:
    predicted = np.array([1.0, 2.0, 3.0])
    reference = np.array([1.0, 2.0, 4.0])
    report = regression_report(predicted, reference)
    assert abs(report["mae"] - 1.0 / 3.0) < 1e-9
    assert abs(report["rmse"] - np.sqrt(1.0 / 3.0)) < 1e-9


def test_staging_report_keys(tiny_labels_and_probs: tuple[np.ndarray, np.ndarray]) -> None:
    probs, labels = tiny_labels_and_probs
    report = staging_report(probs, labels, 3)
    assert set(report) == {"auroc", "accuracy", "macro_f1", "kappa"}
    assert 0.0 <= report["accuracy"] <= 1.0


def test_delong_pvalue_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(120) < 0.5).astype(np.int64)
    good = labels + 0.3 * rng.standard_normal(120)
    weak = 0.1 * rng.standard_normal(120)
    auc_a, auc_b, p = delong_test(labels, good, weak)
    assert auc_a > auc_b
    assert 0.0 <= p <= 1.0


def test_bootstrap_interval_is_ordered() -> None:
    rng = np.random.default_rng(2)
    labels = (rng.random(150) < 0.4).astype(np.int64)
    scores = (labels + 0.5 * rng.standard_normal(150)).reshape(-1, 1)

    def metric(probabilities: np.ndarray, truth: np.ndarray) -> float:
        return binary_auc(probabilities[:, 0], truth)

    mean, lo, hi = bootstrap_ci(scores, labels, metric, n_boot=200, seed=3)
    assert lo <= mean <= hi


def test_paired_pvalue_detects_difference() -> None:
    a = np.array([0.91, 0.89, 0.93, 0.90, 0.92])
    b = np.array([0.80, 0.78, 0.74, 0.82, 0.71])
    assert paired_pvalue(a, b) < 0.05
