from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from nanopa_twin.barrel.dataset import Sample
from nanopa_twin.barrel.loaders import move_batch
from nanopa_twin.movement import NanoPATwin
from nanopa_twin.timegrapher.calibration import expected_calibration_error
from nanopa_twin.timegrapher.classification import per_class_report, staging_report
from nanopa_twin.timegrapher.forecasting import forecast_report
from nanopa_twin.timegrapher.regression import regression_report


def collect_predictions(
    model: NanoPATwin, loader: DataLoader[Sample], device: torch.device
) -> dict[str, torch.Tensor]:
    model.eval()
    stage_probs: list[torch.Tensor] = []
    stage_labels: list[torch.Tensor] = []
    bmd_pred: list[torch.Tensor] = []
    bmd_true: list[torch.Tensor] = []
    prog_probs: list[torch.Tensor] = []
    prog_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            outputs = model(moved)
            stage_probs.append(torch.softmax(outputs.stage_logits, dim=-1).cpu())
            stage_labels.append(moved["stage"].cpu())
            bmd_pred.append(outputs.bmd.cpu())
            bmd_true.append(moved["bmd"].cpu())
            prog_probs.append(torch.softmax(outputs.progression_logits, dim=-1).cpu())
            prog_labels.append(moved["progression"].cpu())
    return {
        "stage_probs": torch.cat(stage_probs),
        "stage_labels": torch.cat(stage_labels),
        "bmd_pred": torch.cat(bmd_pred),
        "bmd_true": torch.cat(bmd_true),
        "prog_probs": torch.cat(prog_probs),
        "prog_labels": torch.cat(prog_labels),
    }


def evaluate(
    model: NanoPATwin,
    loader: DataLoader[Sample],
    device: torch.device,
    horizons: tuple[int, ...],
    n_class: int = 3,
) -> dict[str, Any]:
    predictions = collect_predictions(model, loader, device)
    return {
        "staging": staging_report(predictions["stage_probs"], predictions["stage_labels"], n_class),
        "per_class": per_class_report(
            predictions["stage_probs"], predictions["stage_labels"], n_class
        ),
        "regression": regression_report(predictions["bmd_pred"], predictions["bmd_true"]),
        "forecast": forecast_report(
            predictions["prog_probs"], predictions["prog_labels"], horizons, n_class
        ),
        "ece": expected_calibration_error(predictions["stage_probs"], predictions["stage_labels"]),
    }
