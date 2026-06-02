from __future__ import annotations

import torch

from nanopa_twin.barrel.dataset import Batch
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.movement import build_model


def test_model_forward_shapes(config: ExperimentConfig, tiny_batch: Batch) -> None:
    model = build_model(config)
    outputs = model(tiny_batch)
    batch_size = tiny_batch["seq_mask"].shape[0]
    n_horizon = len(config.data.horizons)
    assert outputs.stage_logits.shape == (batch_size, config.model.n_class)
    assert outputs.bmd.shape == (batch_size,)
    assert outputs.progression_logits.shape == (
        batch_size,
        n_horizon,
        config.model.n_class,
    )


def test_concentration_outputs_match_present_signals(
    config: ExperimentConfig, tiny_batch: Batch
) -> None:
    model = build_model(config)
    outputs = model(tiny_batch)
    present = int(tiny_batch["pa_mask"].sum().item())
    assert outputs.pred_concentrations.shape == (present, 3)
    assert outputs.target_concentrations.shape == (present, 3)


def test_infer_subject_returns_probabilities(config: ExperimentConfig, tiny_batch: Batch) -> None:
    model = build_model(config)
    result = model.infer_subject(tiny_batch)
    stage = result["stage"]
    assert torch.allclose(stage.sum(dim=-1), torch.ones(stage.shape[0]), atol=1e-5)
    assert torch.all(result["progression"] >= 0.0)
