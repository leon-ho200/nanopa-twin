from __future__ import annotations

import torch

from nanopa_twin.barrel.dataset import Batch
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.complications.objectives import CompositeObjective
from nanopa_twin.movement import build_model


def test_every_parameter_receives_gradient(config: ExperimentConfig, tiny_batch: Batch) -> None:
    model = build_model(config)
    objective = CompositeObjective(config.loss, config.model.use_spectral_prior)
    outputs = model(tiny_batch)
    total, components = objective(
        outputs, tiny_batch["stage"], tiny_batch["bmd"], tiny_batch["progression"]
    )
    torch.autograd.backward(total)
    assert torch.isfinite(total)
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name


def test_loss_components_are_present(config: ExperimentConfig, tiny_batch: Batch) -> None:
    model = build_model(config)
    objective = CompositeObjective(config.loss, config.model.use_spectral_prior)
    outputs = model(tiny_batch)
    _, components = objective(
        outputs, tiny_batch["stage"], tiny_batch["bmd"], tiny_batch["progression"]
    )
    assert set(components) == {"cls", "reg", "prog", "spec", "total"}
    for value in components.values():
        assert torch.isfinite(value)
