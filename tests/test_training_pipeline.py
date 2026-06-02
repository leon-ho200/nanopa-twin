from __future__ import annotations

from pathlib import Path

import torch

from nanopa_twin.barrel.loaders import make_train_val_loaders
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.regulator.loop import run_experiment
from nanopa_twin.regulator.seeding import set_seed
from nanopa_twin.regulator.trainer import Trainer
from nanopa_twin.timegrapher.report import evaluate


def test_two_step_training_reduces_loss(config: ExperimentConfig, device: torch.device) -> None:
    set_seed(0)
    train_loader, _ = make_train_val_loaders(config)
    trainer = Trainer(config, device)
    losses = trainer.train_steps(train_loader, 2)
    assert len(losses) == 2
    assert losses[1] <= losses[0] + 1e-3


def test_end_to_end_experiment_produces_report(
    config: ExperimentConfig, device: torch.device
) -> None:
    trainer, report = run_experiment(config, device, seed=0)
    assert set(report) == {"staging", "per_class", "regression", "forecast", "ece"}
    for horizon in config.data.horizons:
        assert horizon in report["forecast"]
    assert 0.0 <= report["staging"]["accuracy"] <= 1.0


def test_checkpoint_round_trip(
    config: ExperimentConfig, device: torch.device, tmp_path: Path
) -> None:
    set_seed(0)
    train_loader, val_loader = make_train_val_loaders(config)
    trainer = Trainer(config, device)
    trainer.train_steps(train_loader, 2)
    checkpoint = tmp_path / "model.pt"
    trainer.save_checkpoint(checkpoint, epoch=1, seed=0)
    restored = Trainer(config, device)
    payload = restored.load_checkpoint(checkpoint)
    assert payload["seed"] == 0
    assert payload["config"]["name"] == config.name
    before = evaluate(trainer.model, val_loader, device, config.data.horizons)
    after = evaluate(restored.model, val_loader, device, config.data.horizons)
    assert abs(before["regression"]["rmse"] - after["regression"]["rmse"]) < 1e-5
