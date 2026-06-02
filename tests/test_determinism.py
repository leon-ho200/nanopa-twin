from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nanopa_twin.barrel.dataset import TwinDataset, collate
from nanopa_twin.barrel.loaders import build_subjects
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.regulator.seeding import set_seed
from nanopa_twin.regulator.trainer import Trainer


def _run(config: ExperimentConfig) -> list[float]:
    set_seed(123)
    subjects = build_subjects(config)[:8]
    dataset = TwinDataset(subjects, config.model.n_sensor, config.model.n_time)
    loader: DataLoader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)
    trainer = Trainer(config, torch.device("cpu"))
    return trainer.train_steps(loader, 5)


def test_seeded_training_is_reproducible(config: ExperimentConfig) -> None:
    first = _run(config)
    second = _run(config)
    assert first == second


def test_cohort_generation_is_reproducible(config: ExperimentConfig) -> None:
    a = build_subjects(config)
    b = build_subjects(config)
    assert [s.visits[-1].stage for s in a] == [s.visits[-1].stage for s in b]
    assert [s.progression for s in a] == [s.progression for s in b]
