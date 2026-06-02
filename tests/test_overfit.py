from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nanopa_twin.barrel.dataset import TwinDataset, collate
from nanopa_twin.barrel.loaders import build_subjects
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.regulator.seeding import set_seed
from nanopa_twin.regulator.trainer import Trainer


def test_single_batch_overfits(config: ExperimentConfig) -> None:
    set_seed(0)
    subjects = build_subjects(config)[:8]
    dataset = TwinDataset(subjects, config.model.n_sensor, config.model.n_time)
    loader: DataLoader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)
    trainer = Trainer(config, torch.device("cpu"))
    losses = trainer.train_steps(loader, 60)
    assert losses[-1] < losses[0] * 0.6
