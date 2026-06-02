from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from nanopa_twin.barrel.dataset import Batch, TwinDataset, collate
from nanopa_twin.barrel.loaders import build_subjects
from nanopa_twin.caliber.presets import load_preset
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.regulator.seeding import set_seed


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def config() -> ExperimentConfig:
    return load_preset("_smoke")


@pytest.fixture
def tiny_batch(config: ExperimentConfig) -> Batch:
    set_seed(0)
    subjects = build_subjects(config)[:6]
    dataset = TwinDataset(subjects, config.model.n_sensor, config.model.n_time)
    samples = [dataset[i] for i in range(len(subjects))]
    return collate(samples)


@pytest.fixture
def tiny_labels_and_probs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    logits = rng.standard_normal((labels.size, 3))
    logits[np.arange(labels.size), labels] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs, labels


def with_overrides(config: ExperimentConfig, **model_kwargs: object) -> ExperimentConfig:
    model = dataclasses.replace(config.model, **model_kwargs)
    return dataclasses.replace(config, model=model)
