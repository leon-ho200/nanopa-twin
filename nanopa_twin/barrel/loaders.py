from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nanopa_twin.barrel.dataset import Batch, Sample, TwinDataset, collate
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.mainspring.cohort import Subject, build_cohort


def build_subjects(config: ExperimentConfig) -> list[Subject]:
    data = config.data
    return build_cohort(
        dataset=data.dataset,
        n_subjects=data.n_subjects,
        min_visits=data.min_visits,
        max_visits=data.max_visits,
        pa_availability=data.pa_availability,
        horizons=data.horizons,
        n_biomarker=config.model.n_biomarker,
        seed=data.seed,
    )


def split_subjects(
    subjects: list[Subject], val_fraction: float
) -> tuple[list[Subject], list[Subject]]:
    n_val = max(1, int(len(subjects) * val_fraction))
    return subjects[n_val:], subjects[:n_val]


def make_dataset(config: ExperimentConfig, subjects: list[Subject]) -> TwinDataset:
    return TwinDataset(
        subjects=subjects,
        n_sensor=config.model.n_sensor,
        n_time=config.model.n_time,
    )


def make_loader(
    config: ExperimentConfig,
    subjects: list[Subject],
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader[Sample]:
    dataset = make_dataset(config, subjects)
    return DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=drop_last,
    )


def make_train_val_loaders(
    config: ExperimentConfig,
) -> tuple[DataLoader[Sample], DataLoader[Sample]]:
    subjects = build_subjects(config)
    train_subjects, val_subjects = split_subjects(subjects, config.data.val_fraction)
    train_loader = make_loader(config, train_subjects, shuffle=True, drop_last=True)
    val_loader = make_loader(config, val_subjects, shuffle=False)
    return train_loader, val_loader


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        biomarkers=batch["biomarkers"].to(device),
        pa=batch["pa"].to(device),
        pa_mask=batch["pa_mask"].to(device),
        delta_t=batch["delta_t"].to(device),
        concentrations=batch["concentrations"].to(device),
        stage=batch["stage"].to(device),
        bmd=batch["bmd"].to(device),
        progression=batch["progression"].to(device),
        length=batch["length"].to(device),
        seq_mask=batch["seq_mask"].to(device),
    )


__all__ = [
    "Batch",
    "Sample",
    "build_subjects",
    "make_dataset",
    "make_loader",
    "make_train_val_loaders",
    "move_batch",
    "split_subjects",
]
