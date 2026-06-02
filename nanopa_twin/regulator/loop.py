from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from nanopa_twin.barrel.loaders import (
    build_subjects,
    make_loader,
    make_train_val_loaders,
    move_batch,
)
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.regulator.seeding import set_seed
from nanopa_twin.regulator.trainer import Trainer
from nanopa_twin.timegrapher.report import evaluate
from nanopa_twin.train_wheels.photoacoustic import PhotoacousticEncoder

logger = logging.getLogger("nanopa_twin.regulator")


def _pretrain_config(config: ExperimentConfig) -> ExperimentConfig:
    data = dataclasses.replace(
        config.data,
        dataset="pa_bone_sim",
        pa_availability=1.0,
        n_subjects=min(config.data.n_subjects, 128),
    )
    return dataclasses.replace(config, data=data)


def pretrain_encoder(
    config: ExperimentConfig, device: torch.device
) -> dict[str, torch.Tensor] | None:
    if not config.model.use_photoacoustic or config.optim.pae_pretrain_epochs <= 0:
        return None
    pre_config = _pretrain_config(config)
    subjects = build_subjects(pre_config)
    loader = make_loader(pre_config, subjects, shuffle=True, drop_last=True)
    encoder = PhotoacousticEncoder(
        config.model.n_wavelength,
        config.model.n_sensor,
        config.model.embed_dim,
        config.model.encoder_channels,
        config.model.encoder_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.optim.lr)
    for epoch in range(config.optim.pae_pretrain_epochs):
        encoder.train()
        for batch in loader:
            moved = move_batch(batch, device)
            mask = moved["pa_mask"].reshape(-1) > 0
            index = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if index.numel() < 2:
                continue
            signals = moved["pa"].reshape(-1, *moved["pa"].shape[2:])[index]
            targets = moved["concentrations"].reshape(-1, 3)[index]
            predicted = encoder.concentrations(encoder(signals))
            loss = F.mse_loss(predicted, targets)
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
        logger.info("pae pretrain epoch %d done", epoch)
    return encoder.state_dict()


def run_experiment(
    config: ExperimentConfig, device: torch.device, seed: int
) -> tuple[Trainer, dict[str, Any]]:
    set_seed(seed)
    pretrained = pretrain_encoder(config, device)
    train_loader, val_loader = make_train_val_loaders(config)
    trainer = Trainer(config, device)
    if pretrained is not None and trainer.model.encoder is not None:
        trainer.model.encoder.load_state_dict(pretrained)
    trainer.fit(train_loader, val_loader)
    report = evaluate(trainer.model, val_loader, device, config.data.horizons, config.model.n_class)
    return trainer, report


def cross_validate(config: ExperimentConfig, device: torch.device) -> dict[str, dict[str, float]]:
    auroc: list[float] = []
    accuracy: list[float] = []
    macro_f1: list[float] = []
    for seed in config.seeds:
        _, report = run_experiment(config, device, seed)
        auroc.append(report["staging"]["auroc"])
        accuracy.append(report["staging"]["accuracy"])
        macro_f1.append(report["staging"]["macro_f1"])
    return {
        "auroc": {"mean": float(np.mean(auroc)), "std": float(np.std(auroc))},
        "accuracy": {"mean": float(np.mean(accuracy)), "std": float(np.std(accuracy))},
        "macro_f1": {"mean": float(np.mean(macro_f1)), "std": float(np.std(macro_f1))},
    }
