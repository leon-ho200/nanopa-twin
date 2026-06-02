from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from nanopa_twin.barrel.dataset import Batch, Sample
from nanopa_twin.barrel.loaders import move_batch
from nanopa_twin.caliber.presets import to_mapping
from nanopa_twin.caliber.specs import ExperimentConfig
from nanopa_twin.complications.objectives import CompositeObjective
from nanopa_twin.movement import NanoPATwin, build_model

logger = logging.getLogger("nanopa_twin.regulator")


class Trainer:
    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model: NanoPATwin = build_model(config).to(device)
        self.objective = CompositeObjective(config.loss, config.model.use_spectral_prior).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )

    def _step(self, batch: Batch) -> Tensor:
        moved = move_batch(batch, self.device)
        outputs = self.model(moved)
        total, _ = self.objective(outputs, moved["stage"], moved["bmd"], moved["progression"])
        loss: Tensor = total
        return loss

    def train_epoch(self, loader: DataLoader[Sample]) -> float:
        self.model.train()
        running = 0.0
        count = 0
        for batch in loader:
            self.optimizer.zero_grad()
            total = self._step(batch)
            torch.autograd.backward(total)
            if self.config.optim.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
            self.optimizer.step()
            running += float(total.item())
            count += 1
        return running / max(1, count)

    def validate(self, loader: DataLoader[Sample]) -> float:
        self.model.eval()
        running = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                running += float(self._step(batch).item())
                count += 1
        return running / max(1, count)

    def fit(self, train_loader: DataLoader[Sample], val_loader: DataLoader[Sample]) -> list[float]:
        best = float("inf")
        best_state: dict[str, Tensor] | None = None
        wait = 0
        history: list[float] = []
        for epoch in range(self.config.optim.max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            history.append(val_loss)
            logger.info("epoch %d train=%.4f val=%.4f", epoch, train_loss, val_loss)
            if val_loss < best - 1e-6:
                best = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.config.optim.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def train_steps(self, loader: DataLoader[Sample], n_steps: int) -> list[float]:
        self.model.train()
        losses: list[float] = []
        iterator = iter(loader)
        for _ in range(n_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            self.optimizer.zero_grad()
            total = self._step(batch)
            torch.autograd.backward(total)
            self.optimizer.step()
            losses.append(float(total.item()))
        return losses

    def save_checkpoint(self, path: Path, epoch: int, seed: int) -> None:
        payload: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": to_mapping(self.config),
            "seed": seed,
            "epoch": epoch,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        payload: dict[str, Any] = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        return payload
