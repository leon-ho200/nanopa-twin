from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import torch

from nanopa_twin.barrel.loaders import make_train_val_loaders
from nanopa_twin.caliber.presets import load_preset
from nanopa_twin.regulator.loop import pretrain_encoder, run_experiment
from nanopa_twin.regulator.trainer import Trainer
from nanopa_twin.timegrapher.report import evaluate

logger = logging.getLogger("nanopa_twin.dial")


def _emit(payload: dict[str, object]) -> None:
    logger.info("%s", json.dumps(payload, indent=2, sort_keys=True))


@dataclasses.dataclass
class Fit:
    preset: str = "_smoke"
    out: Path = Path("runs")
    device: str = "cpu"
    seed: int | None = None
    epochs: int | None = None

    def run(self) -> None:
        config = load_preset(self.preset)
        if self.epochs is not None:
            config = dataclasses.replace(
                config, optim=dataclasses.replace(config.optim, max_epochs=self.epochs)
            )
        device = torch.device(self.device)
        seed = self.seed if self.seed is not None else config.seeds[0]
        trainer, report = run_experiment(config, device, seed)
        self.out.mkdir(parents=True, exist_ok=True)
        checkpoint = self.out / f"{config.name}.pt"
        trainer.save_checkpoint(checkpoint, epoch=config.optim.max_epochs, seed=seed)
        _emit({"checkpoint": str(checkpoint), "report": report})


@dataclasses.dataclass
class Evaluate:
    checkpoint: Path
    preset: str = "_smoke"
    device: str = "cpu"

    def run(self) -> None:
        config = load_preset(self.preset)
        device = torch.device(self.device)
        trainer = Trainer(config, device)
        trainer.load_checkpoint(self.checkpoint)
        _, val_loader = make_train_val_loaders(config)
        report = evaluate(
            trainer.model,
            val_loader,
            device,
            config.data.horizons,
            config.model.n_class,
        )
        _emit({"report": report})


@dataclasses.dataclass
class Forecast:
    checkpoint: Path
    preset: str = "_smoke"
    device: str = "cpu"

    def run(self) -> None:
        config = load_preset(self.preset)
        device = torch.device(self.device)
        trainer = Trainer(config, device)
        trainer.load_checkpoint(self.checkpoint)
        _, val_loader = make_train_val_loaders(config)
        report = evaluate(
            trainer.model,
            val_loader,
            device,
            config.data.horizons,
            config.model.n_class,
        )
        _emit({"forecast": report["forecast"]})


@dataclasses.dataclass
class Encode:
    preset: str = "_smoke"
    out: Path = Path("runs")
    device: str = "cpu"

    def run(self) -> None:
        config = load_preset(self.preset)
        device = torch.device(self.device)
        state = pretrain_encoder(config, device)
        self.out.mkdir(parents=True, exist_ok=True)
        target = self.out / f"{config.name}_encoder.pt"
        torch.save(state, target)
        _emit({"encoder": str(target), "trained": state is not None})


@dataclasses.dataclass
class Export:
    checkpoint: Path
    out: Path = Path("runs/exported.pt")
    preset: str = "_smoke"
    device: str = "cpu"

    def run(self) -> None:
        config = load_preset(self.preset)
        device = torch.device(self.device)
        trainer = Trainer(config, device)
        trainer.load_checkpoint(self.checkpoint)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), self.out)
        _emit({"exported": str(self.out)})
