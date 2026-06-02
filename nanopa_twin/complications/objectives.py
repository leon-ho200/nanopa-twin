from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanopa_twin.caliber.specs import LossConfig
from nanopa_twin.complications.outputs import ModelOutputs


def focal_loss(logits: Tensor, target: Tensor, gamma: float) -> Tensor:
    cross_entropy = F.cross_entropy(logits, target, reduction="none")
    probability = torch.exp(-cross_entropy)
    return ((1.0 - probability) ** gamma * cross_entropy).mean()


def spectral_prior_loss(predicted: Tensor, reference: Tensor) -> Tensor:
    if predicted.numel() == 0:
        return predicted.new_zeros(())
    return F.mse_loss(predicted, reference)


def progression_loss(logits: Tensor, target: Tensor) -> Tensor:
    n_class = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, n_class), target.reshape(-1))


class CompositeObjective(nn.Module):
    def __init__(self, config: LossConfig, use_spectral_prior: bool) -> None:
        super().__init__()
        self.config = config
        self.use_spectral_prior = use_spectral_prior

    def forward(
        self, outputs: ModelOutputs, stage: Tensor, bmd: Tensor, progression: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cls = focal_loss(outputs.stage_logits, stage, self.config.focal_gamma)
        reg = F.l1_loss(outputs.bmd, bmd)
        prog = progression_loss(outputs.progression_logits, progression)
        if self.use_spectral_prior:
            spec = spectral_prior_loss(outputs.pred_concentrations, outputs.target_concentrations)
        else:
            spec = outputs.stage_logits.new_zeros(())
        total = (
            self.config.lambda_cls * cls
            + self.config.lambda_reg * reg
            + self.config.lambda_prog * prog
            + self.config.lambda_spec * spec
        )
        components = {"cls": cls, "reg": reg, "prog": prog, "spec": spec, "total": total}
        return total, components
