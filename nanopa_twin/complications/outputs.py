from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutputs:
    stage_logits: Tensor
    bmd: Tensor
    progression_logits: Tensor
    pred_concentrations: Tensor
    target_concentrations: Tensor
