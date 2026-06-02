from __future__ import annotations

import time

import torch
from torch import nn

from nanopa_twin.barrel.dataset import Batch


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def time_inference(model: nn.Module, batch: Batch, repeats: int = 10) -> float:
    model.eval()
    subjects = max(1, int(batch["seq_mask"].shape[0]))
    with torch.no_grad():
        model(batch)
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        elapsed = time.perf_counter() - start
    return elapsed / repeats / subjects * 1000.0
