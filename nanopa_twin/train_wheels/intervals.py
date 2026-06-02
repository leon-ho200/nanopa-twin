from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class IntervalEmbedding(nn.Module):
    frequencies: Tensor

    def __init__(self, interval_dim: int, scale_months: float = 12.0) -> None:
        super().__init__()
        if interval_dim % 2 != 0:
            raise ValueError("interval_dim must be even")
        self.interval_dim = interval_dim
        self.scale_months = scale_months
        half = interval_dim // 2
        frequencies = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
        self.register_buffer("frequencies", frequencies)
        self.project = nn.Linear(interval_dim, interval_dim)

    def forward(self, delta_t: Tensor) -> Tensor:
        scaled = (delta_t / self.scale_months).unsqueeze(-1)
        angles = scaled * self.frequencies
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        projected: Tensor = self.project(encoding)
        return projected
