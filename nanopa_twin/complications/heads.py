from __future__ import annotations

from torch import Tensor, nn


class StagingHead(nn.Module):
    def __init__(self, proj_dim: int, n_class: int) -> None:
        super().__init__()
        self.linear = nn.Linear(proj_dim, n_class)

    def forward(self, z: Tensor) -> Tensor:
        logits: Tensor = self.linear(z)
        return logits


class DensityHead(nn.Module):
    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(proj_dim, 1)

    def forward(self, z: Tensor) -> Tensor:
        value: Tensor = self.linear(z)
        return value.squeeze(-1)


class ProgressionHead(nn.Module):
    def __init__(self, proj_dim: int, n_class: int) -> None:
        super().__init__()
        self.linear = nn.Linear(proj_dim, n_class)

    def forward(self, z_future: Tensor) -> Tensor:
        logits: Tensor = self.linear(z_future)
        return logits
