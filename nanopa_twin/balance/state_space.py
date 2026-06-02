from __future__ import annotations

import torch
from torch import Tensor, nn

from nanopa_twin.balance.hippo import hippo_legs_matrix


class Projector(nn.Module):
    proj_dim: int

    def project(self, state: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class MetabolicStateSpace(Projector):
    def __init__(self, state_dim: int, hidden_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.lift = nn.Linear(state_dim, hidden_dim)
        self.state_matrix = nn.Parameter(hippo_legs_matrix(hidden_dim))
        self.input_matrix = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.readout = nn.Linear(hidden_dim, proj_dim)

    def project(self, state: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        h_current = self.lift(state)
        z_current = self.readout(h_current)
        identity = torch.eye(self.hidden_dim, dtype=state.dtype, device=state.device)
        futures: list[Tensor] = []
        for delta in deltas:
            transition = torch.linalg.matrix_exp(self.state_matrix * delta)
            driver = torch.linalg.solve(self.state_matrix, transition - identity)
            input_term = driver @ self.input_matrix
            h_future = h_current @ transition.transpose(0, 1) + state @ input_term.transpose(0, 1)
            futures.append(self.readout(h_future))
        z_future = torch.stack(futures, dim=1)
        return z_current, z_future


class MlpProjector(Projector):
    def __init__(self, state_dim: int, hidden_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.current = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.future = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def project(self, state: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        z_current = self.current(state)
        futures: list[Tensor] = []
        for delta in deltas:
            stamp = delta.view(1, 1).expand(state.shape[0], 1)
            futures.append(self.future(torch.cat([state, stamp], dim=-1)))
        z_future = torch.stack(futures, dim=1)
        return z_current, z_future


def build_projector(projection: str, state_dim: int, hidden_dim: int, proj_dim: int) -> Projector:
    if projection == "state_space":
        return MetabolicStateSpace(state_dim, hidden_dim, proj_dim)
    if projection == "mlp":
        return MlpProjector(state_dim, hidden_dim, proj_dim)
    raise ValueError(f"unknown projection {projection!r}")
