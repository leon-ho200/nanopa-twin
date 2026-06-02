from __future__ import annotations

import torch

from nanopa_twin.balance.state_space import MetabolicStateSpace, MlpProjector


def test_state_space_projection_shapes() -> None:
    projector = MetabolicStateSpace(state_dim=16, hidden_dim=12, proj_dim=10)
    state = torch.randn(5, 16)
    deltas = torch.tensor([1.0, 2.0])
    z_current, z_future = projector.project(state, deltas)
    assert z_current.shape == (5, 10)
    assert z_future.shape == (5, 2, 10)


def test_zero_horizon_recovers_current_state() -> None:
    projector = MetabolicStateSpace(state_dim=8, hidden_dim=8, proj_dim=8)
    state = torch.randn(3, 8)
    z_current, z_future = projector.project(state, torch.tensor([0.0]))
    assert torch.allclose(z_current, z_future[:, 0], atol=1e-4)


def test_mlp_projector_matches_interface() -> None:
    projector = MlpProjector(state_dim=8, hidden_dim=6, proj_dim=4)
    state = torch.randn(7, 8)
    z_current, z_future = projector.project(state, torch.tensor([1.0, 2.0, 3.0]))
    assert z_current.shape == (7, 4)
    assert z_future.shape == (7, 3, 4)


def test_projection_is_finite_over_large_horizon() -> None:
    projector = MetabolicStateSpace(state_dim=16, hidden_dim=24, proj_dim=12)
    state = torch.randn(4, 16)
    _, z_future = projector.project(state, torch.tensor([5.0, 10.0]))
    assert torch.isfinite(z_future).all()
