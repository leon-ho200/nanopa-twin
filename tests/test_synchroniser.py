from __future__ import annotations

import torch

from nanopa_twin.escapement.synchroniser import CrossModalSynchroniser


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(4, 3, 32)
    dt = torch.randn(4, 3, 8)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    return obs, dt, mask


def test_gate_values_are_bounded() -> None:
    sync = CrossModalSynchroniser(obs_dim=32, state_dim=16, interval_dim=8, mode="gated")
    obs, dt, mask = _inputs()
    gates = sync.gate_trace(obs, dt, mask)
    assert gates.shape == (4, 3, 16)
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)


def test_all_modes_emit_final_state() -> None:
    obs, dt, mask = _inputs()
    for mode in ("gated", "concat", "transformer"):
        sync = CrossModalSynchroniser(obs_dim=32, state_dim=16, interval_dim=8, mode=mode)
        state = sync(obs, dt, mask)
        assert state.shape == (4, 16)
        assert torch.isfinite(state).all()


def test_padded_steps_do_not_change_recurrent_state() -> None:
    sync = CrossModalSynchroniser(obs_dim=32, state_dim=16, interval_dim=8, mode="gated")
    obs, dt, mask = _inputs()
    short_mask = mask.clone()
    state_a = sync(obs, dt, short_mask)
    obs_perturbed = obs.clone()
    obs_perturbed[mask == 0] += 5.0
    state_b = sync(obs_perturbed, dt, short_mask)
    assert torch.allclose(state_a, state_b, atol=1e-5)
