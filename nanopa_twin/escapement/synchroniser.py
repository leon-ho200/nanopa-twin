from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossModalSynchroniser(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        interval_dim: int,
        mode: str = "gated",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.mode = mode
        if mode == "gated":
            self.gate = nn.Linear(obs_dim + state_dim + interval_dim, state_dim)
            self.candidate = nn.Linear(obs_dim + state_dim, state_dim)
        elif mode == "concat":
            self.cell = nn.GRUCell(obs_dim, state_dim)
        elif mode == "transformer":
            self.input_proj = nn.Linear(obs_dim + interval_dim, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim,
                nhead=transformer_heads,
                dim_feedforward=state_dim * 2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        else:
            raise ValueError(f"unknown synchroniser mode {mode!r}")

    def _recur(self, obs: Tensor, dt_emb: Tensor, seq_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = obs.shape
        state = obs.new_zeros(batch, self.state_dim)
        gates = obs.new_zeros(batch, length, self.state_dim)
        for t in range(length):
            active = seq_mask[:, t].unsqueeze(-1)
            o = obs[:, t]
            if self.mode == "gated":
                gate_input = torch.cat([o, state, dt_emb[:, t]], dim=-1)
                g = torch.sigmoid(self.gate(gate_input))
                cand = torch.tanh(self.candidate(torch.cat([o, g * state], dim=-1)))
                updated = (1.0 - g) * state + g * cand
                gates[:, t] = g
            else:
                updated = self.cell(o, state)
            state = torch.where(active > 0, updated, state)
        return state, gates

    def _attend(self, obs: Tensor, dt_emb: Tensor, seq_mask: Tensor) -> Tensor:
        x = self.input_proj(torch.cat([obs, dt_emb], dim=-1))
        padding = seq_mask <= 0
        encoded: Tensor = self.encoder(x, src_key_padding_mask=padding)
        lengths = seq_mask.sum(dim=1).clamp(min=1.0).long() - 1
        index = lengths.view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
        return encoded.gather(1, index).squeeze(1)

    def forward(self, obs: Tensor, dt_emb: Tensor, seq_mask: Tensor) -> Tensor:
        if self.mode == "transformer":
            return self._attend(obs, dt_emb, seq_mask)
        state, _ = self._recur(obs, dt_emb, seq_mask)
        return state

    def gate_trace(self, obs: Tensor, dt_emb: Tensor, seq_mask: Tensor) -> Tensor:
        if self.mode != "gated":
            raise RuntimeError("gate_trace is only defined for the gated core")
        _, gates = self._recur(obs, dt_emb, seq_mask)
        return gates
