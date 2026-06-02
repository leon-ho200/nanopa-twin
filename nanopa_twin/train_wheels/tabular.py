from __future__ import annotations

from torch import Tensor, nn


class TabularEmbedding(nn.Module):
    def __init__(self, n_biomarker: int, embed_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(n_biomarker, embed_dim)
        self.activation = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, biomarkers: Tensor) -> Tensor:
        hidden = self.activation(self.input(biomarkers))
        embedded: Tensor = self.norm(hidden + self.mlp(hidden))
        return embedded
