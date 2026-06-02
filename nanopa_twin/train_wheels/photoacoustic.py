from __future__ import annotations

from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        activated: Tensor = self.activation(x + residual)
        return activated


class PhotoacousticEncoder(nn.Module):
    def __init__(
        self,
        n_wavelength: int,
        n_sensor: int,
        embed_dim: int,
        channels: int,
        blocks: int,
    ) -> None:
        super().__init__()
        in_channels = n_wavelength * n_sensor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(blocks))
        self.project = nn.Linear(channels, embed_dim)
        self.spectral = nn.Linear(embed_dim, 3)
        self.softplus = nn.Softplus()

    def forward(self, pa: Tensor) -> Tensor:
        n, n_wave, n_sensor, n_time = pa.shape
        x = pa.reshape(n, n_wave * n_sensor, n_time)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=-1)
        embedding: Tensor = self.project(pooled)
        return embedding

    def concentrations(self, embedding: Tensor) -> Tensor:
        values: Tensor = self.softplus(self.spectral(embedding))
        return values
