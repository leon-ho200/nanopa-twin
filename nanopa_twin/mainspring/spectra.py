from __future__ import annotations

import torch
from torch import Tensor

WAVELENGTHS_NM: tuple[int, int, int] = (1064, 1300, 1550)

MOLAR_EXTINCTION: dict[int, tuple[float, float, float]] = {
    1064: (0.97, 0.69, 0.12),
    1300: (1.05, 0.84, 0.40),
    1550: (0.74, 1.18, 0.83),
}

CHROMOPHORES: tuple[str, str, str] = ("HbO2", "Hb", "HA")


def extinction_matrix(
    wavelengths: tuple[int, ...] = WAVELENGTHS_NM,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    rows = [MOLAR_EXTINCTION[w] for w in wavelengths]
    return torch.tensor(rows, dtype=dtype)


def decompose_absorption(
    concentrations: Tensor,
    wavelengths: tuple[int, ...] = WAVELENGTHS_NM,
) -> Tensor:
    matrix = extinction_matrix(wavelengths, dtype=concentrations.dtype)
    matrix = matrix.to(concentrations.device)
    return concentrations @ matrix.transpose(0, 1)
