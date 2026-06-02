from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nanopa_twin.mainspring.spectra import WAVELENGTHS_NM, decompose_absorption


@dataclass(frozen=True)
class Microstructure:
    trabecular_spacing: float
    porosity: float
    attenuation: float


def microstructure_from_density(density: float, jitter: Tensor) -> Microstructure:
    spacing = float(0.35 + 0.9 * (1.0 - density) + 0.05 * jitter[0].item())
    porosity = float(min(0.95, max(0.05, 0.8 - 0.6 * density + 0.05 * jitter[1].item())))
    attenuation = float(2.0 + 4.0 * density + 0.2 * jitter[2].item())
    return Microstructure(spacing, porosity, attenuation)


def simulate_pa_signal(
    concentrations: Tensor,
    structure: Microstructure,
    n_sensor: int,
    n_time: int,
    generator: torch.Generator,
    wavelengths: tuple[int, ...] = WAVELENGTHS_NM,
) -> Tensor:
    device = concentrations.device
    n_wave = len(wavelengths)
    mu = decompose_absorption(concentrations, wavelengths)
    time = torch.linspace(0.0, 1.0, n_time, device=device)
    sensor_phase = torch.linspace(0.0, float(torch.pi), n_sensor, device=device)
    base_freq = 6.0 + 30.0 / structure.trabecular_spacing
    decay = structure.attenuation
    envelope = torch.exp(-decay * time)
    carrier = torch.sin(2.0 * torch.pi * base_freq * time.unsqueeze(0) + sensor_phase.unsqueeze(1))
    speckle = 1.0 + structure.porosity * torch.sin(
        2.0 * torch.pi * (base_freq * 0.5) * time.unsqueeze(0) + 2.0 * sensor_phase.unsqueeze(1)
    )
    base = (carrier * speckle * envelope.unsqueeze(0)).unsqueeze(0)
    amplitude = mu.view(n_wave, 1, 1)
    noise = 0.02 * torch.randn(n_wave, n_sensor, n_time, device=device, generator=generator)
    return amplitude * base + noise
