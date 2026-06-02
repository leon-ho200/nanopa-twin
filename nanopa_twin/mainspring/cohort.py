from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nanopa_twin.mainspring.phantoms import Microstructure, microstructure_from_density

BMD_MEAN: float = 0.858
BMD_SD: float = 0.120


@dataclass
class Visit:
    time_months: float
    biomarkers: tuple[float, ...]
    concentrations: tuple[float, float, float]
    structure: Microstructure
    pa_present: bool
    bmd: float
    stage: int


@dataclass
class Subject:
    subject_id: int
    visits: list[Visit]
    progression: tuple[int, ...]


def stage_from_tscore(tscore: float) -> int:
    if tscore > -1.0:
        return 0
    if tscore > -2.5:
        return 1
    return 2


def tscore_from_bmd(bmd: float) -> float:
    return (bmd - BMD_MEAN) / BMD_SD


def _normal(generator: torch.Generator, *shape: int) -> Tensor:
    return torch.randn(*shape, generator=generator)


def _density_norm(bmd: float) -> float:
    return float(min(1.0, max(0.0, (bmd - 0.45) / 0.7)))


def _biomarkers(
    tscore: float,
    turnover: float,
    n_biomarker: int,
    noise: Tensor,
) -> tuple[float, ...]:
    age = 0.5 * (-tscore) + 0.3 * noise[0].item()
    sex = 1.0 if noise[1].item() > 0.0 else 0.0
    bmi = 0.2 * tscore + 0.4 * noise[2].item()
    calcium = 0.4 * tscore - 0.3 * turnover + 0.3 * noise[3].item()
    alkaline_phosphatase = 0.9 * turnover - 0.2 * tscore + 0.3 * noise[4].item()
    vitamin_d = 0.5 * tscore + 0.3 * noise[5].item()
    creatinine = 0.2 * (-tscore) + 0.4 * noise[6].item()
    parathyroid = 0.6 * turnover - 0.3 * tscore + 0.3 * noise[7 % noise.numel()].item()
    canonical = [
        age,
        sex,
        bmi,
        calcium,
        alkaline_phosphatase,
        vitamin_d,
        creatinine,
        parathyroid,
    ]
    if n_biomarker <= len(canonical):
        return tuple(canonical[:n_biomarker])
    span = noise.numel()
    extra = [0.5 * noise[(8 + k) % span].item() for k in range(n_biomarker - len(canonical))]
    return tuple(canonical + extra)


def _concentrations(bmd: float, perfusion: float, noise: Tensor) -> tuple[float, float, float]:
    density = _density_norm(bmd)
    c_hbo2 = max(0.02, 0.55 + 0.30 * perfusion + 0.05 * noise[0].item())
    c_hb = max(0.02, 0.45 - 0.20 * perfusion + 0.05 * noise[1].item())
    c_ha = max(0.02, 0.25 + 1.20 * density + 0.05 * noise[2].item())
    return (c_hbo2, c_hb, c_ha)


def build_cohort(
    dataset: str,
    n_subjects: int,
    min_visits: int,
    max_visits: int,
    pa_availability: float,
    horizons: tuple[int, ...],
    n_biomarker: int,
    seed: int,
) -> list[Subject]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    subjects: list[Subject] = []
    pa_only = dataset == "pa_bone_sim"
    for subject_id in range(n_subjects):
        baseline_t = float(-0.6 + 1.4 * _normal(generator, 1).item())
        slope = float(-0.18 - 0.12 * abs(_normal(generator, 1).item()))
        turnover = float(_normal(generator, 1).item())
        perfusion = float(0.5 + 0.3 * _normal(generator, 1).item())
        n_visits = int(torch.randint(min_visits, max_visits + 1, (1,), generator=generator).item())
        times: list[float] = [0.0]
        for _ in range(n_visits - 1):
            gap = float(8.0 + 8.0 * torch.rand(1, generator=generator).item())
            times.append(times[-1] + gap)
        visits: list[Visit] = []
        for t in times:
            tscore = baseline_t + slope * (t / 12.0)
            bmd = BMD_MEAN + tscore * BMD_SD
            noise = _normal(generator, 12)
            structure = microstructure_from_density(_density_norm(bmd), _normal(generator, 3))
            present = pa_only or torch.rand(1, generator=generator).item() < pa_availability
            visits.append(
                Visit(
                    time_months=t,
                    biomarkers=_biomarkers(tscore, turnover, n_biomarker, noise),
                    concentrations=_concentrations(bmd, perfusion, _normal(generator, 3)),
                    structure=structure,
                    pa_present=present,
                    bmd=bmd,
                    stage=stage_from_tscore(tscore),
                )
            )
        last_t = times[-1]
        progression = tuple(
            stage_from_tscore(baseline_t + slope * ((last_t + h) / 12.0)) for h in horizons
        )
        subjects.append(Subject(subject_id=subject_id, visits=visits, progression=progression))
    return subjects
