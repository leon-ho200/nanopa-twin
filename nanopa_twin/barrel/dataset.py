from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from nanopa_twin.mainspring.cohort import Subject
from nanopa_twin.mainspring.phantoms import simulate_pa_signal
from nanopa_twin.mainspring.spectra import WAVELENGTHS_NM


class Sample(TypedDict):
    biomarkers: Tensor
    pa: Tensor
    pa_mask: Tensor
    delta_t: Tensor
    concentrations: Tensor
    stage: Tensor
    bmd: Tensor
    progression: Tensor
    length: Tensor


class Batch(TypedDict):
    biomarkers: Tensor
    pa: Tensor
    pa_mask: Tensor
    delta_t: Tensor
    concentrations: Tensor
    stage: Tensor
    bmd: Tensor
    progression: Tensor
    length: Tensor
    seq_mask: Tensor


class TwinDataset(Dataset[Sample]):
    def __init__(
        self,
        subjects: list[Subject],
        n_sensor: int,
        n_time: int,
        wavelengths: tuple[int, ...] = WAVELENGTHS_NM,
    ) -> None:
        self.subjects = subjects
        self.n_sensor = n_sensor
        self.n_time = n_time
        self.wavelengths = wavelengths

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> Sample:
        subject = self.subjects[index]
        n_wave = len(self.wavelengths)
        length = len(subject.visits)
        biomarkers = torch.tensor([v.biomarkers for v in subject.visits], dtype=torch.float32)
        concentrations = torch.tensor(
            [v.concentrations for v in subject.visits], dtype=torch.float32
        )
        pa = torch.zeros(length, n_wave, self.n_sensor, self.n_time, dtype=torch.float32)
        pa_mask = torch.zeros(length, dtype=torch.float32)
        delta_t = torch.zeros(length, dtype=torch.float32)
        previous = 0.0
        for i, visit in enumerate(subject.visits):
            delta_t[i] = visit.time_months - previous if i > 0 else 0.0
            previous = visit.time_months
            if visit.pa_present:
                generator = torch.Generator()
                generator.manual_seed(subject.subject_id * 131 + i + 1)
                pa[i] = simulate_pa_signal(
                    concentrations[i],
                    visit.structure,
                    self.n_sensor,
                    self.n_time,
                    generator,
                    self.wavelengths,
                )
                pa_mask[i] = 1.0
        return Sample(
            biomarkers=biomarkers,
            pa=pa,
            pa_mask=pa_mask,
            delta_t=delta_t,
            concentrations=concentrations,
            stage=torch.tensor(subject.visits[-1].stage, dtype=torch.long),
            bmd=torch.tensor(subject.visits[-1].bmd, dtype=torch.float32),
            progression=torch.tensor(subject.progression, dtype=torch.long),
            length=torch.tensor(length, dtype=torch.long),
        )


def collate(samples: list[Sample]) -> Batch:
    max_len = max(int(s["length"].item()) for s in samples)
    batch_size = len(samples)
    first = samples[0]
    n_biomarker = first["biomarkers"].shape[1]
    n_wave, n_sensor, n_time = first["pa"].shape[1:]
    n_chrom = first["concentrations"].shape[1]
    n_horizon = first["progression"].shape[0]

    biomarkers = torch.zeros(batch_size, max_len, n_biomarker)
    pa = torch.zeros(batch_size, max_len, n_wave, n_sensor, n_time)
    pa_mask = torch.zeros(batch_size, max_len)
    delta_t = torch.zeros(batch_size, max_len)
    concentrations = torch.zeros(batch_size, max_len, n_chrom)
    seq_mask = torch.zeros(batch_size, max_len)
    stage = torch.zeros(batch_size, dtype=torch.long)
    bmd = torch.zeros(batch_size)
    progression = torch.zeros(batch_size, n_horizon, dtype=torch.long)
    length = torch.zeros(batch_size, dtype=torch.long)

    for b, sample in enumerate(samples):
        t = int(sample["length"].item())
        biomarkers[b, :t] = sample["biomarkers"]
        pa[b, :t] = sample["pa"]
        pa_mask[b, :t] = sample["pa_mask"]
        delta_t[b, :t] = sample["delta_t"]
        concentrations[b, :t] = sample["concentrations"]
        seq_mask[b, :t] = 1.0
        stage[b] = sample["stage"]
        bmd[b] = sample["bmd"]
        progression[b] = sample["progression"]
        length[b] = sample["length"]

    return Batch(
        biomarkers=biomarkers,
        pa=pa,
        pa_mask=pa_mask,
        delta_t=delta_t,
        concentrations=concentrations,
        stage=stage,
        bmd=bmd,
        progression=progression,
        length=length,
        seq_mask=seq_mask,
    )
