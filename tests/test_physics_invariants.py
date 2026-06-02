from __future__ import annotations

import torch

from nanopa_twin.balance.hippo import hippo_legs_matrix
from nanopa_twin.mainspring.spectra import WAVELENGTHS_NM, decompose_absorption, extinction_matrix
from nanopa_twin.train_wheels.intervals import IntervalEmbedding
from nanopa_twin.train_wheels.photoacoustic import PhotoacousticEncoder


def test_absorption_is_nonnegative_for_nonnegative_concentrations() -> None:
    concentrations = torch.rand(32, 3)
    mu = decompose_absorption(concentrations)
    assert torch.all(mu >= 0.0)
    assert mu.shape == (32, len(WAVELENGTHS_NM))


def test_absorption_monotone_in_each_chromophore() -> None:
    base = torch.full((1, 3), 0.5)
    matrix = extinction_matrix()
    for k in range(3):
        bumped = base.clone()
        bumped[0, k] += 0.3
        delta = decompose_absorption(bumped) - decompose_absorption(base)
        assert torch.all(delta >= -1e-6)
        assert torch.allclose(delta[0], 0.3 * matrix[:, k], atol=1e-5)


def test_photoacoustic_concentration_head_is_nonnegative() -> None:
    encoder = PhotoacousticEncoder(3, 8, 16, 12, 2)
    signals = torch.randn(4, 3, 8, 64)
    embedding = encoder(signals)
    concentrations = encoder.concentrations(embedding)
    assert concentrations.shape == (4, 3)
    assert torch.all(concentrations >= 0.0)


def test_hippo_eigenvalues_have_negative_real_part() -> None:
    matrix = hippo_legs_matrix(16)
    eigenvalues = torch.linalg.eigvals(matrix)
    assert torch.all(eigenvalues.real < 0.0)
    transition = torch.linalg.matrix_exp(matrix)
    assert torch.isfinite(transition).all()
    assert torch.linalg.norm(transition) < torch.linalg.norm(matrix)


def test_interval_embedding_zero_gap_is_deterministic() -> None:
    embedding = IntervalEmbedding(8)
    zero = embedding(torch.zeros(5))
    again = embedding(torch.zeros(5))
    assert torch.allclose(zero, again)
    assert torch.isfinite(embedding(torch.tensor([0.0, 12.0, 24.0]))).all()
