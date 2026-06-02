from __future__ import annotations

import torch
import torch.nn.functional as F

from nanopa_twin.complications.objectives import focal_loss, progression_loss, spectral_prior_loss


def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero() -> None:
    logits = torch.randn(16, 3)
    target = torch.randint(0, 3, (16,))
    focal = focal_loss(logits, target, gamma=0.0)
    reference = F.cross_entropy(logits, target)
    assert torch.allclose(focal, reference, atol=1e-6)


def test_focal_loss_downweights_easy_examples() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.2, 0.1, 0.0]])
    target = torch.tensor([0, 0])
    focal = focal_loss(logits, target, gamma=2.0)
    plain = F.cross_entropy(logits, target)
    assert focal < plain


def test_spectral_prior_zero_when_predictions_match() -> None:
    reference = torch.rand(8, 3)
    assert torch.allclose(spectral_prior_loss(reference, reference), torch.zeros(()))


def test_spectral_prior_handles_empty_batch() -> None:
    empty = torch.zeros(0, 3)
    assert spectral_prior_loss(empty, empty).item() == 0.0


def test_progression_loss_matches_flattened_cross_entropy() -> None:
    logits = torch.randn(5, 2, 3)
    target = torch.randint(0, 3, (5, 2))
    expected = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))
    assert torch.allclose(progression_loss(logits, target), expected)
