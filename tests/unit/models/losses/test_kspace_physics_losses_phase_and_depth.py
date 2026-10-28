"""Regression tests: complex-branch phase sensitivity, 5-D depth folding, reduction guard."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.kspace_physics_losses import (
    FrequencyWeightedL1Loss,
    SobolevKSpaceLoss,
)


def test_frequency_weighted_l1_penalises_a_kspace_phase_ramp() -> None:
    """A linear phase ramp in k-space is an image shift; |pred| - |target| scored it 0."""
    torch.manual_seed(0)
    k = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    ramp = torch.exp(1j * torch.linspace(0, 3.0, 16)).to(torch.complex64).view(1, 1, 1, 16)
    assert FrequencyWeightedL1Loss()(k * ramp, k).item() > 0.1


def test_frequency_weighted_l1_complex_is_the_weighted_modulus_of_the_difference() -> None:
    torch.manual_seed(1)
    pred = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    target = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    loss = FrequencyWeightedL1Loss(alpha=2.0)
    weight = loss._get_weight_map(8, 8, pred.device, pred.dtype)
    torch.testing.assert_close(loss(pred, target), (weight * (pred - target).abs()).mean())


def test_sobolev_kspace_5d_pairs_real_and_imag_within_each_slice() -> None:
    """[B, C, D, H, W] must fold D into B, not interleave D with the re/im channels."""
    torch.manual_seed(2)
    pred = torch.randn(2, 2, 3, 8, 8)
    target = torch.randn(2, 2, 3, 8, 8)
    loss = SobolevKSpaceLoss(reduction="none")
    per_slice = torch.cat([loss(pred[:, :, d], target[:, :, d]) for d in range(3)])
    folded = loss(pred, target)
    torch.testing.assert_close(folded.sum(), per_slice.sum())
    torch.testing.assert_close(SobolevKSpaceLoss()(pred, target), per_slice.mean())


def test_frequency_weighted_l1_5d_matches_the_per_slice_loss() -> None:
    pred = torch.randn(1, 2, 2, 8, 8)
    target = torch.randn(1, 2, 2, 8, 8)
    loss = FrequencyWeightedL1Loss(reduction="none")
    folded = loss(pred, target)
    torch.testing.assert_close(folded[1], loss(pred[:, :, 1], target[:, :, 1])[0])


@pytest.mark.parametrize("cls", [FrequencyWeightedL1Loss, SobolevKSpaceLoss])
def test_an_unknown_reduction_raises_at_construction(cls: type) -> None:
    with pytest.raises(ValueError, match="Unknown reduction"):
        cls(reduction="meen")


@pytest.mark.parametrize("cls", [FrequencyWeightedL1Loss, SobolevKSpaceLoss])
def test_a_complex128_call_does_not_promote_later_float32_losses(cls: type) -> None:
    """A cache keyed on shape alone would hand a float64 map to later fp32 calls."""
    loss = cls()
    loss(
        torch.randn(1, 1, 8, 8, dtype=torch.complex128),
        torch.randn(1, 1, 8, 8, dtype=torch.complex128),
    )
    assert loss(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)).dtype == torch.float32


def test_sobolev_restricted_to_acquired_ignores_never_acquired_lines() -> None:
    pred = torch.zeros(1, 2, 8, 8)
    target = torch.zeros_like(pred)
    target[..., :4] = 1.0
    acquired = torch.zeros(1, 1, 8, 8)
    acquired[..., :4] = 1.0
    scoped = SobolevKSpaceLoss(restrict_to_acquired=True)(pred, target, acquired=acquired)
    diluted = SobolevKSpaceLoss()(pred, target)
    assert scoped > diluted > 0.0
    with pytest.raises(ValueError, match="acquired"):
        SobolevKSpaceLoss(restrict_to_acquired=True)(pred, target)
    torch.testing.assert_close(SobolevKSpaceLoss()(pred, target, acquired=acquired), diluted)
