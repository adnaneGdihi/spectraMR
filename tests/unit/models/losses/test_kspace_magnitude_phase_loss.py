"""``kspace_magnitude_phase`` asks for spectral energy where the phase is unknown.

The planted shape is the shrinkage this loss exists for: under a target phase
that is uniform and independent of the input, complex L1 is minimised by a zero
prediction while this loss is minimised at the target magnitude.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.losses.kspace_magnitude_phase_loss import KSpaceMagnitudePhaseLoss
from spectramr.models.losses.registry import LossRegistry


def _interleave(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=2).flatten(1, 2)


def test_registered_under_its_name() -> None:
    assert "kspace_magnitude_phase" in LossRegistry.list_available()


def test_zero_on_identical_inputs() -> None:
    torch.manual_seed(0)
    z = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
    loss = KSpaceMagnitudePhaseLoss(region="all")
    assert loss(_interleave(z), _interleave(z)).item() < 1e-5


def test_magnitude_minimiser_is_not_zero_under_random_phase() -> None:
    """Planted: complex L1 prefers p = 0 here; this loss prefers |p| = |t|."""
    torch.manual_seed(0)
    mag = torch.ones(64, 1, 4, 4)
    phase = torch.rand(64, 1, 4, 4) * 2 * math.pi
    target = _interleave(torch.polar(mag, phase))
    zero = torch.zeros_like(target)
    right_mag = _interleave(torch.polar(mag, torch.zeros_like(phase)))

    complex_l1 = lambda p: (p - target).abs().mean()  # noqa: E731
    assert complex_l1(zero) < complex_l1(right_mag), "planted shrinkage shape not present"

    loss = KSpaceMagnitudePhaseLoss(phase_weight=0.0, region="all")
    assert loss(right_mag, target) < 1e-5 < loss(zero, target)


def test_phase_term_is_magnitude_weighted_and_wrap_safe() -> None:
    mag = torch.tensor([[[[1.0, 0.0]]]])
    target = torch.polar(mag, torch.tensor([[[[math.pi - 0.01, 0.0]]]]))
    pred = torch.polar(mag + 0.0, torch.tensor([[[[-math.pi + 0.01, 3.0]]]]))
    loss = KSpaceMagnitudePhaseLoss(phase_weight=1.0, region="all")
    # Across the +-pi wrap the phase error is 0.02 rad; the zero-magnitude bin has no weight.
    assert loss(pred, target).item() == pytest.approx(1 - math.cos(0.02), abs=1e-4)


def test_unobserved_region_ignores_acquired_bins() -> None:
    torch.manual_seed(0)
    target = torch.randn(1, 2, 8, 8)
    pred = target.clone()
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., ::2] = 1.0
    pred[..., ::2] += 5.0  # error only on acquired columns
    assert KSpaceMagnitudePhaseLoss()(pred, target, mask=mask).item() < 1e-5
    assert KSpaceMagnitudePhaseLoss(region="all")(pred, target, mask=mask).item() > 0.1


def test_gradient_is_finite_at_zero_prediction() -> None:
    pred = torch.zeros(1, 2, 4, 4, requires_grad=True)
    target = torch.randn(1, 2, 4, 4)
    KSpaceMagnitudePhaseLoss(region="all")(pred, target).backward()
    assert torch.isfinite(pred.grad).all()


def test_phase_gradient_is_bounded_near_zero_prediction() -> None:
    """Unobserved bins predicted near zero must not dominate gradient clipping.

    The gradient of the phase term is bounded by ``2|t|`` per bin over the
    ``sum |t|^2`` normaliser; the former ``|t|(1 - cos)`` form grew as ``1/|p|``.
    """
    torch.manual_seed(0)
    target = torch.randn(1, 2, 8, 8)
    pred = (1e-4 * torch.randn(1, 2, 8, 8)).requires_grad_()
    KSpaceMagnitudePhaseLoss(phase_weight=1.0, region="all")(pred, target).backward()
    t = torch.complex(target[:, 0::2], target[:, 1::2])
    bound = 2 * t.abs().max() / t.abs().square().sum() + 1.0 / t.numel()
    assert pred.grad.abs().max() <= bound * 1.01


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_inputs_are_scored_in_float32(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    target = torch.randn(1, 2, 8, 8)
    pred = target + 0.1 * torch.randn_like(target)
    reference = KSpaceMagnitudePhaseLoss(region="all")(pred, target)
    low = KSpaceMagnitudePhaseLoss(region="all")(pred.to(dtype), target.to(dtype))
    assert torch.isfinite(low)
    assert low.item() == pytest.approx(reference.item(), rel=5e-2)


def test_a_3d_mask_keeps_the_batch_axis() -> None:
    """``[B, H, W]`` masks are per sample; they must not be unioned across the batch."""
    torch.manual_seed(0)
    target = torch.randn(2, 2, 4, 4)
    pred = target.clone()
    pred[0, :, :, 0] += 5.0  # error on column 0 of sample 0 only
    mask = torch.zeros(2, 4, 4)
    mask[1, :, 0] = 1.0  # sample 1 observed column 0; sample 0 did not
    assert KSpaceMagnitudePhaseLoss()(pred, target, mask=mask).item() > 0.1


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="region"):
        KSpaceMagnitudePhaseLoss(region="observed")
    with pytest.raises(ValueError, match="phase_weight"):
        KSpaceMagnitudePhaseLoss(phase_weight=-1.0)
    with pytest.raises(ValueError, match="interleaved"):
        KSpaceMagnitudePhaseLoss()(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4))
