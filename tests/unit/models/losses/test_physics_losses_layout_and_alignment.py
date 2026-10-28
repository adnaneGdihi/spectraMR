"""Regression tests for the layout, alignment and fallback defects in ``physics_losses``.

Each test plants a shape whose failure is silent: a finite value over the wrong
tensor, never an exception.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.physics_losses import (
    DataConsistencyLoss,
    DivergenceFreeLoss,
    GraphConsistencyLoss,
    InvertedFocalFrequencyLoss,
    ParallelImagingKSpaceLoss,
    PhysicsInformedLoss,
    SNRPreservingLoss,
)
from spectramr.models.losses.registry import LossRegistry


def _interleave(z: torch.Tensor) -> torch.Tensor:
    """Complex ``[B, C, H, W]`` -> real ``[B, 2C, H, W]`` with (re, im) channel pairs."""
    return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(z.shape[0], -1, *z.shape[-2:])


def test_divergence_free_is_zero_on_a_solenoidal_field() -> None:
    """v = curl(sin x sin y) has zero divergence; misaligned stencils read ~5e-5."""
    n = 64
    y = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64).view(n, 1).expand(n, n)
    x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64).view(1, n).expand(n, n)
    flow = torch.stack([torch.sin(x) * torch.cos(y), -torch.cos(x) * torch.sin(y)]).unsqueeze(0)
    assert DivergenceFreeLoss()(flow).item() < 1e-20


def test_divergence_free_matches_the_aligned_central_difference() -> None:
    flow = torch.randn(2, 2, 12, 10, dtype=torch.float64)
    vx, vy = flow[:, 0], flow[:, 1]
    div = (vx[:, 1:-1, 2:] - vx[:, 1:-1, :-2]) / 2 + (vy[:, 2:, 1:-1] - vy[:, :-2, 1:-1]) / 2
    torch.testing.assert_close(DivergenceFreeLoss()(flow), div.pow(2).mean())


def test_data_consistency_is_zero_for_the_documented_interleaved_layout() -> None:
    """The docstring's [B, 2, H, W] measured k-space; a plain complex cast reads 0.25 here."""
    torch.manual_seed(0)
    img = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = DataConsistencyLoss()(_interleave(img), _interleave(fft2c(img)), mask)
    assert loss.item() < 1e-10


def test_data_consistency_complex_path_matches_the_formula() -> None:
    torch.manual_seed(1)
    img = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    meas = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    expected = 2.0 * ((fft2c(img) - meas) * mask).abs().pow(2).mean()
    torch.testing.assert_close(DataConsistencyLoss(lambda_dc=2.0)(img, meas, mask), expected)


def test_data_consistency_is_still_the_registered_name() -> None:
    assert LossRegistry._custom_losses["data_consistency"] is DataConsistencyLoss


def test_physics_informed_scores_each_sample_on_its_own_mask() -> None:
    """Unsqueezing a [B, 1, H, W] mask would make it 5-D and broadcast a phantom batch axis."""
    pk = torch.randn(3, 2, 8, 8)
    target = pk.clone()
    pred = pk.clone()
    pred[1] += 5.0  # differs only where sample 1's (all-zero) mask hides it
    mask = torch.zeros(3, 1, 8, 8)
    mask[0] = 1.0
    out = PhysicsInformedLoss()(
        pred[:, :1], pred[:, :1], pred_kspace=pred, target_kspace=target, mask=mask
    )
    assert out["dc_kspace"].item() == 0.0


def test_physics_informed_channelless_mask_equals_channel_mask() -> None:
    pred, target = torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)
    mask = (torch.rand(2, 8, 8) > 0.5).float()
    loss = PhysicsInformedLoss()
    a = loss(pred, pred, pred_kspace=pred, target_kspace=target, mask=mask)["dc_kspace"]
    b = loss(pred, pred, pred_kspace=pred, target_kspace=target, mask=mask.unsqueeze(1))[
        "dc_kspace"
    ]
    torch.testing.assert_close(a, b)


def test_snr_preserving_is_finite_on_an_all_zero_target() -> None:
    zeros = torch.zeros(1, 1, 8, 8, requires_grad=True)
    out = SNRPreservingLoss()(zeros, torch.zeros(1, 1, 8, 8))
    out.backward()
    assert torch.isfinite(out)
    assert zeros.grad is not None and torch.isfinite(zeros.grad).all()


def test_parallel_imaging_single_coil_measurement_is_not_cross_batch() -> None:
    """[B, H, W] measured against [B, 1, H, W] predicted must not broadcast to [B, B, H, W]."""
    img = torch.randn(3, 1, 8, 8, dtype=torch.complex64)
    k = fft2c(img)
    loss = ParallelImagingKSpaceLoss()
    assert loss(img, k[:, 0]).item() < 1e-10
    torch.testing.assert_close(loss(img, k[:, 0]), loss(img, k))


def test_parallel_imaging_raises_on_a_coil_count_mismatch() -> None:
    img = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="do not match"):
        ParallelImagingKSpaceLoss()(img, torch.randn(1, 4, 8, 8, dtype=torch.complex64))


def test_graph_consistency_raises_on_a_mismatched_dcf() -> None:
    with pytest.raises(ValueError, match="dcf"):
        GraphConsistencyLoss()(torch.randn(1, 10, 2), torch.randn(1, 10, 2), dcf=torch.rand(1, 5))


def test_graph_consistency_weights_by_a_matching_dcf() -> None:
    pred, target, dcf = torch.randn(2, 10, 2), torch.randn(2, 10, 2), torch.rand(2, 10)
    expected = ((pred - target) ** 2).sum(-1).mul(dcf).mean()
    torch.testing.assert_close(GraphConsistencyLoss()(pred, target, dcf=dcf), expected)


def test_focal_frequency_cached_weight_matches_the_formula() -> None:
    """The radial weight is cached per shape; its value must match the formula."""
    torch.manual_seed(2)
    pred, target = torch.randn(2, 1, 12, 10), torch.randn(2, 1, 12, 10)
    loss = InvertedFocalFrequencyLoss(alpha=3.0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 10), indexing="ij")
    weight = 1.0 + 3.0 * (xx**2 + yy**2)
    expected = ((fft2c(pred) - fft2c(target)).abs() ** 2 * weight).mean()
    torch.testing.assert_close(loss(pred, target), expected)
    torch.testing.assert_close(loss(pred, target), expected)  # second call hits the cache
