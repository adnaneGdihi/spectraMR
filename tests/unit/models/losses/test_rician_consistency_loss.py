"""Tests for ``RicianConsistencyLoss``'s fixed-sigma path and bias correction."""

from __future__ import annotations

import torch

from spectramr.models.losses.rician_consistency_loss import (
    RicianConsistencyLoss,
    _rician_unbiased,
)


def test_float_sigma_matches_the_broadcast_tensor_sigma() -> None:
    torch.manual_seed(0)
    magnitude = torch.rand(3, 1, 8, 8, dtype=torch.float64)
    as_tensor = torch.full((3, 1, 1, 1), 0.07, dtype=torch.float64)
    torch.testing.assert_close(
        _rician_unbiased(magnitude, 0.07), _rician_unbiased(magnitude, as_tensor)
    )


def test_fixed_sigma_loss_matches_the_closed_form() -> None:
    torch.manual_seed(1)
    pred = torch.rand(2, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    target = torch.rand(2, 1, 8, 8, dtype=torch.float64)
    sigma = 0.03
    loss = RicianConsistencyLoss(use_fourier_bridge=False, sigma=sigma)
    out = loss(pred, target)
    expected = (pred - torch.sqrt((target**2 - 2 * sigma**2).clamp_min(0.0) + 1e-12)).abs().mean()
    torch.testing.assert_close(out, expected)
    (grad,) = torch.autograd.grad(out, pred)
    assert torch.isfinite(grad).all()


def test_fixed_sigma_removes_the_rician_floor_in_background() -> None:
    torch.manual_seed(2)
    sigma = 0.05
    noise = sigma * torch.randn(2, 1, 1, 64, 64)
    background = noise.pow(2).sum(0).sqrt()  # Rayleigh magnitude of a zero signal
    loss = RicianConsistencyLoss(use_fourier_bridge=False, sigma=sigma)
    zero_pred = torch.zeros_like(background)
    corrected = loss(zero_pred, background)
    uncorrected = RicianConsistencyLoss(use_fourier_bridge=False, bias_correction=False)(
        zero_pred, background
    )
    assert float(corrected) < 0.5 * float(uncorrected)


def test_sigma_mad_matches_per_sample_loop() -> None:
    """The NaN-masked batch estimate equals the per-sample boolean-index version."""
    import torch

    from spectramr.models.losses.rician_consistency_loss import _estimate_sigma_mad

    torch.manual_seed(0)
    img = torch.rand(4, 1, 16, 16)
    img[1] = 0.5  # constant image: every voxel is background, MAD is 0
    img[2, :, :, :] = (img[2] > 0.99).float()  # almost all zero: sparse-background fallback path
    flat = img.reshape(4, -1)
    thr = torch.quantile(flat, 0.1, dim=1, keepdim=True)
    ref = []
    for b in range(4):
        vals = flat[b][flat[b] <= thr[b]]
        if vals.numel() < 4:
            vals = flat[b]
        ref.append((vals - vals.median()).abs().median() / 0.6745)
    ref = torch.stack(ref).view(4, 1, 1, 1).clamp(min=1e-8)
    torch.testing.assert_close(_estimate_sigma_mad(img), ref)
