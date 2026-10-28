"""Tests for ``PhysicsInformedLoss`` (``physics_informed_integration``)."""

from __future__ import annotations

import torch

from spectramr.models.losses.physics_informed_integration_loss import PhysicsInformedLoss


def test_unmasked_dc_accepts_complex_kspace() -> None:
    torch.manual_seed(0)
    image = torch.rand(2, 1, 8, 8)
    ref = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    pred = (ref + 0.1 * torch.randn_like(ref)).requires_grad_(True)
    out = PhysicsInformedLoss(lambda_dc=1.0)(image, image, pred, ref, None)
    expected = (pred - ref).abs().pow(2).mean()
    torch.testing.assert_close(out["dc_loss"], expected)
    (grad,) = torch.autograd.grad(out["total"], pred)
    assert torch.isfinite(torch.view_as_real(grad)).all()


def test_unmasked_dc_on_real_kspace_is_mse() -> None:
    torch.manual_seed(1)
    image = torch.rand(2, 1, 8, 8)
    ref = torch.randn(2, 1, 8, 8, 2)
    pred = ref + 0.1 * torch.randn_like(ref)
    out = PhysicsInformedLoss(lambda_dc=1.0)(image, image, pred, ref, None)
    torch.testing.assert_close(out["dc_loss"], torch.nn.functional.mse_loss(pred, ref))


def test_masked_dc_is_the_squared_norm_on_sampled_locations() -> None:
    torch.manual_seed(2)
    image = torch.rand(2, 1, 8, 8)
    ref = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    pred = ref + 0.1 * torch.randn_like(ref)
    mask = (torch.rand(2, 8, 8) > 0.5).float()
    out = PhysicsInformedLoss(lambda_dc=1.0)(image, image, pred, ref, mask)
    expected = ((pred - ref) * mask.unsqueeze(1)).abs().pow(2).sum()
    torch.testing.assert_close(out["dc_loss"], expected)
