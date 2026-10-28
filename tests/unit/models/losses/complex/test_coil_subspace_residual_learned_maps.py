"""Coil-subspace residual: finite gradients for learned maps that vanish outside the support."""

from __future__ import annotations

import torch

from spectramr.models.losses.complex.coil_subspace_residual import CoilSubspaceResidualLoss


def _interleave(z: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(z.shape[0], -1, *z.shape[-2:])


def test_zero_support_maps_that_require_grad_get_a_finite_gradient() -> None:
    """sqrt'(0) is infinite, so the squared norm must be clamped before the root."""
    torch.manual_seed(0)
    maps = torch.randn(1, 4, 8, 8, dtype=torch.complex64)
    maps[..., :2] = 0
    maps.requires_grad_(True)
    images = torch.randn(1, 4, 8, 8, dtype=torch.complex64)
    CoilSubspaceResidualLoss(input_domain="image")(_interleave(images), None, maps).backward()
    assert maps.grad is not None
    assert torch.isfinite(torch.view_as_real(maps.grad)).all()


def test_value_matches_an_explicit_projector() -> None:
    torch.manual_seed(1)
    maps = torch.randn(2, 3, 6, 6, dtype=torch.complex128)
    images = torch.randn(2, 3, 6, 6, dtype=torch.complex128)
    norm2 = maps.abs().pow(2).sum(1, keepdim=True)
    unit = maps / norm2.sqrt()
    along = (unit.conj() * images).sum(1, keepdim=True) * unit
    residual = (images - along).abs().pow(2).sum(1)
    weights = norm2.squeeze(1)
    expected = (weights * residual).sum() / weights.sum()
    out = CoilSubspaceResidualLoss(input_domain="image")(_interleave(images), None, maps)
    torch.testing.assert_close(out, expected)
