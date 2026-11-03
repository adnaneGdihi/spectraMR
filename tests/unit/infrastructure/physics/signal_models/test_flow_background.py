"""Tests for the polynomial background-phase fit."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.signal_models.flow_background import (
    fit_background_phase,
    polynomial_basis,
)


def test_basis_counts_monomials_of_bounded_degree() -> None:
    assert polynomial_basis((4, 5), 1).shape == (20, 3)
    assert polynomial_basis((4, 5, 3), 2).shape == (60, 10)


def test_fit_recovers_a_linear_background_from_static_tissue_only() -> None:
    """Flowing voxels carry extra phase; fitting on the static mask ignores them."""
    h, w = 16, 12
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    background = 0.3 + 0.4 * yy - 0.2 * xx
    phase = background.expand(3, 2, h, w).clone()
    phase[..., 6:10, 4:8] += 1.0
    static = torch.ones(h, w)
    static[6:10, 4:8] = 0.0
    fitted = fit_background_phase(phase, static, order=1, spatial_ndim=2)
    assert fitted.shape == phase.shape
    assert torch.allclose(fitted, background.expand_as(phase), atol=1e-4)


def test_fit_handles_a_volume_and_a_quadratic() -> None:
    d, h, w = 5, 6, 7
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    background = 0.1 * zz**2 - 0.3 * yy * xx + 0.05
    fitted = fit_background_phase(background[None], torch.ones(d, h, w), order=2, spatial_ndim=3)
    assert torch.allclose(fitted[0], background, atol=1e-4)


def test_empty_mask_stays_finite() -> None:
    fitted = fit_background_phase(torch.randn(2, 8, 8), torch.zeros(8, 8), order=1)
    assert torch.isfinite(fitted).all()


def test_rejects_a_bad_spatial_rank() -> None:
    with pytest.raises(ValueError, match="spatial_ndim"):
        fit_background_phase(torch.zeros(4, 4), torch.ones(4, 4), spatial_ndim=1)
