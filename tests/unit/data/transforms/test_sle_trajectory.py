"""Tests for the SLE-kappa k-space sampling trajectory."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.data.transforms.sle_trajectory import (
    build_sle_budget_mask,
    build_sle_kspace_mask,
    kappa_to_dimension,
    sample_sle_trace,
    sample_sle_traces,
)


def test_kappa_to_dimension_formula() -> None:
    assert kappa_to_dimension(0.0) == pytest.approx(1.0)
    assert kappa_to_dimension(8.0) == pytest.approx(2.0)
    assert kappa_to_dimension(4.0) == pytest.approx(1.5)
    # Saturation at 2 above κ = 8.
    assert kappa_to_dimension(12.0) == pytest.approx(2.0)


def test_invalid_kappa_raises() -> None:
    with pytest.raises(ValueError):
        sample_sle_trace(kappa=-0.1, n_steps=10)
    with pytest.raises(ValueError):
        sample_sle_trace(kappa=8.0, n_steps=10)


def test_trace_shape_and_upper_half_plane() -> None:
    """The trace lives in the closed upper half-plane (y >= 0)."""
    gen = torch.Generator().manual_seed(0)
    trace = sample_sle_trace(kappa=2.0, n_steps=128, generator=gen)
    assert trace.shape == (129, 2)
    assert torch.all(trace[:, 1] >= -1e-6)


def test_kappa_zero_reduces_to_vertical_segment() -> None:
    """The κ → 0 limit gives a deterministic vertical segment (Cartesian line).

    For κ = 0 the driver vanishes and the slit-map composition produces a
    trace whose x-coordinate is exactly zero throughout.
    """
    trace = sample_sle_trace(kappa=0.0, n_steps=64)
    assert torch.allclose(trace[:, 0], torch.zeros_like(trace[:, 0]), atol=1e-5)


def test_mask_contains_dc_and_is_binary() -> None:
    gen = torch.Generator().manual_seed(1)
    mask = build_sle_kspace_mask(
        kappa=2.0, shape=(32, 32), n_steps=200, generator=gen
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (32, 32)
    # DC is always sampled.
    assert mask[16, 16].item() is True
    # Some non-DC points are sampled.
    assert mask.sum().item() > 1


def test_mask_acceleration_scales_with_n_steps() -> None:
    """More steps -> denser mask (monotone in n_steps for fixed κ, T)."""
    gen = torch.Generator().manual_seed(42)
    state = gen.get_state()
    short = build_sle_kspace_mask(kappa=2.0, shape=(64, 64), n_steps=50, generator=gen)
    gen.set_state(state)
    long_ = build_sle_kspace_mask(kappa=2.0, shape=(64, 64), n_steps=500, generator=gen)
    assert long_.sum().item() >= short.sum().item()


def test_trace_reproducibility() -> None:
    """Same generator state -> identical trace."""
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_sle_trace(kappa=3.0, n_steps=32, generator=g1)
    b = sample_sle_trace(kappa=3.0, n_steps=32, generator=g2)
    assert torch.allclose(a, b)


def test_mask_type_sle_kappa_dispatch() -> None:
    """The canonical MaskGenerator.generate_mask should dispatch on SLE_KAPPA."""
    from spectramr.infrastructure.physics.sampling import MaskGenerator, MaskType

    gen = MaskGenerator(seed=11)
    mask = gen.generate_mask(
        MaskType.SLE_KAPPA, (1, 32, 32), acceleration_factor=4.0, kappa=2.0
    )
    assert mask.shape == (1, 32, 32)
    assert mask.dtype == torch.bool
    # Multi-shot: exactly round(H*W / R) bins, so R is honoured.
    assert int(mask.sum()) == 256


def test_kappa_zero_tip_height_is_two_sqrt_t() -> None:
    """With hcap 2t (dg/dt = 2/(g-U)) the SLE_0 slit at time T has height 2*sqrt(T)."""
    trace = sample_sle_trace(kappa=0.0, n_steps=64, T=1.0)
    assert trace[-1, 1].item() == pytest.approx(2.0, rel=1e-4)


def test_batched_traces_are_independent_and_in_the_upper_half_plane() -> None:
    a = sample_sle_traces(2.0, 50, 3, generator=torch.Generator().manual_seed(3))
    assert a.shape == (3, 51, 2)
    assert torch.all(a[..., 1] >= 0)
    assert not torch.allclose(a[0], a[1])


def test_budget_mask_hits_the_budget_inside_the_allowed_bins() -> None:
    allowed = torch.ones(32, 32, dtype=torch.bool)
    allowed[14:18] = False
    gen = torch.Generator().manual_seed(5)
    mask, shots = build_sle_budget_mask(2.0, (32, 32), 300, allowed=allowed, generator=gen)
    assert int(mask.sum()) == 300 and shots >= 1
    assert not mask[14:18].any()


def test_budget_beyond_the_allowed_bins_raises() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        build_sle_budget_mask(2.0, (8, 8), 65)


def test_transforms_package_exposes_sle_api() -> None:
    """sle_trajectory module is mounted in src/data/transforms/__init__.py."""
    from spectramr.data import transforms as t

    assert hasattr(t, "build_sle_kspace_mask")
    assert hasattr(t, "kappa_to_dimension")
    assert hasattr(t, "sample_sle_trace")
