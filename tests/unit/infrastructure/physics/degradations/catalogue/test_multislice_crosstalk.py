"""slice_crosstalk: neighbouring slice profiles saturate each slice, with T1 recovery by slice order."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_crosstalk import (
    neighbour_saturation,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .invariants import GEOMETRIES, assert_axis_invariants

NAME = "slice_crosstalk"
M = AcquisitionMode


def _geom(pitch=6.0, thickness=4.0, order=SliceOrder.INTERLEAVED):
    return AcquisitionGeometry(
        mode=M.MULTISLICE_2D,
        voxel_mm=(pitch, 1.0, 1.0),
        slice_thickness_mm=thickness,
        slice_order=order,
    )


def _ones(d=2, h=16):
    return torch.ones(1, 1, d, h, 8, dtype=torch.complex64)


def _saturation(pitch, thickness, sigma):
    """Mean over the slab of the neighbour's rect(t) * Gaussian(sigma) profile, by a 2-D Riemann sum."""
    n = 800
    u = (torch.arange(n, dtype=torch.float64) + 0.5) / n * thickness - thickness / 2
    g = torch.exp(-((pitch - u[:, None] - u[None, :]) ** 2) / (2 * sigma**2))
    return float(g.mean() * thickness / (sigma * math.sqrt(2 * math.pi)))


@pytest.mark.parametrize("geom", ["ms_interleaved", "ms_sequential"])
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_saturation_is_the_analytic_gaussian_overlap_with_anisotropic_voxels():
    # voxel (6, 1, 1) mm with 4 mm slices: a 2 mm gap; theta 1 gives sigma = thk/2 = 2 mm.
    s = _saturation(pitch=6.0, thickness=4.0, sigma=2.0)
    assert 0.02 < s < 0.6
    assert neighbour_saturation(4.0 / 6.0, 2.0 / 6.0) == pytest.approx(s, abs=1e-6)
    # Two interleaved slices are TR/2 apart, so each keeps exp(-TR/(2 T1)) of the saturation.
    out = apply_degradation(NAME, _ones(), 1.0, geometry=_geom(), tr_ms=500.0, t1_ms=1000.0)
    expected = 1.0 - s * math.exp(-0.5 * 0.5 / 1.0)
    torch.testing.assert_close(out.real, torch.full_like(out.real, expected), rtol=0, atol=2e-5)
    assert torch.equal(out.imag, torch.zeros_like(out.imag))


def test_no_gap_saturates_more_than_half_a_thickness_of_gap():
    # Same 4 mm slices: pitch 4 (gap 0) against pitch 6 (gap 0.5 thk = 2 mm).
    no_gap = apply_degradation(NAME, _ones(), 1.0, geometry=_geom(pitch=4.0))
    gapped = apply_degradation(NAME, _ones(), 1.0, geometry=_geom(pitch=6.0))
    assert (1 - no_gap.real).mean() > (1 - gapped.real).mean() > 0


def test_unset_thickness_is_contiguous_slices():
    contiguous = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(4.0, 1.0, 1.0))
    torch.testing.assert_close(
        apply_degradation(NAME, _ones(), 1.0, geometry=contiguous),
        apply_degradation(NAME, _ones(), 1.0, geometry=_geom(pitch=4.0)),
    )


def test_sequential_blocks_recover_longer_than_the_interleaved_neighbours():
    # Repo orders: sequential acquires a slice's whole block before the next, so a neighbour was
    # last excited (n_ky/2 + 1) TR earlier; interleaved excites every slice within each TR.
    h = 16
    seq = apply_degradation(NAME, _ones(h=h), 1.0, geometry=_geom(order=SliceOrder.SEQUENTIAL))
    inter = apply_degradation(NAME, _ones(h=h), 1.0, geometry=_geom())
    s = _saturation(6.0, 4.0, 2.0)
    expected = 1.0 - s * math.exp(-(h // 2 + 1) * 0.5 / 1.0)
    torch.testing.assert_close(seq.real, torch.full_like(seq.real, expected), rtol=0, atol=2e-5)
    assert (1 - inter.real).mean() > (1 - seq.real).mean() > 0


def test_interleaved_order_sets_the_neighbour_interval():
    # Four slices interleave as 0, 2, 1, 3: the neighbours of slice 1 (slices 0 and 2) are excited
    # 2/4 TR and 1/4 TR before it.
    out = apply_degradation(NAME, _ones(d=4), 1.0, geometry=_geom())
    s = _saturation(6.0, 4.0, 2.0)
    e = lambda frac: math.exp(-frac * 0.5)  # noqa: E731 -- TR/T1 = 0.5
    expected = (1 - s * e(0.5)) * (1 - s * e(0.25))
    assert float(out.real[0, 0, 1, 0, 0]) == pytest.approx(expected, abs=2e-5)
    assert float(out.real[0, 0, 1, 0, 0]) != pytest.approx(float(out.real[0, 0, 0, 0, 0]), abs=1e-4)


def test_the_effect_starts_continuously_from_zero():
    ones = _ones()
    for pitch in (4.0, 6.0):  # gap 0 and gap 0.5 thk
        g = _geom(pitch=pitch)
        full = (1 - apply_degradation(NAME, ones, 1.0, geometry=g).real).mean()
        tiny = (1 - apply_degradation(NAME, ones, 1e-4, geometry=g).real).mean()
        assert 0 <= tiny < 1e-3 * full


@pytest.mark.parametrize("kw", [{"tr_ms": 0.0}, {"t1_ms": -1.0}, {"tr_ms": True}])
def test_non_positive_timing_raises(kw):
    with pytest.raises(ValueError, match="positive"):
        apply_degradation(NAME, _ones(), 1.0, geometry=_geom(), **kw)


def test_thickness_without_voxel_size_raises():
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None, slice_thickness_mm=4.0)
    with pytest.raises(ValueError, match="voxel_mm"):
        apply_degradation(NAME, _ones(), 1.0, geometry=geom)


def test_profile_tr_is_forwarded_and_an_unset_tr_keeps_the_default():
    prof = replace(FASTMRI_3T, geometry=_geom(), tr_ms=2000.0)
    by_profile = apply_degradation(NAME, _ones(), 1.0, profile=prof)
    torch.testing.assert_close(
        by_profile, apply_degradation(NAME, _ones(), 1.0, geometry=_geom(), tr_ms=2000.0)
    )
    unset = apply_degradation(NAME, _ones(), 1.0, profile=replace(prof, tr_ms=None))
    torch.testing.assert_close(unset, apply_degradation(NAME, _ones(), 1.0, geometry=_geom()))
    with pytest.raises(ValueError, match="tr_ms"):
        apply_degradation(NAME, _ones(), 1.0, profile=prof, tr_ms=100.0)
