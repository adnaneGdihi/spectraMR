"""off_resonance_slice_shift: dz = thk * df / BW_rf along the slice-select axis."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_offres import (
    slice_select_gradient_mt_per_m,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .invariants import GEOMETRIES, assert_axis_invariants

NAME = "off_resonance_slice_shift"
M = AcquisitionMode
MS = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=4.0)
VOL = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))


def _ramp(d=8, h=4, w=4):
    z = torch.arange(float(d)).view(1, 1, d, 1, 1)
    return (z * (1 + 0j)).expand(1, 1, d, h, w).to(torch.complex64)


def _uniform(hz, d=8, h=4, w=4):
    return torch.full((1, 1, d, h, w), float(hz))


@pytest.mark.parametrize("geom", ["ms_interleaved", "ms_sequential", "volume", "volume_centric"])
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


@pytest.mark.parametrize(("bw", "pitches"), [(1000.0, 0.4), (2000.0, 0.2)])
def test_a_uniform_offset_shifts_the_stack_by_thk_df_over_bw(bw, pitches):
    # 4 mm slices on a 5 mm pitch, df = 500 Hz: dz = 4 * 500 / 1000 = 2 mm = 0.4 pitch (linear).
    out = apply_degradation(NAME, _ramp(), 1.0, geometry=MS, b0_map_hz=_uniform(500.0), bw_rf_hz=bw)
    expected = torch.arange(8.0) + pitches
    torch.testing.assert_close(out[0, 0, :-1, 0, 0].real, expected[:-1], rtol=0, atol=1e-5)
    assert float(out[0, 0, -1, 0, 0].real) == pytest.approx(7.0, abs=1e-5)  # replicated edge
    torch.testing.assert_close(out.imag, torch.zeros_like(out.imag))


def test_theta_scales_the_map_and_a_negative_offset_shifts_the_other_way():
    out = apply_degradation(NAME, _ramp(), 0.5, geometry=MS, b0_map_hz=_uniform(-500.0))
    torch.testing.assert_close(
        out[0, 0, 1:, 0, 0].real, torch.arange(1.0, 8.0) - 0.2, rtol=0, atol=1e-5
    )


def test_contiguous_slices_shift_by_df_over_bw_pitches():
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0))
    out = apply_degradation(NAME, _ramp(), 1.0, geometry=geom, b0_map_hz=_uniform(300.0))
    torch.testing.assert_close(
        out[0, 0, :-1, 0, 0].real, torch.arange(7.0) + 0.3, rtol=0, atol=1e-5
    )


def test_3d_only_the_slab_boundary_moves():
    # The slab is the 8 partitions; df = 100 Hz moves it 8 * 100 / 1000 = 0.8 voxel.
    x = torch.ones(1, 1, 8, 4, 4, dtype=torch.complex64)
    out = apply_degradation(NAME, x, 1.0, geometry=VOL, b0_map_hz=_uniform(100.0))
    assert float(out[0, 0, 0].real.mean()) == pytest.approx(0.2, abs=1e-6)
    torch.testing.assert_close(out[0, 0, 1:], x[0, 0, 1:], rtol=0, atol=1e-6)
    neg = apply_degradation(NAME, x, 1.0, geometry=VOL, b0_map_hz=_uniform(-100.0))
    assert float(neg[0, 0, -1].real.mean()) == pytest.approx(0.2, abs=1e-6)
    assert float(neg[0, 0, 0].real.mean()) == pytest.approx(1.0, abs=1e-6)


def test_the_seeded_field_peaks_at_the_declared_offset_and_scales_with_b0():
    x = _ramp(d=16)
    at3 = apply_degradation(NAME, x, 1.0, seed=3, geometry=MS)
    half = apply_degradation(NAME, x, 1.0, seed=3, geometry=MS, field_scale=0.5)
    shift3, shift_half = (at3 - x).real, (half - x).real
    interior = (slice(None), slice(None), slice(1, -2))
    # Peak |df| = 200 Hz: |dz| <= 4 * 200 / 1000 / 5 = 0.16 pitch, reached somewhere in the field.
    assert float(shift3[interior].abs().max()) <= 0.16 + 1e-4
    assert float(shift3[interior].abs().max()) > 0.08
    torch.testing.assert_close(shift_half[interior], 0.5 * shift3[interior], rtol=0.02, atol=2e-3)


def test_a_profile_scales_the_field_by_b0_over_3_tesla():
    x = _ramp(d=16)
    prof3 = replace(FASTMRI_3T, geometry=MS, b0_T=3.0)
    prof15 = replace(prof3, b0_T=1.5)
    a = apply_degradation(NAME, x, 1.0, seed=3, profile=prof3)
    b = apply_degradation(NAME, x, 1.0, seed=3, profile=prof15)
    c = apply_degradation(NAME, x, 1.0, seed=3, geometry=MS, field_scale=0.5)
    torch.testing.assert_close(b, c)
    assert not torch.allclose(a, b)


def test_the_slice_select_gradient_follows_from_the_bandwidth():
    # G_ss = BW / (gamma_bar thk): 1 kHz over 4 mm at 42.577 MHz/T is 5.87 mT/m.
    assert slice_select_gradient_mt_per_m(1000.0, 4.0) == pytest.approx(5.872, abs=2e-3)
    assert slice_select_gradient_mt_per_m(2000.0, 4.0) == pytest.approx(
        2 * slice_select_gradient_mt_per_m(1000.0, 4.0)
    )


def test_a_gradient_limit_below_g_ss_raises_and_one_above_passes():
    x = _ramp()
    with pytest.raises(ValueError, match="gradient_max_mT_per_m"):
        apply_degradation(NAME, x, 0.5, geometry=MS, gradient_max_mT_per_m=5.0)
    apply_degradation(NAME, x, 0.5, geometry=MS, gradient_max_mT_per_m=6.0)
    prof = replace(FASTMRI_3T, geometry=MS, gradient_max_mT_per_m=5.0)
    with pytest.raises(ValueError, match="gradient_max_mT_per_m"):
        apply_degradation(NAME, x, 0.5, profile=prof)
    # The slab of a 3-D acquisition is D pitches thick, so its gradient is far weaker.
    apply_degradation(NAME, x, 0.5, geometry=VOL, gradient_max_mT_per_m=1.0)
    assert slice_select_gradient_mt_per_m(1000.0, 8 * 5.0) < 1.0


@pytest.mark.parametrize(
    "kw",
    [{"bw_rf_hz": 0.0}, {"peak_hz": -1.0}, {"field_scale": 0.0}, {"b0_map_hz": torch.ones(3)}],
)
def test_bad_values_raise(kw):
    with pytest.raises(ValueError):
        apply_degradation(NAME, _ramp(), 0.5, geometry=MS, **kw)


def test_a_map_that_does_not_broadcast_raises():
    with pytest.raises(ValueError, match="broadcast"):
        apply_degradation(NAME, _ramp(), 0.5, geometry=MS, b0_map_hz=torch.ones(1, 1, 5, 4, 4))


def test_thickness_in_mm_needs_the_voxel_size():
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None, slice_thickness_mm=4.0)
    with pytest.raises(ValueError, match="voxel_mm"):
        apply_degradation(NAME, _ramp(), 0.5, geometry=geom)
    assert math.isfinite(
        float(
            apply_degradation(
                NAME, _ramp(), 0.5, geometry=replace(geom, slice_thickness_mm=None)
            ).real.sum()
        )
    )


def test_slice_select_shift_has_the_assignment_polarity():
    from spectramr.infrastructure.physics.degradations.catalogue.metal_chemistry import (
        slice_select_shift_mm,
    )

    shift = slice_select_shift_mm(torch.tensor([500.0]), 4.0, 1000.0)
    assert float(shift) == pytest.approx(-2.0)


@pytest.mark.parametrize("kw", [{"peak_hz": 200.0}, {"field_scale": 1.0}])
def test_a_measured_map_refuses_the_seeded_field_scales(kw):
    with pytest.raises(ValueError, match="measured field"):
        apply_degradation(NAME, _ramp(), 0.5, geometry=MS, b0_map_hz=_uniform(100.0), **kw)
    prof = replace(FASTMRI_3T, geometry=MS)
    with pytest.raises(ValueError, match="measured field"):
        apply_degradation(NAME, _ramp(), 0.5, profile=prof, b0_map_hz=_uniform(100.0), **kw)


def test_a_profile_with_a_measured_map_uses_the_map_as_given():
    # The profile's B0 / 3 T scales the seeded field only; a measured map is not rescaled.
    prof = replace(FASTMRI_3T, geometry=MS, b0_T=1.5)
    assert DEGRADATION_REGISTRY[NAME].condition(prof, 0.5, MS, (1, 1, 8, 4, 4)) == {
        "profile_field_scale": 0.5
    }
    for p in (prof, replace(FASTMRI_3T, geometry=MS)):
        with_profile = apply_degradation(NAME, _ramp(), 0.5, profile=p, b0_map_hz=_uniform(300.0))
        alone = apply_degradation(NAME, _ramp(), 0.5, geometry=MS, b0_map_hz=_uniform(300.0))
        torch.testing.assert_close(with_profile, alone, rtol=0, atol=0)
    assert not torch.allclose(alone, _ramp())


def test_a_fastmri_profile_and_a_measured_map_equal_the_map_alone():
    # The preset's slice geometry, acquired as a multislice stack (the axis has no 2-D mode).
    geom = replace(FASTMRI_3T.geometry, mode=M.MULTISLICE_2D)
    vol = torch.rand(1, 1, 6, 8, 8, generator=torch.Generator().manual_seed(1)) + 0j
    b0 = 50.0 * torch.randn(1, 1, 6, 8, 8, generator=torch.Generator().manual_seed(2))
    out = apply_degradation(
        NAME, vol, 1.0, profile=replace(FASTMRI_3T, geometry=geom), b0_map_hz=b0
    )
    ref = apply_degradation(NAME, vol, 1.0, geometry=geom, b0_map_hz=b0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert not torch.allclose(ref, vol)


def test_field_scale_and_a_profile_scaling_together_raise():
    with pytest.raises(ValueError, match="both given"):
        apply_degradation(
            NAME, _ramp(), 0.5, profile=replace(FASTMRI_3T, geometry=MS), field_scale=1.0
        )
