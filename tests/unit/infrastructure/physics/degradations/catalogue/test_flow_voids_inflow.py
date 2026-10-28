"""flow_voids and inflow_enhancement: closed-form washout and inflow gain."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    apply_degradation,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .flow_helpers import GEOMS, VOXEL, M, _apply, _phantom, _rms  # noqa: F401

# ---- flow_voids: closed form ----------------------------------------------------------------


def _factor(theta, te_ms, thk_mm):
    return max(0.0, 1.0 - theta * 50.0 * 0.01 * (te_ms / 2.0) / thk_mm)


def _inside_outside(shape):
    vessel = torch.zeros(*shape[:2], *([1] * (len(shape) - 4)), *shape[-2:])
    vessel[..., 8:12, 9:13] = 1.0
    g = torch.Generator().manual_seed(0)
    x = torch.polar(0.5 + torch.rand(shape, generator=g), torch.rand(shape, generator=g))
    return x, vessel


def test_voids_inside_the_vessel_follow_the_washout_formula_in_a_volume():
    # Slab thickness = D x voxel_mm[0] = 6 x 5 = 30 mm (anisotropic voxel).
    x, vessel = _inside_outside((1, 1, 6, 24, 24))
    for theta in (0.3, 0.7, 1.0):
        y = _apply("flow_voids", x, theta, M.VOLUME_3D, vessel_map=vessel, te_ms=20.0)
        f = _factor(theta, 20.0, 30.0)
        inside = vessel.bool().expand_as(x)
        torch.testing.assert_close(y[inside], f * x[inside], rtol=1e-6, atol=0)
        assert torch.equal(y[~inside], x[~inside])


def test_voids_use_the_slice_thickness_of_a_stack_not_its_pitch():
    x, vessel = _inside_outside((1, 1, 6, 24, 24))
    geom = AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(8.0, 1.0, 1.0), slice_thickness_mm=7.0, slice_gap_mm=1.0
    )
    y = apply_degradation("flow_voids", x, 0.4, geometry=geom, vessel_map=vessel, te_ms=20.0)
    inside = vessel.bool().expand_as(x)
    torch.testing.assert_close(y[inside], _factor(0.4, 20.0, 7.0) * x[inside], rtol=1e-6, atol=0)


def test_voids_take_a_gapless_stack_thickness_from_its_pitch():
    x, vessel = _inside_outside((1, 1, 6, 24, 24))
    inside = vessel.bool().expand_as(x)
    for gap, thk in ((0.0, 3.0), (1.0, 2.0)):
        geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(3.0, 1.0, 1.0), slice_gap_mm=gap)
        y = apply_degradation("flow_voids", x, 0.4, geometry=geom, vessel_map=vessel)
        torch.testing.assert_close(
            y[inside], _factor(0.4, 20.0, thk) * x[inside], rtol=1e-6, atol=0
        )


def test_voids_default_to_five_mm_and_vanish_at_two_thk_over_te():
    x, vessel = _inside_outside((1, 1, 24, 24))
    inside = vessel.bool().expand_as(x)
    y = apply_degradation("flow_voids", x, 0.25, vessel_map=vessel)
    torch.testing.assert_close(y[inside], _factor(0.25, 20.0, 5.0) * x[inside], rtol=1e-6, atol=0)
    # v = 2 thk / TE = 50 cm/s is the first zero: theta = 1 at the defaults.
    full = apply_degradation("flow_voids", x, 1.0, vessel_map=vessel)
    assert full[inside].abs().max().item() < 1e-6
    # TE = 40 ms halves the zero velocity to 25 cm/s (theta = 0.5); beyond it the vessel stays empty.
    kw = {"vessel_map": vessel, "te_ms": 40.0}
    assert apply_degradation("flow_voids", x, 0.5, **kw)[inside].abs().max().item() < 1e-6
    assert torch.all(apply_degradation("flow_voids", x, 0.8, **kw)[inside] == 0)


def test_a_slice_takes_its_thickness_from_voxel_mm():
    # Anisotropic slice voxel (3, 1, 1): thk = voxel_mm[0] = 3 mm, not the 5 mm fallback.
    x, vessel = _inside_outside((1, 1, 24, 24))
    inside = vessel.bool().expand_as(x)
    geom = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(3.0, 1.0, 1.0))
    y = apply_degradation("flow_voids", x, 0.4, geometry=geom, vessel_map=vessel)
    torch.testing.assert_close(y[inside], _factor(0.4, 20.0, 3.0) * x[inside], rtol=1e-6, atol=0)


def test_voids_follow_the_profile_te():
    x, vessel = _inside_outside((1, 1, 24, 24))
    # The profile's slice geometry sets the thickness: voxel_mm[0] = 5 mm.
    geom = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(5.0, 0.5, 0.5))
    prof = replace(FASTMRI_3T, tr_ms=500.0, te_ms=40.0, geometry=geom)
    y = apply_degradation("flow_voids", x, 0.2, profile=prof, vessel_map=vessel)
    inside = vessel.bool().expand_as(x)
    torch.testing.assert_close(y[inside], _factor(0.2, 40.0, 5.0) * x[inside], rtol=1e-6, atol=0)
    with pytest.raises(ValueError, match="te_ms"):
        apply_degradation("flow_voids", x, 0.2, profile=prof, te_ms=10.0)


# ---- inflow_enhancement: closed form ---------------------------------------------------------


def _recursion_gain(n, tr_ms=30.0, t1_ms=1500.0, fa_deg=30.0):
    e1, ca = math.exp(-tr_ms / t1_ms), math.cos(math.radians(fa_deg))
    m_ss = 1.0
    for _ in range(20000):  # tissue: the fixed point of the recursion
        m_ss = m_ss * ca * e1 + (1 - e1)
    m = 1.0  # a fully relaxed spin entering the volume
    for _ in range(n):
        m = m * ca * e1 + (1 - e1)
    return m / m_ss  # both signals carry the same sin(a)


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_inflow_follows_the_pulse_recursion_with_anisotropic_voxels(mode):
    # theta = 0.5 -> 25 cm/s = 0.25 mm/ms -> 7.5 mm per TR (30 ms); slice pitch 5 mm. Depth is
    # taken at the slice centre, (k + 1/2) 5 mm, so k = 1, 4, 7 see n = 1, 3, 5 pulses.
    x = torch.ones(1, 1, 8, 12, 12)
    vessel = torch.ones(1, 1, 1, 12, 12)
    y = _apply("inflow_enhancement", x, 0.5, mode, vessel_map=vessel)
    for k, n in ((1, 1), (4, 3), (7, 5)):
        assert y[0, 0, k, 5, 5].item() == pytest.approx(_recursion_gain(n), rel=1e-5)
    gains = y[0, 0, :, 5, 5]
    assert gains[0] > 1.0 and torch.all(gains[1:] <= gains[:-1])


def test_inflow_leaves_stationary_tissue_alone():
    x = torch.ones(1, 1, 7, 12, 12)
    vessel = torch.zeros(1, 1, 1, 12, 12)
    vessel[..., 4:6, 4:6] = 1.0
    y = _apply("inflow_enhancement", x, 1.0, M.VOLUME_3D, vessel_map=vessel)
    assert torch.equal(y[..., vessel[0, 0, 0] == 0], x[..., vessel[0, 0, 0] == 0])


def test_slow_flow_does_not_enhance_the_entry_slice():
    # The gain is continuous at v -> 0: the first slice is refreshed only a fraction per TR.
    x = torch.ones(1, 1, 7, 12, 12)
    vessel = torch.ones(1, 1, 1, 12, 12)
    y = _apply("inflow_enhancement", x, 0.001, M.VOLUME_3D, vessel_map=vessel)
    assert y[0, 0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_faster_flow_enhances_more_and_slower_flow_saturates():
    x = torch.ones(1, 1, 8, 12, 12)
    vessel = torch.ones(1, 1, 1, 12, 12)
    deep = [
        _apply("inflow_enhancement", x, t, M.VOLUME_3D, vessel_map=vessel)[0, 0, 4, 0, 0].item()
        for t in (0.05, 0.3, 1.0)
    ]
    assert 1.0 < deep[0] < deep[1] < deep[2]
    # theta = 0.05 -> 0.75 mm per TR; centre depth 22.5 mm -> 30 pulses: nearly saturated.
    assert deep[0] == pytest.approx(_recursion_gain(30), rel=1e-5)


def test_inflow_follows_the_profile_tr_and_flip_angle():
    x = torch.ones(1, 1, 7, 12, 12)
    vessel = torch.ones(1, 1, 1, 12, 12)
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(8.0, 1.0, 1.0))
    prof = replace(FASTMRI_3T, tr_ms=40.0, fa_deg=20.0, geometry=geom)
    y = apply_degradation("inflow_enhancement", x, 0.6, profile=prof, vessel_map=vessel)
    # theta = 0.6 -> 0.3 mm/ms -> 12 mm per TR (40 ms); k = 1 sits 12 mm deep: one pulse.
    assert y[0, 0, 1, 0, 0].item() == pytest.approx(
        _recursion_gain(1, tr_ms=40.0, fa_deg=20.0), rel=1e-5
    )
    with pytest.raises(ValueError, match="fa_deg"):
        apply_degradation("inflow_enhancement", x, 0.6, geometry=geom, fa_deg=120.0)
    with pytest.raises(ValueError, match="fa_deg"):
        apply_degradation("inflow_enhancement", x, 0.6, profile=prof, fa_deg=20.0)


def test_voids_refuse_a_gap_that_leaves_no_thickness():
    x, vessel = _inside_outside((1, 1, 6, 24, 24))
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(3.0, 1.0, 1.0), slice_gap_mm=3.0)
    with pytest.raises(ValueError, match=r"voxel_mm\[0\].*slice_gap_mm"):
        apply_degradation("flow_voids", x, 0.4, geometry=geom, vessel_map=vessel)
