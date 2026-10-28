"""imperfect_fat_sat (CHESS residual fat) and imperfect_inversion (IR efficiency, shifted null)."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionMode,
    MissingInput,
    Realisation,
    Requirement,
    apply_degradation,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
FAT, INV = "imperfect_fat_sat", "imperfect_inversion"
ALL_GEOMETRIES = [*GEOMETRIES.items()]
QMAPS = {"t1": torch.tensor(800.0), "pd": torch.tensor(1.0)}


def _real(geometry):
    shape = (1, 1, 16, 16) if geometry is None else (1, 1, 4, 16, 16)
    return torch.ones(shape)


def _apply(name, x, theta, geometry=None, seed=1, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=geometry, **kw)


@pytest.mark.parametrize("name", [FAT, INV])
@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native(name, mode):
    assert DEGRADATION_REGISTRY[name].realisation[mode] is Realisation.NATIVE


def test_requirements_and_declared_severity():
    assert DEGRADATION_REGISTRY[FAT].requires == frozenset()
    assert DEGRADATION_REGISTRY[INV].requires == frozenset({Requirement.QMAPS})
    assert DEGRADATION_REGISTRY[FAT].severity.physical_value(1.0) == 300.0
    assert DEGRADATION_REGISTRY[INV].severity.physical_value(1.0) == pytest.approx(0.3)


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_invariants(label, geometry):
    assert_axis_invariants(FAT, geometry)
    assert_axis_invariants(INV, geometry, qmaps=QMAPS, ti_ms=600.0)


# --- imperfect_fat_sat ---------------------------------------------------------------------


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_uniform_off_resonance_at_the_band_edge_leaves_one_minus_e_inverse_of_the_fat(
    label, geometry
):
    x = _real(geometry)
    out = _apply(FAT, x, 1.0, geometry, fat_map=torch.tensor(0.3), b0_map_hz=torch.tensor(100.0))
    torch.testing.assert_close(
        out - x, torch.full_like(x, 0.3 * (1 - math.exp(-1.0))), rtol=1e-5, atol=1e-6
    )
    on_resonance = _apply(
        FAT, x, 1.0, geometry, fat_map=torch.tensor(0.3), b0_map_hz=torch.tensor(0.0)
    )
    assert torch.equal(on_resonance, x)


def test_theta_scales_a_given_off_resonance_map():
    x = _real(None)
    out = _apply(FAT, x, 0.5, fat_map=torch.tensor(1.0), b0_map_hz=torch.tensor(200.0))
    torch.testing.assert_close(
        out - x, torch.full_like(x, 1 - math.exp(-1.0)), rtol=1e-5, atol=1e-6
    )


def test_realised_peak_off_resonance_matches_the_declared_300_hz():
    x = torch.ones((1, 1, 32, 32), dtype=torch.float64)
    out = _apply(FAT, x, 0.5, fat_map=torch.tensor(1.0, dtype=torch.float64))
    eta = 1.0 - (out - x)
    peak_hz = 100.0 * float(torch.sqrt(-torch.log(eta)).max())
    assert peak_hz == pytest.approx(0.5 * 300.0, rel=1e-2)


def test_seeded_fat_scales_with_the_fat_fraction_and_stays_in_the_support():
    x = torch.ones((1, 1, 16, 16))
    x[..., 8:] = 0.0
    a = _apply(FAT, x, 1.0, fat_fraction=0.1) - x
    b = _apply(FAT, x, 1.0, fat_fraction=0.2) - x
    torch.testing.assert_close(b, 2 * a, rtol=1e-4, atol=1e-6)
    assert float(a.abs().max()) > 0 and torch.equal(a[..., 8:], torch.zeros_like(a[..., 8:]))
    assert torch.equal(_apply(FAT, x, 1.0, fat_fraction=0.0), x)
    p = dataclasses.replace(FASTMRI_3T, fat_fraction=0.3)
    torch.testing.assert_close(
        _apply(FAT, x, 1.0, profile=p), _apply(FAT, x, 1.0, p.geometry, fat_fraction=0.3)
    )


def test_residual_fat_is_in_phase_with_the_image():
    g = torch.Generator().manual_seed(0)
    x = torch.polar(
        torch.rand(1, 2, 16, 16, generator=g) + 0.5, torch.randn(1, 2, 16, 16, generator=g)
    )
    out = _apply(FAT, x, 1.0)
    torch.testing.assert_close(torch.angle(out), torch.angle(x), rtol=0, atol=1e-5)
    assert (out.abs() >= x.abs()).all()


def test_a_real_negative_image_gains_residual_fat_with_its_own_sign():
    x = -torch.ones(1, 1, 16, 16)
    out = _apply(FAT, x, 1.0, fat_map=torch.tensor(0.5), b0_map_hz=torch.tensor(100.0))
    torch.testing.assert_close(out, x * (1 + 0.5 * (1 - math.exp(-1.0))), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bad", [{"band_hz": 0.0}, {"fat_fraction": 1.5}])
def test_bad_fat_sat_arguments_raise(bad):
    with pytest.raises(ValueError):
        _apply(FAT, _real(None), 0.5, **bad)


# --- imperfect_inversion -------------------------------------------------------------------


def _mz(kappa, ti, t1):
    return abs(1.0 - (1.0 + kappa) * math.exp(-ti / t1))


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_the_null_moves_from_t1_ln2_to_t1_ln_one_plus_kappa(label, geometry):
    t1, theta = 1000.0, 1.0
    kappa = 1.0 - 0.3 * theta
    qmaps = {"t1": torch.tensor(t1), "pd": torch.tensor(1.0)}
    ti = t1 * math.log(1.0 + kappa)
    x = _real(geometry) * _mz(1.0, ti, t1)
    out = _apply(INV, x, theta, geometry, qmaps=qmaps, ti_ms=ti)
    assert float(x.max()) > 0.1 and float(out.abs().max()) < 1e-6
    ti2 = t1 * math.log(2.0)
    out2 = _apply(INV, _real(geometry) * _mz(1.0, ti2, t1), theta, geometry, qmaps=qmaps, ti_ms=ti2)
    torch.testing.assert_close(out2, torch.full_like(out2, (1.0 - kappa) / 2), rtol=1e-4, atol=1e-6)


def test_a_perfect_inversion_nulls_at_t1_ln2_and_is_the_identity():
    t1 = 1000.0
    ti = t1 * math.log(2.0)
    x = _real(None) * _mz(1.0, ti, t1)
    assert float(x.max()) < 1e-6
    assert torch.equal(
        _apply(INV, x, 0.0, qmaps={"t1": torch.tensor(t1), "pd": torch.tensor(1.0)}, ti_ms=ti), x
    )


def test_output_is_the_ideal_image_times_the_signal_ratio_for_a_t1_map():
    t1 = torch.linspace(500.0, 1500.0, 16).view(1, 1, 1, 16)
    pd = torch.rand(1, 1, 16, 16) + 0.5
    ti = 600.0
    x = pd * (1.0 - 2.0 * torch.exp(-ti / t1)).abs()
    out = _apply(INV, x, 0.5, qmaps={"t1": t1, "pd": pd}, ti_ms=ti)
    expected = pd * (1.0 - 1.85 * torch.exp(-ti / t1)).abs()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_missing_maps_raise_missing_input(label, geometry):
    with pytest.raises(MissingInput):
        _apply(INV, _real(geometry), 0.5, geometry, ti_ms=600.0)


def test_incomplete_maps_and_missing_ti_raise_even_at_theta_zero():
    x = _real(None)
    with pytest.raises(MissingInput):
        _apply(INV, x, 0.0, qmaps={"t1": torch.tensor(800.0)}, ti_ms=600.0)
    for ti in (None, 0.0, -5.0):
        kw = {} if ti is None else {"ti_ms": ti}
        with pytest.raises(ValueError, match="ti_ms"):
            _apply(INV, x, 0.0, qmaps=QMAPS, **kw)


def test_ti_comes_from_the_profile_and_has_no_default():
    x = _real(None)
    p = dataclasses.replace(FASTMRI_3T, ti_ms=600.0)
    torch.testing.assert_close(
        _apply(INV, x, 0.8, profile=p, qmaps=QMAPS),
        _apply(INV, x, 0.8, p.geometry, qmaps=QMAPS, ti_ms=600.0),
    )
    with pytest.raises(ValueError, match="ti_ms"):
        _apply(INV, x, 0.8, profile=FASTMRI_3T, qmaps=QMAPS)


def test_the_meta_sweep_does_not_carry_an_axis_that_needs_maps():
    sweep = pytest.importorskip("scripts.sim2rank.degradation")  # sim2rank does not ship publicly
    assert INV not in sweep.META_PHYSICS_AXES  # the exact set is pinned in canonical_counts
