"""b1_transmit and dielectric_b1: the B1+ scale map, its signal equation and the RF wavelength."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit import (
    centre_bright_shape,
    dielectric_kappa,
    rf_wavelength_mm,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_signal import (
    NEEDS_FA,
    SIGNAL_MODELS,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence

from .invariants import GEOMETRIES, VOXEL, assert_axis_invariants

M = AcquisitionMode
B1, DIEL = "b1_transmit", "dielectric_b1"
ALL_GEOMETRIES = [*GEOMETRIES.items()]


def _ones(geometry, size=16):
    shape = (1, 1, size, size) if geometry is None else (1, 1, 4, size, size)
    return torch.ones(shape, dtype=torch.complex64)


def _apply(name, x, theta, geometry=None, **kw):
    return apply_degradation(name, x, theta, seed=2, geometry=geometry, **kw)


@pytest.mark.parametrize("name", [B1, DIEL])
@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native_and_nothing_is_required(name, mode):
    spec = DEGRADATION_REGISTRY[name]
    assert spec.realisation[mode] is Realisation.NATIVE
    assert spec.requires == frozenset()


def test_declared_severity_ranges():
    assert DEGRADATION_REGISTRY[B1].severity.physical_value(1.0) == 0.4
    assert DEGRADATION_REGISTRY[DIEL].severity.physical_value(1.0) == 0.5


@pytest.mark.parametrize("name", [B1, DIEL])
@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_invariants(name, label, geometry):
    assert_axis_invariants(name, geometry)


def _gre_ernst(kappa, fa, tr, t1):
    e1, a = math.exp(-tr / t1), math.radians(fa)
    s = lambda k: math.sin(k * a) / (1 - e1 * math.cos(k * a))  # noqa: E731
    return s(kappa) / s(1.0)


FORMULAS = {
    "gre": lambda c: math.sin(c * math.radians(30.0)) / math.sin(math.radians(30.0)),
    "spoiled_gre": lambda c: math.sin(c * math.radians(30.0)) / math.sin(math.radians(30.0)),
    "spin_echo": lambda c: math.sin(math.radians(90 * c)) ** 3,
    "tse": lambda c: math.sin(math.radians(90 * c)) * math.sin(math.radians(180 * c / 2)) ** 2,
}


def test_every_modelled_sequence_has_a_formula_test():
    assert set(SIGNAL_MODELS) == set(FORMULAS)


@pytest.mark.parametrize("sequence", sorted(FORMULAS))
@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_uniform_kappa_ratio_equals_the_formula(sequence, label, geometry):
    c = torch.tensor(1.3)
    fa = {"fa_deg": 30.0} if sequence in NEEDS_FA else {}
    out = _apply(B1, _ones(geometry), 1.0, geometry, sequence=sequence, kappa_map=c, **fa)
    expected = FORMULAS[sequence](1.3)
    torch.testing.assert_close(out.real, torch.full_like(out.real, expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_ernst_form_applies_when_tr_is_given(label, geometry):
    kw = {"sequence": "gre", "fa_deg": 25.0, "tr_ms": 400.0, "t1_ms": 900.0}
    out = _apply(B1, _ones(geometry), 1.0, geometry, kappa_map=torch.tensor(0.7), **kw)
    expected = _gre_ernst(0.7, 25.0, 400.0, 900.0)
    torch.testing.assert_close(out.real, torch.full_like(out.real, expected), rtol=1e-5, atol=1e-6)
    small_tr = _apply(
        B1, _ones(geometry), 1.0, geometry, kappa_map=torch.tensor(0.7), sequence="gre", fa_deg=25.0
    )
    assert not torch.allclose(out, small_tr)


def test_severity_scales_kappa_linearly():
    x = _ones(None)
    half = _apply(B1, x, 0.5, sequence="gre", fa_deg=30.0, kappa_map=torch.tensor(1.4))
    expected = FORMULAS["gre"](1.2)
    torch.testing.assert_close(
        half.real, torch.full_like(half.real, expected), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_unknown_sequence_raises(label, geometry):
    with pytest.raises(ValueError, match="no transmit signal model"):
        _apply(B1, _ones(geometry), 0.0, geometry, sequence="epi")
    with pytest.raises(ValueError, match="no transmit signal model"):
        _apply(DIEL, _ones(geometry), 1.0, geometry, sequence="radial")


@pytest.mark.parametrize(
    "bad", [{"fa_deg": 0.0}, {"fa_deg": 180.0}, {"tr_ms": -1.0}, {"t1_ms": 0.0}]
)
def test_unphysical_signal_arguments_raise(bad):
    with pytest.raises(ValueError):
        _apply(B1, _ones(None), 0.5, **bad)


def test_map_peak_is_the_declared_one_and_centre_bright():
    s = centre_bright_shape(3, (6, 33, 33), VOXEL, seed=5, device=torch.device("cpu"))
    assert torch.allclose(s.abs().amax(dim=(1, 2, 3, 4)), torch.ones(3))
    centre, corners = s[:, 0, 3, 16, 16], s[:, 0, :, [0, -1]][..., [0, -1]].mean(dim=(1, 2, 3))
    assert (centre > 0).all() and (corners < 0).all()


def test_realised_kappa_is_the_calibrated_declared_map():
    x = torch.ones((1, 1, 32, 32), dtype=torch.complex64)
    out = _apply(B1, x, 0.5, sequence="gre", fa_deg=90.0)
    s_map = centre_bright_shape(1, (1, 32, 32), None, seed=2, device=torch.device("cpu"))[:, :, 0]
    full = 1.0 + 0.4 * s_map
    kappa = 1.0 + 0.5 * (full / full.mean() - 1.0)
    expected = torch.sin(kappa * math.pi / 2)
    torch.testing.assert_close(out.real, expected, rtol=1e-5, atol=1e-5)


CAL_VOXEL = (5.0, 2.0, 2.0)
CAL_GEOMETRIES = {
    "slice_2d": AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=CAL_VOXEL),
    "multislice": AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=CAL_VOXEL),
    "volume": AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=CAL_VOXEL),
}


@pytest.mark.parametrize("b0", [0.3, 3.0])
@pytest.mark.parametrize("label", sorted(CAL_GEOMETRIES))
def test_calibration_sets_the_mean_flip_angle_over_the_object_to_nominal(label, b0):
    geometry, n = CAL_GEOMETRIES[label], 64
    yy, xx = torch.meshgrid(torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij")
    disc = (yy * 2.0) ** 2 + (xx * 2.0) ** 2 <= 80.0**2
    depth = 1 if geometry.mode is M.SLICE_2D else 4
    support = disc.expand(depth, n, n)
    x = support.to(torch.complex64).unsqueeze(0).unsqueeze(0)
    x = x[:, :, 0] if geometry.mode is M.SLICE_2D else x
    out = _apply(DIEL, x, 1.0, geometry, b0_T=b0).real.reshape(depth, n, n)
    full = dielectric_kappa((depth, n, n), CAL_VOXEL, b0_T=b0)
    kappa = torch.where(support, full / full[support].mean(), torch.ones(()))
    assert float(kappa[support].mean()) == pytest.approx(1.0, abs=1e-5)
    if b0 == 0.3:  # lambda = 3.3 m: the calibrated pattern vanishes over a 16 cm head
        assert float((kappa[support] - 1.0).abs().max()) < 1e-2
    else:
        assert float((kappa[support] - 1.0).abs().max()) > 0.1
    torch.testing.assert_close(
        out, torch.sin(kappa * math.pi / 2) ** 3 * support, rtol=1e-5, atol=1e-5
    )


def test_profile_conditions_the_sequence_and_flip_angle():
    spin = _apply(B1, _ones(None), 0.8, profile=FASTMRI_3T)
    explicit = _apply(B1, _ones(None), 0.8, sequence="spin_echo")
    torch.testing.assert_close(spin, explicit, rtol=0, atol=0)
    gre = dataclasses.replace(FASTMRI_3T, sequence=Sequence.GRE, fa_deg=15.0, tr_ms=300.0)
    torch.testing.assert_close(
        _apply(B1, _ones(None), 0.8, profile=gre),
        _apply(B1, _ones(None), 0.8, sequence="gre", fa_deg=15.0, tr_ms=300.0),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="needs fa_deg"):
        _apply(B1, _ones(None), 0.8, profile=dataclasses.replace(FASTMRI_3T, sequence=Sequence.GRE))
    with pytest.raises(ValueError, match="set by scanner profile"):
        _apply(B1, _ones(None), 0.8, profile=FASTMRI_3T, sequence="gre")


def test_wavelength_follows_the_formula_and_the_field():
    lam3, lam15 = rf_wavelength_mm(3.0), rf_wavelength_mm(1.5)
    assert lam3 == pytest.approx(299_792_458e3 / (42.577e6 * 3.0 * math.sqrt(50.0)))
    assert lam15 == pytest.approx(2 * lam3) and lam3 < lam15
    with pytest.raises(ValueError):
        rf_wavelength_mm(0.0)


def test_radial_profile_is_cos_with_anisotropic_voxels():
    shape, voxel, b0 = (8, 64, 64), (5.0, 1.0, 1.5), 7.0
    kappa = dielectric_kappa(shape, voxel, b0_T=b0)
    z = (torch.arange(8) - 4).view(8, 1, 1) * voxel[0]
    y = (torch.arange(64) - 32).view(1, 64, 1) * voxel[1]
    x = (torch.arange(64) - 32).view(1, 1, 64) * voxel[2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    expected = 1.0 + 0.5 * torch.cos(2 * math.pi * r / rf_wavelength_mm(b0))
    torch.testing.assert_close(kappa, expected, rtol=1e-5, atol=1e-5)


def test_ring_spacing_changes_with_the_field():
    def first_trough_mm(b0):
        k = dielectric_kappa((1, 1, 256), (1.0, 1.0, 1.5), b0_T=b0)[0, 0, 128:]
        return float(k.argmin()) * 1.5

    assert first_trough_mm(7.0) == pytest.approx(rf_wavelength_mm(7.0) / 2, abs=1.5)
    assert first_trough_mm(3.0) == pytest.approx(rf_wavelength_mm(3.0) / 2, abs=1.5)
    assert first_trough_mm(3.0) / first_trough_mm(7.0) == pytest.approx(7 / 3, rel=0.03)


@pytest.mark.parametrize("label,geometry", ALL_GEOMETRIES)
def test_dielectric_goes_through_the_b1_signal_equation(label, geometry):
    shape = (1, 1, 32, 32) if geometry is None else (1, 1, 6, 32, 32)
    x = torch.ones(shape, dtype=torch.complex64)
    out = _apply(DIEL, x, 1.0, geometry, b0_T=7.0, sequence="spin_echo")
    full = dielectric_kappa(
        tuple(shape[-3:]) if geometry else (1, 32, 32),
        None if geometry is None else VOXEL,
        b0_T=7.0,
    )
    kappa = (full / full.mean() if geometry else full.squeeze(0) / full.mean()).expand(shape)
    expected = torch.sin(kappa * math.pi / 2) ** 3
    torch.testing.assert_close(out.real, expected, rtol=1e-5, atol=1e-5)


def test_support_leaves_the_background_untouched_and_b0_comes_from_the_profile():
    x = torch.ones((1, 1, 16, 16), dtype=torch.complex64)
    x[..., 8:] = 0.01
    out = _apply(DIEL, x, 1.0, b0_T=3.0)
    assert torch.equal(out[..., 8:], x[..., 8:]) and not torch.equal(out[..., :8], x[..., :8])
    p7 = dataclasses.replace(FASTMRI_3T, b0_T=7.0)
    explicit = _apply(DIEL, _ones(None), 1.0, p7.geometry, b0_T=7.0, sequence="spin_echo")
    torch.testing.assert_close(_apply(DIEL, _ones(None), 1.0, profile=p7), explicit, rtol=0, atol=0)
    other = _apply(DIEL, _ones(None), 1.0, p7.geometry, b0_T=1.5, sequence="spin_echo")
    assert not torch.allclose(explicit, other)


def test_non_positive_field_or_permittivity_raises():
    for bad in ({"b0_T": 0.0}, {"eps_r": -1.0}):
        with pytest.raises(ValueError):
            _apply(DIEL, _ones(None), 0.0, **bad)


def test_ernst_short_tr_differs_sharply_from_the_long_tr_limit():
    # alpha 30, kappa 1.3, TR 10 ms, T1 1000 ms: S(ka)/S(a) = 0.78, not sin(ka)/sin(a) = 1.26.
    kw = {"sequence": "gre", "fa_deg": 30.0, "kappa_map": torch.tensor(1.3)}
    short = _apply(B1, _ones(None), 1.0, tr_ms=10.0, t1_ms=1000.0, **kw)
    long_tr = _apply(B1, _ones(None), 1.0, **kw)
    assert float(short.real.mean()) == pytest.approx(0.78, abs=0.01)
    assert float(long_tr.real.mean()) == pytest.approx(math.sin(math.radians(39)) / 0.5, rel=1e-5)


def test_a_fixed_flip_sequence_refuses_fa_and_tr_and_a_large_gre_angle_raises():
    for kw in ({"fa_deg": 30.0}, {"tr_ms": 500.0}):
        with pytest.raises(ValueError, match="not modelled by this axis"):
            _apply(B1, _ones(None), 0.5, sequence="tse", **kw)
    with pytest.raises(ValueError, match="stays below 180"):
        _apply(B1, _ones(None), 0.5, sequence="gre", fa_deg=150.0)
    p = dataclasses.replace(FASTMRI_3T, sequence=Sequence.TSE, fa_deg=150.0, tr_ms=3000.0)
    assert _apply(B1, _ones(None), 0.5, profile=p).shape == (1, 1, 16, 16)
