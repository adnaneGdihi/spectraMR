"""metal_artifact: dipole offset, readout forward mapping, GRE dephasing, conditioning."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations._common import readout_offset_px
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.metal_chemistry import (
    dipole_offset_hz,
    forward_map_readout,
    metal_artifact_volume,
    sphere_centre_index,
    sphere_z0_mm,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence

from .invariants import GEOMETRIES, assert_axis_invariants, phantom, rms

M = AcquisitionMode
NAME = "metal_artifact"
GAMMA_HZ_PER_T = 42.577e6
VOXEL_ISO = (1.0, 1.0, 1.0)
VOLUME = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 2.0))


@pytest.mark.parametrize("geom", GEOMETRIES)
@pytest.mark.parametrize("gre", [False, True])
def test_invariants(geom, gre):
    assert_axis_invariants(NAME, GEOMETRIES[geom], gradient_echo=gre)


def test_declaration():
    spec = DEGRADATION_REGISTRY[NAME]
    assert all(r is Realisation.NATIVE for r in spec.realisation.values())
    assert set(spec.implementations) == set(M)
    assert spec.requires == frozenset()
    (p,) = spec.severity.params
    assert (p.units, p.at_theta_min, p.at_theta_max) == ("mm", 0.0, 5.0)


def test_dipole_on_ro_axis_is_minus_one_over_three_r_cubed():
    a, b0, dchi = 3.0, 3.0, 200.0
    for r in (4.0, 6.0, 12.0):
        rel = (torch.zeros(1), torch.zeros(1), torch.full((1,), r))
        hz, grad = dipole_offset_hz(rel, a, b0_axis=0, b0_T=b0, delta_chi_ppm=dchi)
        expect = -GAMMA_HZ_PER_T * b0 * dchi * 1e-6 * a**3 / (3 * r**3)
        assert hz.item() == pytest.approx(expect, rel=1e-6)
        assert grad.abs().item() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("axis", [0, 2])
def test_slice_gradient_matches_finite_difference(axis):
    base = torch.tensor([4.0, 3.0, 5.0], dtype=torch.float64)
    h = 1e-4

    def field(dz):
        rel = tuple((base[i] + (dz if i == 0 else 0.0)).view(1) for i in range(3))
        return dipole_offset_hz(rel, 2.0, b0_axis=axis, b0_T=3.0, delta_chi_ppm=200.0)[0]

    _, grad = dipole_offset_hz(
        tuple(base[i].view(1) for i in range(3)), 2.0, b0_axis=axis, b0_T=3.0, delta_chi_ppm=200.0
    )
    fd = (field(h) - field(-h)) / (2 * h)
    assert grad.item() == pytest.approx(fd.item(), rel=1e-5)


def _point_source(offset_px, shape=(6, 32, 32), seed=1):
    cz, cy, cx = sphere_centre_index(1, shape, seed)[0].long().tolist()
    x = torch.zeros(1, 1, *shape)
    x[0, 0, cz, cy, cx + offset_px] = 1.0
    return x, cx + offset_px


def _hz_at_ro_6mm(a=3.0, seed=1):
    """df at 6 mm along RO from a sphere whose centre is z0 above the source's slice (B0 along z)."""
    z0 = sphere_z0_mm(1, a, seed)[0].item()
    r2 = z0**2 + 36.0
    return -GAMMA_HZ_PER_T * 3.0 * 200e-6 * a**3 / 3 * (1 - 3 * z0**2 / r2) / r2**1.5


def test_readout_displacement_is_gamma_db_over_bw_with_anisotropic_voxels():
    # RO voxel 2 mm: 3 px = 6 mm; a = 0.6 * 5 = 3 mm, so a is converted through voxel_mm.
    x, pos = _point_source(3)
    out = apply_degradation(NAME, x, 0.6, seed=1, geometry=VOLUME).abs()
    centroid = (out.sum((0, 1, 2, 3)) * torch.arange(32)).sum() / out.sum()
    expected = -_hz_at_ro_6mm() / 250.0
    assert abs(expected) > 1.0
    assert centroid.item() == pytest.approx(pos + expected, abs=1e-4)
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("b0,bw,factor", [(0.3, 250.0, 0.1), (3.0, 125.0, 2.0)])
def test_displacement_scales_with_b0_over_bw(b0, bw, factor):
    x, pos = _point_source(3)

    def shift(**kw):
        out = apply_degradation(NAME, x, 0.6, seed=1, geometry=VOLUME, **kw).abs()
        return (out.sum((0, 1, 2, 3)) * torch.arange(32)).sum().item() / out.sum().item() - pos

    assert shift(b0_T=b0, readout_bw_hz_per_px=bw) == pytest.approx(factor * shift(), rel=1e-3)


def test_signal_void_inside_sphere_and_pile_up_outside():
    centre = _point_source(0)[0]
    assert apply_degradation(NAME, centre, 0.8, seed=1, geometry=VOLUME).abs().max() == 0.0
    out = apply_degradation(NAME, torch.ones(1, 1, 6, 32, 32), 0.8, seed=1, geometry=VOLUME)
    assert out.max() > 1.2


def test_forward_map_conserves_signal_and_piles_up():
    x = torch.ones(1, 1, 1, 1, 8)
    s = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 1, 1, 8)
    y = forward_map_readout(x, s)
    assert y.sum() == pytest.approx(8.0)
    assert y[0, 0, 0, 0, 4] == pytest.approx(2.0) and y[0, 0, 0, 0, 3] == 0.0


def test_gradient_echo_dephases_in_volume_and_off_centre_slice():
    x = torch.ones(1, 1, 6, 32, 32)
    se = apply_degradation(NAME, x, 0.8, seed=1, geometry=VOLUME)
    gre = apply_degradation(NAME, x, 0.8, seed=1, geometry=VOLUME, gradient_echo=True)
    assert gre.sum() < se.sum()
    x2 = phantom(None)
    assert not torch.equal(
        apply_degradation(NAME, x2, 0.8, seed=1, gradient_echo=True),
        apply_degradation(NAME, x2, 0.8, seed=1),
    )


def test_slice_2d_offset_is_within_the_sphere_radius():
    z0 = sphere_z0_mm(64, 3.0, 5)
    assert z0.abs().max() <= 3.0 and z0.std() > 0.5


def test_polarity_matches_b0_geometric_distortion():
    # b0_geometric_volume pulls back out(r) = x(r + df/BW_px): content moves by -df/BW_px.
    x = torch.zeros(1, 1, 1, 1, 16)
    x[..., 8] = 1.0
    hz, bw = torch.full((1, 1, 1, 1, 16), 100.0), 250.0
    pos = torch.arange(16.0)

    def centroid(y):
        return (y.sum((0, 1, 2, 3)) * pos).sum() / y.sum()

    d = torch.zeros(1, 3, 1, 1, 16)
    d[:, 2] = readout_offset_px(hz[:, 0], bw)
    pulled = centroid(warp_volume(x, d))
    mapped = centroid(forward_map_readout(x, -readout_offset_px(hz, bw)))
    assert pulled.item() == pytest.approx(8 - 0.4, abs=1e-5)
    assert mapped.item() == pytest.approx(pulled.item(), abs=1e-5)


def test_multislice_selective_excitation_shift_matches_closed_form():
    ms = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 2.0), slice_thickness_mm=4.0)
    x, _ = _point_source(3)
    cz = sphere_centre_index(1, (6, 32, 32), 1)[0, 0].item()
    out = apply_degradation(NAME, x, 0.6, seed=1, geometry=ms).abs()
    zc = (out.sum((0, 1, 3, 4)) * torch.arange(6)).sum() / out.sum()
    hz = _hz_at_ro_6mm()
    dz_slices = -4.0 * hz / 1000.0 / 5.0
    assert 0.0 < abs(dz_slices) < 1.0
    assert zc.item() == pytest.approx(cz + dz_slices, abs=1e-4)
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)
    wide = apply_degradation(NAME, x, 0.6, seed=1, geometry=ms, rf_bandwidth_hz=2000.0).abs()
    zw = (wide.sum((0, 1, 3, 4)) * torch.arange(6)).sum() / wide.sum()
    assert (zw - cz).item() == pytest.approx(dz_slices / 2, abs=1e-4)


def test_multislice_differs_from_volume():
    x = phantom(VOLUME)
    ms = replace(VOLUME, mode=M.MULTISLICE_2D)
    a = apply_degradation(NAME, x, 0.6, seed=1, geometry=ms)
    b = apply_degradation(NAME, x, 0.6, seed=1, geometry=VOLUME)
    assert rms(a, b) > 1e-3


def test_profile_conditions_and_defaults():
    g = GEOMETRIES["volume"]
    x = phantom(g)
    bare = apply_degradation(NAME, x, 0.6, seed=1, geometry=g)
    explicit = apply_degradation(
        NAME, x, 0.6, seed=1, geometry=g, b0_T=3.0, readout_bw_hz_per_px=250.0, te_ms=10.0
    )
    torch.testing.assert_close(bare, explicit, rtol=0, atol=0)
    vol = replace(FASTMRI_3T, geometry=g)
    low = apply_degradation(NAME, x, 0.6, seed=1, profile=replace(vol, b0_T=0.3))
    ref = apply_degradation(
        NAME, x, 0.6, seed=1, geometry=g, b0_T=0.3, readout_bw_hz_per_px=vol.readout_bw_hz_per_px
    )
    torch.testing.assert_close(low, ref, rtol=0, atol=0)
    assert rms(x, low) < rms(x, bare)
    gre_prof = replace(vol, sequence=Sequence.GRE, te_ms=20.0, tr_ms=100.0)
    cond = DEGRADATION_REGISTRY[NAME].condition(gre_prof, 0.6, g, tuple(x.shape))
    assert cond["gradient_echo"] is True and cond["te_ms"] == 20.0
    with pytest.raises(ValueError, match="te_ms"):
        apply_degradation(NAME, x, 0.6, seed=1, profile=gre_prof, te_ms=5.0)


@pytest.mark.parametrize("seq", [Sequence.EPI, Sequence.RADIAL, Sequence.SPIRAL])
def test_unmodelled_sequence_raises(seq):
    prof = replace(FASTMRI_3T, sequence=seq)
    with pytest.raises(ValueError, match="model"):
        apply_degradation(NAME, phantom(None), 0.5, profile=prof)  # slice_2d profile geometry


def test_bssfp_has_no_refocusing_so_it_dephases():
    prof = replace(FASTMRI_3T, sequence=Sequence.BSSFP)
    cond = DEGRADATION_REGISTRY[NAME].condition(prof, 0.5, prof.geometry, (1, 1, 1, 8, 8))
    assert cond["gradient_echo"] is True


def test_negative_susceptibility_flips_the_dipole():
    x, pos = _point_source(3)

    def shift(chi):
        out = apply_degradation(NAME, x, 0.6, seed=1, geometry=VOLUME, delta_chi_ppm=chi).abs()
        return (out.sum((0, 1, 2, 3)) * torch.arange(32)).sum().item() / out.sum().item() - pos

    assert shift(-200.0) == pytest.approx(-shift(200.0), rel=1e-3)
    with pytest.raises(ValueError, match="non-zero"):
        apply_degradation(NAME, x, 0.6, geometry=VOLUME, delta_chi_ppm=0.0)


def test_one_slice_stack_is_the_slice_2d_acquisition():
    x = phantom(None)
    g = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL_ISO)
    for gre in (False, True):
        two_d = apply_degradation(NAME, x, 0.7, seed=2, gradient_echo=gre)
        stack = apply_degradation(NAME, x.unsqueeze(2), 0.7, seed=2, geometry=g, gradient_echo=gre)
        torch.testing.assert_close(stack.squeeze(2), two_d, rtol=0, atol=0)
    assert not torch.equal(
        apply_degradation(NAME, x, 0.7, seed=2, gradient_echo=True),
        apply_degradation(NAME, x, 0.7, seed=2),
    )


@pytest.mark.parametrize(
    "kw", [{"max_radius_mm": 0.0}, {"te_ms": -1.0}, {"readout_bw_hz_per_px": 0}]
)
def test_bad_values_raise(kw):
    with pytest.raises(ValueError):
        apply_degradation(NAME, phantom(None), 0.5, **kw)


def test_slice_2d_has_b0_through_plane():
    g = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(1.0, 1.0, 1.0), b0_axis="ro")
    with pytest.raises(ValueError, match="through-plane"):
        metal_artifact_volume(phantom(None).unsqueeze(2), 0.5, 0, g)
