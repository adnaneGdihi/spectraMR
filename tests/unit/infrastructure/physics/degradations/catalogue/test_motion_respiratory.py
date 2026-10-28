"""respiratory_motion: SI translation and AP chest-wall scaling along a breathing trace, per line."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue import motion_respiratory
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc

from .invariants import GEOMETRIES, assert_axis_invariants, phantom, rms

M = AcquisitionMode
NAME = "respiratory_motion"
# A held breath at full inspiration, referenced to end-expiration: a pure translation.
PURE_SHIFT = {"trace": "constant", "reference": "exhale", "peak_ap_scale": 0.0}


def _apply(x, theta, geometry=None, seed=1, **kw):
    return apply_degradation(NAME, x, theta, seed=seed, geometry=geometry, **kw)


def _fourier_shift(x: torch.Tensor, d_px: float, dim: int, dims: tuple[int, ...]) -> torch.Tensor:
    view = [1] * x.dim()
    view[dim] = x.shape[dim]
    f = centred_freqs(x.shape[dim], x.device).view(view)
    return ifftnc(fftnc(x, dims=dims) * torch.exp(-2j * math.pi * f * d_px), dims=dims)


@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native(mode):
    spec = DEGRADATION_REGISTRY[NAME]
    assert spec.realisation[mode] is Realisation.NATIVE
    assert spec.requires == frozenset() and spec.condition is None


def test_declared_severity_is_in_mm_and_relative_scale():
    params = {p.name: p for p in DEGRADATION_REGISTRY[NAME].severity.params}
    si, ap = params["peak_si_translation"], params["peak_ap_scale"]
    assert (si.units, si.at_theta_min, si.at_theta_max) == ("mm", 0.0, 20.0)
    assert (ap.units, ap.at_theta_min, ap.at_theta_max) == ("fraction", 0.0, 0.04)
    assert "0.04" in DEGRADATION_REGISTRY[NAME].severity.notes


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


# ── closed form: a constant trace is a pure Fourier shift by d / voxel_mm ──


def test_slice_2d_constant_trace_is_the_fourier_shift():
    x = phantom(None)
    out = _apply(x, 0.35, **PURE_SHIFT)  # 7 mm on 1 mm voxels: +7 px along PE (default SI)
    torch.testing.assert_close(out, _fourier_shift(x, 7.0, -2, (-2, -1)), rtol=0, atol=1e-5)


def test_slice_2d_reads_the_geometry_voxel_size():
    geom = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(3.0, 0.5, 0.5))
    x = phantom(None)
    out = _apply(x, 0.35, geom, **PURE_SHIFT)  # 7 mm on 0.5 mm pixels: +14 px along PE
    torch.testing.assert_close(out, _fourier_shift(x, 14.0, -2, (-2, -1)), rtol=0, atol=1e-5)


def test_a_profile_supplies_the_voxel_size_and_its_tr_is_not_read():
    from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

    x = phantom(None)
    by_profile = apply_degradation(NAME, x, 0.6, seed=1, profile=replace(FASTMRI_3T, tr_ms=7.0))
    torch.testing.assert_close(by_profile, _apply(x, 0.6, FASTMRI_3T.geometry), rtol=0, atol=0)


def test_volume_constant_trace_is_the_kz_fourier_shift_in_anisotropic_voxels():
    geom = GEOMETRIES["volume"]  # voxel (5, 1, 1): 7 mm SI along the slice axis = 1.4 voxels
    x = phantom(geom)
    out = _apply(x, 0.35, geom, **PURE_SHIFT)
    ref = _fourier_shift(x, 7.0 / 5.0, -3, (-3, -2, -1))
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


def test_multislice_in_plane_si_is_the_fourier_shift_in_pe_voxels():
    geom = replace(GEOMETRIES["ms_interleaved"], voxel_mm=(5.0, 2.0, 1.0))
    x = phantom(geom)
    out = _apply(x, 0.35, geom, si_axis="pe", ap_axis="ro", **PURE_SHIFT)
    torch.testing.assert_close(out, _fourier_shift(x, 3.5, -2, (-2, -1)), rtol=0, atol=1e-5)


def test_multislice_through_plane_si_moves_the_excited_slices():
    geom = GEOMETRIES["ms_sequential"]  # 10 mm on 5 mm slices: two slices, nothing wraps
    x = phantom(geom)
    out = _apply(x, 0.5, geom, **PURE_SHIFT)
    ref = torch.zeros_like(x)
    ref[:, :, 2:] = x[:, :, :-2]
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)


def test_ap_scale_is_four_percent_about_the_fft_centre():
    x = torch.zeros(1, 1, 64, 64, dtype=torch.complex64)
    x[..., 20, 32 + 25] = 1.0  # 25 px from the centre column 32 along RO (default AP in 2-D)
    out = _apply(x, 1.0, trace="constant", reference="exhale", peak_si_mm=0.0)
    assert int(out[0, 0, 20].abs().argmax()) == 32 + 26  # 25 * 1.04
    torch.testing.assert_close(out[0, 0, 20, 58].abs(), torch.tensor(1.0), rtol=0, atol=1e-5)


def test_a_constant_trace_referenced_to_its_mean_is_the_identity():
    geom = GEOMETRIES["volume"]
    x = phantom(geom)
    out = _apply(x, 1.0, geom, trace="constant")
    torch.testing.assert_close(out, x, rtol=0, atol=1e-5)


def test_the_mean_reference_centres_the_displacement():
    # Same motion, two references: they differ by the bulk shift A * mean(u) only.
    x = torch.zeros(1, 1, 64, 64, dtype=torch.complex64)
    x[..., 32, 20] = 1.0
    kw = {"si_axis": "ro", "ap_axis": "pe", "peak_ap_scale": 0.0}
    centred = _apply(x, 1.0, **kw)
    exhale = _apply(x, 1.0, reference="exhale", **kw)
    u = motion_respiratory.breathing_state(
        torch.arange(64, dtype=torch.float32)[None] * 0.5, "breathing", 4.0, seed=1
    )
    bulk = 20.0 * float(u.mean())
    ref = _fourier_shift(centred, bulk, -1, (-2, -1))
    torch.testing.assert_close(exhale, ref, rtol=0, atol=1e-5)


# ── closed form: a periodic trace ghosts along PE at multiples of N_pe * TR / T_resp ──


@pytest.mark.parametrize("trace", ["sinusoid", "breathing"])
@pytest.mark.parametrize(("tr_s", "spacing"), [(0.5, 8), (0.25, 4)])
def test_periodic_trace_ghosts_at_multiples_of_n_tr_over_t(trace, tr_s, spacing):
    n_pe, period = 64, 4.0
    assert n_pe * tr_s / period == spacing
    x = torch.zeros(1, 1, n_pe, 64, dtype=torch.complex64)
    x[..., 32, 20] = 1.0
    out = _apply(x, 1.0, si_axis="ro", ap_axis="pe", peak_ap_scale=0.0, trace=trace, tr_s=tr_s)
    energy = out[0, 0].abs().pow(2).sum(-1)
    on_grid = torch.arange(n_pe) % spacing == 32 % spacing
    assert float(energy[~on_grid].max()) < 1e-8
    assert float(energy[32 + spacing]) > 1e-3 and float(energy[32 - spacing]) > 1e-3


@pytest.mark.parametrize("trace", ["sinusoid", "breathing"])
def test_the_first_order_ghost_is_at_n_tr_over_t(trace):
    # Small amplitude: the fundamental 1 / T_resp dominates, so +-N TR / T beats +-2 N TR / T.
    x = torch.zeros(1, 1, 64, 64, dtype=torch.complex64)
    x[..., 32, 20] = 1.0
    out = _apply(x, 0.05, si_axis="ro", ap_axis="pe", peak_ap_scale=0.0, trace=trace)
    energy = out[0, 0].abs().pow(2).sum(-1)
    for sign in (1, -1):
        assert float(energy[32 + 8 * sign]) > float(energy[32 + 16 * sign]), energy[
            [16, 24, 40, 48]
        ]


# ── timing follows the geometry ──


def test_slice_order_changes_the_artefact():
    geom = GEOMETRIES["ms_interleaved"]
    x = phantom(geom)
    a = _apply(x, 0.6, geom)
    b = _apply(x, 0.6, replace(geom, slice_order=SliceOrder.SEQUENTIAL))
    assert rms(a, b) > 1e-3


def test_line_order_changes_the_artefact():
    x = phantom(GEOMETRIES["volume"])
    a = _apply(x, 0.6, GEOMETRIES["volume"])
    b = _apply(x, 0.6, GEOMETRIES["volume_centric"])
    assert rms(a, b) > 1e-3


def test_states_stream_one_render_at_a_time(monkeypatch):
    seen = []
    real = motion_respiratory.assemble_from_states

    def spy(n, index, render, *, encoding):
        seen.append((n, encoding))
        return real(n, index, render, encoding=encoding)

    monkeypatch.setattr(motion_respiratory, "assemble_from_states", spy)
    _apply(phantom(GEOMETRIES["volume"]), 0.6, GEOMETRIES["volume"])
    assert len(seen) == 1 and 1 < seen[0][0] <= 16 and seen[0][1] == "volume"


# ── declared axes and knobs are validated ──


@pytest.mark.parametrize(
    ("geom", "kw", "match"),
    [
        (None, {"si_axis": "slice"}, "slice"),
        (None, {"ap_axis": "slice", "si_axis": "pe"}, "slice"),
        ("volume", {"si_axis": "pe", "ap_axis": "pe"}, "distinct"),
        ("volume", {"si_axis": "z"}, "si_axis"),
        ("volume", {"ap_axis": "x"}, "ap_axis"),
        ("volume", {"trace": "gasp"}, "trace"),
        ("volume", {"reference": "inhale"}, "reference"),
        ("volume", {"peak_si_mm": -1.0}, "peak_si_mm"),
        ("volume", {"peak_ap_scale": 1.0}, "peak_ap_scale"),
        ("volume", {"period_s": 0.0}, "period_s"),
        ("volume", {"tr_s": 0.0}, "tr_s"),
        (None, {"tr_s": -1.0}, "tr_s"),
        ("volume", {"n_states": 1}, "n_states"),
    ],
)
def test_bad_knobs_raise(geom, kw, match):
    g = None if geom is None else GEOMETRIES[geom]
    with pytest.raises(ValueError, match=match):
        _apply(phantom(g), 0.5, g, **kw)


def test_voxel_mm_none_is_one_mm_isotropic():
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None)
    x = phantom(geom)
    out = _apply(x, 0.05, geom, **PURE_SHIFT)  # 1 mm = 1 voxel along the slice axis
    torch.testing.assert_close(out, _fourier_shift(x, 1.0, -3, (-3, -2, -1)), rtol=0, atol=1e-5)
