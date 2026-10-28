"""Native multislice and 3-D B0 / susceptibility fields: closed-form physics checks."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.volume.fields_b0 import (
    b0_offset_hz,
    through_slice_factor,
)

M = AcquisitionMode
MODES = (M.MULTISLICE_2D, M.VOLUME_3D)


def _x(shape=(1, 2, 4, 16, 16), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


@pytest.mark.parametrize("mode", MODES)
def test_b0_geometric_shifts_readout_by_hz_over_bandwidth(mode):
    b, d, h, w = 1, 4, 16, 32
    geom = AcquisitionGeometry(mode=mode, voxel_mm=None)
    ramp = torch.arange(w, dtype=torch.float32).expand(b, 1, d, h, w)
    y = apply_degradation(
        "b0_geometric_distortion",
        torch.complex(ramp, torch.zeros_like(ramp)),
        1.0,
        seed=5,
        geometry=geom,
    )
    # Spin-warp: dx = dB0 / BW_px with BW_px = 1 / (W * dwell), dwell 5e-5 s.
    dx = b0_offset_hz(b, (d, h, w), 1.0, 5, None, device=torch.device("cpu")) * (w * 5e-5)
    assert float(dx.abs().max()) > 0.5
    inside = ((ramp + dx) >= 0) & ((ramp + dx) <= w - 1)
    torch.testing.assert_close(y.real[inside], (ramp + dx)[inside], rtol=0, atol=1e-4)
    assert float(y.imag.abs().max()) == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_b0_geometric_leaves_h_and_d_profiles(mode):
    # An object constant along the readout cannot move under a readout-only shift.
    d, h, w = 4, 16, 16
    prof = torch.randn(1, 1, d, h, 1, generator=torch.Generator().manual_seed(2))
    x = torch.complex(prof, -prof).expand(1, 1, d, h, w)
    y = apply_degradation(
        "b0_geometric_distortion",
        x,
        1.0,
        seed=5,
        geometry=AcquisitionGeometry(mode=mode, voxel_mm=None),
    )
    torch.testing.assert_close(y, x, rtol=0, atol=1e-5)


def test_b0_offset_peak_is_600hz_times_strength():
    f = b0_offset_hz(3, (6, 16, 16), 0.5, 0, None, device=torch.device("cpu"))
    assert float(f.abs().amax()) <= 300.0 + 1e-3
    assert float(f.abs().amax()) > 150.0


def test_through_slice_factor_is_one_for_a_field_constant_along_d():
    phase = torch.randn(2, 1, 1, 8, 8).expand(2, 1, 5, 8, 8)
    geom = AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=4.0
    )
    assert torch.equal(through_slice_factor(phase, geom), torch.ones_like(phase))


def test_through_slice_factor_closed_form():
    # A linear phase ramp along D of a rad per slice, slice thickness t / spacing s:
    # |sinc(a * t / s / 2 pi)| (torch.sinc is the normalised sinc).
    a = 1.3
    phase = (a * torch.arange(6, dtype=torch.float32)).view(1, 1, 6, 1, 1).expand(1, 1, 6, 4, 4)
    geom = AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=2.5
    )
    dphi = a * 2.5 / 5.0
    expected = abs(torch.sinc(torch.tensor(dphi / (2 * torch.pi))))
    torch.testing.assert_close(
        through_slice_factor(phase, geom), torch.full_like(phase, float(expected))
    )


@pytest.mark.parametrize("name", ["b0", "susceptibility"])
def test_multislice_dropout_grows_with_thickness(name):
    x = torch.ones(1, 1, 6, 16, 16, dtype=torch.complex64)
    mags = []
    for thk in (1.0, 2.5, 4.0):
        geom = AcquisitionGeometry(
            mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=thk
        )
        mags.append(apply_degradation(name, x, 0.3, seed=2, geometry=geom).abs())
    for thin, thick in pairwise(mags):
        assert bool((thick <= thin + 1e-6).all())
        assert float(thick.mean()) < float(thin.mean())


def test_thickness_ratio_is_one_when_unset():
    # Without slice_thickness_mm the slices are taken as contiguous (ratio 1).
    x = _x()
    base = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0))
    explicit = AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=5.0
    )
    for name in ("b0", "susceptibility"):
        torch.testing.assert_close(
            apply_degradation(name, x, 0.5, seed=1, geometry=base),
            apply_degradation(name, x, 0.5, seed=1, geometry=explicit),
        )


@pytest.mark.parametrize("b0_axis", ["slice", "pe", "ro"])
def test_susceptibility_runs_for_every_b0_axis(b0_axis):
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, b0_axis=b0_axis)
    y = apply_degradation("susceptibility", _x(), 0.5, seed=1, geometry=geom)
    assert torch.isfinite(torch.view_as_real(y)).all()


def test_susceptibility_b0_axis_changes_the_field():
    x = _x()
    outs = [
        apply_degradation(
            "susceptibility",
            x,
            0.5,
            seed=1,
            geometry=AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, b0_axis=a),
        )
        for a in ("slice", "pe", "ro")
    ]
    assert not torch.allclose(outs[0], outs[1]) and not torch.allclose(outs[1], outs[2])


def _max_dropout(geom, x) -> float:
    worst = 0.0
    for seed in range(10):
        y = apply_degradation("susceptibility", x, 1.0, seed=seed, geometry=geom)
        worst = max(worst, float(1.0 - y.abs().min()))
    return worst


def test_susceptibility_multislice_dropout_exceeds_declared_in_plane_max():
    # The declared 1 - exp(-4) bounds the in-plane proxy; through-slice dephasing adds to it in MS.
    x = torch.ones(1, 1, 8, 16, 16, dtype=torch.complex64)
    ms = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=5.0)
    assert _max_dropout(ms, x) > _IN_PLANE_MAX


# The declared max_dropout, 1 - exp(-4).
_IN_PLANE_MAX = 1.0 - math.exp(-4.0)


def test_susceptibility_in_plane_dropout_within_declared_max():
    x = torch.ones(1, 1, 8, 16, 16, dtype=torch.complex64)
    vol = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))
    assert _max_dropout(vol, x) <= _IN_PLANE_MAX + 1e-5
    # One slice: no neighbour, so the through-slice factor is 1 and only the proxy remains.
    one = AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=5.0
    )
    assert _max_dropout(one, x[:, :, :1]) <= _IN_PLANE_MAX + 1e-5
