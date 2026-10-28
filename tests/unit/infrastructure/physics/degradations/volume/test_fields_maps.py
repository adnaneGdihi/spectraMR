"""Native multislice and 3-D field maps (bias, coil_miscal, concomitant_phase) + shared invariants."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.maxwell_concomitant import GAMMA_BAR_PROTON_RAD_S_T

M = AcquisitionMode
MODES = (M.MULTISLICE_2D, M.VOLUME_3D)
FIELDS = (
    "b0",
    "b0_geometric_distortion",
    "bias",
    "coil_miscal",
    "susceptibility",
    "concomitant_phase",
)
CASES = [(n, m) for n in FIELDS for m in MODES]
# Concomitant phase needs physical distances; 10 mm voxels make its phase non-trivial.
VOXEL = (10.0, 10.0, 10.0)


def _geom(mode, voxel_mm=VOXEL, **kw) -> AcquisitionGeometry:
    return AcquisitionGeometry(mode=mode, voxel_mm=voxel_mm, **kw)


def _x(shape=(1, 2, 4, 16, 16), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _apply(name, x, theta, mode, seed=1, geometry=None, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=geometry or _geom(mode), **kw)


def _residual(name, x, y) -> float:
    # b0's phase pi * field wraps past pi (in 2-D too), so its monotone effect is the dropout.
    if name == "b0":
        x, y = x.abs(), y.abs()
    return float((y - x).abs().pow(2).mean().sqrt())


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_declared_native(name, mode):
    assert DEGRADATION_REGISTRY[name].realisation[mode] is Realisation.NATIVE


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_theta_zero_is_identity(name, mode):
    x = _x()
    assert torch.equal(_apply(name, x, 0.0, mode), x)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_monotone_finite_shape_deterministic(name, mode):
    x = _x()
    outs = [_apply(name, x, t, mode) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y)).all()
    res = [_residual(name, x, y) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(name, x, 0.6, mode), outs[2])


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_batch_invariant(name, mode):
    x = _x()
    alone = _apply(name, x, 0.6, mode)
    pair = _apply(name, torch.cat([x, 10 * _x(seed=9)]), 0.6, mode)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("name", "mode"), [c for c in CASES if c[0] != "concomitant_phase"])
def test_runs_without_voxel_size(name, mode):
    x = _x()
    y = _apply(name, x, 0.6, mode, geometry=_geom(mode, voxel_mm=None))
    assert y.shape == x.shape and torch.isfinite(torch.view_as_real(y)).all()


@pytest.mark.parametrize("mode", MODES)
def test_concomitant_needs_voxel_mm(mode):
    with pytest.raises(ValueError, match="voxel_mm"):
        _apply("concomitant_phase", _x(), 0.5, mode, geometry=_geom(mode, voxel_mm=None))


@pytest.mark.parametrize(
    "name", ["bias", "coil_miscal", "concomitant_phase", "b0_geometric_distortion"]
)
def test_multislice_samples_the_volume_field(name):
    # One field per subject: a multislice stack sees the same field as the volume.
    x = _x()
    ms = _apply(name, x, 0.7, M.MULTISLICE_2D)
    vol = _apply(name, x, 0.7, M.VOLUME_3D)
    torch.testing.assert_close(ms, vol, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_bias_peak_log_within_declared_range(mode):
    x = torch.ones(2, 1, 6, 24, 24, dtype=torch.complex64)
    y = _apply("bias", x, 1.0, mode)
    log_b = y.real.log()
    assert float(log_b.abs().max()) <= 1.2 + 1e-5
    assert float(log_b.abs().max()) > 0.1
    assert torch.allclose(y.imag, torch.zeros_like(y.imag))


@pytest.mark.parametrize("mode", MODES)
def test_bias_varies_through_slices(mode):
    y = _apply("bias", torch.ones(1, 1, 8, 16, 16, dtype=torch.complex64), 1.0, mode)
    assert float((y[:, :, 1:] - y[:, :, :-1]).abs().max()) > 1e-3


@pytest.mark.parametrize("mode", MODES)
def test_coil_miscal_relative_error_within_declared_range(mode):
    x = torch.ones(2, 1, 6, 24, 24, dtype=torch.complex64)
    y = _apply("coil_miscal", x, 0.5, mode)
    assert float((y.real - 1).abs().max()) <= 0.5 + 1e-5


@pytest.mark.parametrize("mode", MODES)
def test_concomitant_magnitude_preserved(mode):
    x = _x((1, 2, 6, 16, 16))
    y = _apply("concomitant_phase", x, 1.0, mode)
    rel = (y.abs() - x.abs()).norm() / x.abs().norm()
    assert float(rel) < 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_concomitant_phase_is_the_maxwell_r2_term(mode):
    # The 2-D primitive drives the slice-axis gradient: phi = gamma G^2 tau (x^2 + y^2) / (8 B0).
    d, h, w = 4, 16, 12
    voxel = (5.0, 10.0, 8.0)
    x = torch.ones(1, 1, d, h, w, dtype=torch.complex64)
    theta = 0.5
    y = _apply("concomitant_phase", x, theta, mode, geometry=_geom(mode, voxel_mm=voxel))
    yy = ((torch.arange(h) - (h - 1) / 2) * voxel[1] * 1e-3).view(h, 1)
    xx = ((torch.arange(w) - (w - 1) / 2) * voxel[2] * 1e-3).view(1, w)
    g = theta * 30e-3
    expected = GAMMA_BAR_PROTON_RAD_S_T * g**2 * 5e-3 * (xx**2 + yy**2) / (8 * 0.3)
    assert float(expected.max()) < math.pi  # no wrap
    for k in range(d):  # independent of the slice position
        torch.testing.assert_close(y[0, 0, k].angle(), expected.float(), rtol=1e-4, atol=1e-5)


def test_d1_concomitant_matches_2d_sign():
    # Same sign as the 2-D kernel; the grids differ (N - 1 vs N spacing), so compare signs only.
    x = torch.ones(1, 1, 16, 16, dtype=torch.complex64)
    ref = apply_degradation("concomitant_phase", x, 0.5, voxel_size_mm=10.0)
    geom = _geom(M.MULTISLICE_2D, voxel_mm=(10.0, 10.0, 10.0))
    out = apply_degradation("concomitant_phase", x.unsqueeze(2), 0.5, geometry=geom)
    assert float(ref.angle()[0, 0, 0, 0]) > 0 and float(out.angle()[0, 0, 0, 0, 0]) > 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("b0_axis", ["pe", "ro"])
def test_concomitant_rejects_in_plane_b0(mode, b0_axis):
    with pytest.raises(ValueError, match="b0_axis='slice'"):
        _apply("concomitant_phase", _x(), 0.5, mode, geometry=_geom(mode, b0_axis=b0_axis))
