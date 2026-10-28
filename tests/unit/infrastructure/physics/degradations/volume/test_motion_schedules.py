"""Native readout-motion schedules: translation_drift, sudden_motion, random_motion."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    Realisation,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid
from spectramr.infrastructure.physics.fft_ops import fftnc

M = AcquisitionMode
NAMES = ("translation_drift", "sudden_motion", "random_motion")
SEQ = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None, slice_order=SliceOrder.SEQUENTIAL)
INTER = replace(SEQ, slice_order=SliceOrder.INTERLEAVED)
VOL = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None)
GEOMS = {"ms_sequential": SEQ, "volume": VOL}
CASES = [(n, g) for n in NAMES for g in GEOMS]
DIMS = {M.MULTISLICE_2D: (-2, -1), M.VOLUME_3D: (-3, -2, -1)}
LIMIT = {"translation_drift": 10.0, "sudden_motion": 12.0, "random_motion": 3.0}


def _phantom(shape=(1, 2, 4, 32, 32), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z, y, x = mm_grid(tuple(shape[-3:]), None)
    amp = 1.0 + torch.rand((*shape[:2], 1, 1, 1), generator=g)
    mag = torch.exp(-(x**2 + y**2 + z**2) / 0.3) + 0.5 * torch.cos(2.0 * x + y)
    return torch.polar(amp * mag.abs(), (1.5 * x - y + z).expand(shape))


def _noise(shape=(1, 1, 4, 16, 32), seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _rms(a, b) -> float:
    return float((a - b).abs().pow(2).mean().sqrt())


def _line_shift_px(x, y, geometry) -> torch.Tensor:
    # Readout shift of every line from the phase at f_x = 1 / W: -2 pi dx / W.
    w = x.shape[-1]
    kx, ky = fftnc(x, dims=DIMS[geometry.mode]), fftnc(y, dims=DIMS[geometry.mode])
    ratio = ky[..., w // 2 + 1] / kx[..., w // 2 + 1]
    return -torch.angle(ratio) * w / (2.0 * math.pi)


@pytest.mark.parametrize("name", NAMES)
def test_declared_native_in_both_modes(name):
    spec = DEGRADATION_REGISTRY[name]
    assert spec.realisation[M.MULTISLICE_2D] is Realisation.NATIVE
    assert spec.realisation[M.VOLUME_3D] is Realisation.NATIVE


@pytest.mark.parametrize(("name", "geom"), CASES)
def test_theta_zero_is_identity(name, geom):
    x = _phantom()
    assert torch.equal(apply_degradation(name, x, 0.0, seed=1, geometry=GEOMS[geom]), x)


@pytest.mark.parametrize(("name", "geom"), CASES)
def test_monotone_finite_shape_deterministic(name, geom):
    x, g = _phantom(), GEOMS[geom]
    outs = [apply_degradation(name, x, t, seed=1, geometry=g) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape and torch.isfinite(torch.view_as_real(y)).all()
    res = [_rms(x, y) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(apply_degradation(name, x, 0.6, seed=1, geometry=g), outs[2])


@pytest.mark.parametrize(("name", "geom"), CASES)
def test_batch_invariant(name, geom):
    x, g = _phantom(), GEOMS[geom]
    alone = apply_degradation(name, x, 0.6, seed=1, geometry=g)
    pair = apply_degradation(name, torch.cat([x, 10 * _phantom(seed=9)]), 0.6, seed=1, geometry=g)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("name", "geom"), CASES)
def test_real_input_is_promoted(name, geom):
    y = apply_degradation(name, _phantom().abs(), 0.6, seed=1, geometry=GEOMS[geom])
    assert y.is_complex()


@pytest.mark.parametrize("name", NAMES)
def test_interleaved_keeps_the_2d_kernel_per_slice(name):
    x = _noise()
    y = apply_degradation(name, x, 0.6, seed=2, geometry=INTER)
    for d in range(x.shape[2]):
        ref = apply_degradation(name, x[:, :, d], 0.6, seed=2)
        torch.testing.assert_close(y[:, :, d], ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_sequential_differs_from_interleaved(name):
    x = _noise()
    seq = apply_degradation(name, x, 0.6, seed=2, geometry=SEQ)
    inter = apply_degradation(name, x, 0.6, seed=2, geometry=INTER)
    assert _rms(seq, inter) > 0.1


@pytest.mark.parametrize(("name", "geom"), CASES)
def test_line_shifts_follow_the_schedule(name, geom):
    g, theta = GEOMS[geom], 0.8
    x = _noise()
    dx = _line_shift_px(x, apply_degradation(name, x, theta, seed=2, geometry=g), g)[0, 0]
    d, h = x.shape[2], x.shape[3]
    t = line_times(g, d, h, tr_s=0.5).double()
    peak = theta * LIMIT[name]
    if name == "translation_drift":
        expected = peak * (2.0 * (t - t.min()) / (t.max() - t.min()) - 1.0)
    elif name == "sudden_motion":
        expected = peak * (t >= t.flatten().sort().values[t.numel() // 2]).double()
    else:
        assert float(dx.abs().max()) <= peak + 1e-4
        assert float(dx.mean()) == pytest.approx(0.0, abs=0.2 * peak)
        assert float(dx.std()) == pytest.approx(peak / math.sqrt(3.0), rel=0.2)
        return
    torch.testing.assert_close(dx.double(), expected, rtol=0, atol=1e-3)


def test_sequential_drift_differs_between_identical_slices():
    one = _noise((1, 1, 1, 16, 32))
    y = apply_degradation("translation_drift", one.expand(1, 1, 4, 16, 32), 0.6, geometry=SEQ)
    for d in range(1, 4):
        assert _rms(y[:, :, d], y[:, :, 0]) > 0.1


def test_sudden_step_sits_at_the_median_time_in_a_sequential_stack():
    # Four slices: the median line is (2, 0), so slices 0-1 are clean and 2-3 fully shifted.
    x = _noise((1, 1, 4, 16, 32))
    dx = _line_shift_px(x, apply_degradation("sudden_motion", x, 1.0, geometry=SEQ), SEQ)[0, 0]
    torch.testing.assert_close(dx[:2], torch.zeros(2, 16), rtol=0, atol=1e-3)
    torch.testing.assert_close(dx[2:], torch.full((2, 16), 12.0), rtol=0, atol=1e-3)


def test_sudden_centric_volume_leaves_a_ghost_not_a_global_shift():
    # Centric order acquires the centre first; the median-time step splits k-space in two.
    g = replace(VOL, line_order=LineOrder.CENTRIC)
    x = _noise()
    dx = _line_shift_px(x, apply_degradation("sudden_motion", x, 1.0, geometry=g), g)[0, 0]
    shifted = (dx - 12.0).abs() < 1e-3
    assert bool(((dx.abs() < 1e-3) | shifted).all())
    assert int(shifted.sum()) == dx.numel() - dx.numel() // 2
    assert not bool(shifted[x.shape[2] // 2, x.shape[3] // 2])  # the centre is acquired early


def test_sudden_step_sits_at_the_centre_line_time():
    # Sequential: slices before the centre slice are clean, later ones shifted throughout.
    x = _noise((1, 1, 5, 16, 32))
    dx = _line_shift_px(x, apply_degradation("sudden_motion", x, 1.0, geometry=SEQ), SEQ)[0, 0]
    torch.testing.assert_close(dx[:2], torch.zeros(2, 16), rtol=0, atol=1e-3)
    torch.testing.assert_close(dx[3:], torch.full((2, 16), 12.0), rtol=0, atol=1e-3)
    torch.testing.assert_close(dx[2, :8], torch.zeros(8), rtol=0, atol=1e-3)
    torch.testing.assert_close(dx[2, 8:], torch.full((8,), 12.0), rtol=0, atol=1e-3)


def test_volume_drift_follows_the_line_order():
    x = _noise()
    kz_inner = replace(VOL, line_order=LineOrder.KZ_INNER)
    a = apply_degradation("translation_drift", x, 0.6, geometry=VOL)
    b = apply_degradation("translation_drift", x, 0.6, geometry=kz_inner)
    assert _rms(a, b) > 0.1


def test_random_draws_are_per_line_in_sequential():
    x = _noise((1, 1, 1, 16, 32)).expand(1, 1, 4, 16, 32)
    dx = _line_shift_px(x, apply_degradation("random_motion", x, 1.0, geometry=SEQ), SEQ)[0, 0]
    assert float((dx[1:] - dx[:1]).abs().max()) > 0.5
