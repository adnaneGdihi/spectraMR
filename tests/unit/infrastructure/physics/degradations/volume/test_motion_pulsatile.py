"""Native multislice and 3-D pulsatile_motion: cardiac phase per line, one render per bin."""

from __future__ import annotations

import math
import weakref
from dataclasses import replace
from itertools import pairwise

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
from spectramr.infrastructure.physics.degradations.blocks import time_states
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import mm_to_px, warp_volume
from spectramr.infrastructure.physics.degradations.volume import motion_pulsatile
from spectramr.infrastructure.physics.degradations.volume.motion_pulsatile import (
    cardiac_bins,
    pulsation_field_mm,
)

M = AcquisitionMode
VOXEL = (2.0, 1.0, 1.0)
GEOMS = {
    "ms_sequential": AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=VOXEL, slice_order=SliceOrder.SEQUENTIAL
    ),
    "ms_interleaved": AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL),
    "volume": AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL),
}
CPU = torch.device("cpu")


def _phantom(shape=(1, 2, 6, 24, 24), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z, y, x = mm_grid(tuple(shape[-3:]), None)
    amp = 1.0 + torch.rand((*shape[:2], 1, 1, 1), generator=g)
    mag = torch.exp(-(x**2 + y**2 + z**2) / 0.3) + 0.5 * torch.cos(2.0 * x + y)
    return torch.polar(amp * mag.abs(), (1.5 * x - y + z).expand(shape))


def _apply(x, theta, geom, seed=1, geometry=None, **kw):
    g = geometry or GEOMS[geom]
    return apply_degradation("pulsatile_motion", x, theta, seed=seed, geometry=g, **kw)


def _rms(a, b) -> float:
    return float((a - b).abs().pow(2).mean().sqrt())


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_declared_native(mode):
    assert DEGRADATION_REGISTRY["pulsatile_motion"].realisation[mode] is Realisation.NATIVE


@pytest.mark.parametrize("geom", GEOMS)
def test_theta_zero_is_identity(geom):
    x = _phantom()
    assert torch.equal(_apply(x, 0.0, geom), x)


@pytest.mark.parametrize("geom", GEOMS)
def test_monotone_finite_shape_deterministic(geom):
    x = _phantom()
    outs = [_apply(x, t, geom) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape and torch.isfinite(torch.view_as_real(y)).all()
    res = [_rms(x, y) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(x, 0.6, geom), outs[2])


@pytest.mark.parametrize("geom", GEOMS)
def test_batch_invariant(geom):
    x = _phantom()
    alone = _apply(x, 0.6, geom)
    pair = _apply(torch.cat([x, 10 * _phantom(seed=9)]), 0.6, geom)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("geom", GEOMS)
def test_runs_on_real_input_without_voxel_size(geom):
    y = _apply(_phantom().abs(), 0.6, geom, geometry=replace(GEOMS[geom], voxel_mm=None))
    assert y.is_complex() and torch.isfinite(torch.view_as_real(y)).all()


@pytest.mark.parametrize("voxel_mm", [(5.0, 1.0, 1.0), (1.0, 0.5, 0.5)])
def test_field_peak_in_mm_is_the_declared_one(voxel_mm):
    # The r = 0.5 sphere passes through grid points when N // 2 is even along the widest axis.
    shape, theta = (8, 32, 32), 0.7
    peak = theta * 9.0 * voxel_mm[1]
    d = pulsation_field_mm(shape, voxel_mm, peak, CPU)
    assert float(d.pow(2).sum(1).sqrt().amax()) == pytest.approx(peak, rel=1e-5)


def test_static_pose_displacement_in_mm_is_declared():
    # base_freq_hz = 0: every line in one cardiac bin, so the output is that single pose.
    shape, theta, vox = (8, 32, 32), 0.6, (5.0, 1.0, 1.0)
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=vox)
    times = line_times(geom, shape[0], shape[1], tr_s=0.5)
    bins = cardiac_bins(times, 0.0, 16, seed=4)
    assert torch.unique(bins).numel() == 1
    pulse = math.sin((int(bins[0, 0]) + 0.5) * 2.0 * math.pi / 16)
    d_mm = pulse * pulsation_field_mm(shape, vox, theta * 9.0 * vox[1], CPU)

    z, y, x = mm_grid(shape, vox)
    ramps = torch.stack([z, y, x]).unsqueeze(0).to(torch.complex64)
    out = _apply(ramps, theta, "volume", seed=4, geometry=geom, base_freq_hz=0.0)
    torch.testing.assert_close(out, warp_volume(ramps, mm_to_px(d_mm, vox)), atol=1e-4, rtol=0)
    measured = (out - ramps).real[0]  # a linear ramp records the pull-back displacement in mm
    assert float(measured.pow(2).sum(0).sqrt().amax()) == pytest.approx(
        abs(pulse) * theta * 9.0 * vox[1], rel=1e-3
    )


def test_bins_follow_line_times():
    geom = GEOMS["ms_sequential"]
    t = line_times(geom, 4, 8, tr_s=0.5)
    bins = cardiac_bins(t, 1.2, 16, seed=0)
    phase0 = torch.rand(1, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    expected = ((2 * math.pi * 1.2 * t.double() + phase0) % (2 * math.pi)) / (2 * math.pi) * 16
    assert torch.equal(bins, expected.long())
    assert torch.unique(bins[:, 0]).numel() > 1  # sequential slices sit at different phases


def test_sequential_differs_from_interleaved():
    x = _phantom()
    assert _rms(_apply(x, 0.6, "ms_sequential"), _apply(x, 0.6, "ms_interleaved")) > 1e-3


@pytest.mark.parametrize("geom", ["ms_sequential", "volume"])
def test_bin_renders_stream_one_at_a_time(monkeypatch, geom):
    refs: list[weakref.ref] = []
    alive: list[int] = []
    real = time_states.assemble_from_states

    def tracked(n, idx, render, *, encoding):
        def wrapped(p):
            alive.append(sum(r() is not None for r in refs))
            out = render(p)
            refs.append(weakref.ref(out))
            return out

        return real(n, idx, wrapped, encoding=encoding)

    monkeypatch.setattr(motion_pulsatile, "assemble_from_states", tracked)
    _apply(_phantom(), 0.6, geom)
    assert 1 < len(refs) <= 16
    assert alive == [0] * len(refs)
    assert all(r() is None for r in refs)


@pytest.mark.parametrize("kw", [{"n_phase_bins": 0}, {"max_amplitude_px": -1.0}])
def test_rejects_bad_knobs(kw):
    with pytest.raises(ValueError, match="pulsatile_motion"):
        _apply(_phantom(), 0.5, "volume", **kw)
