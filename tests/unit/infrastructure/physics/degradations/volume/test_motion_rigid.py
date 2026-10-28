"""Native multislice and 3-D rigid_motion: rotate per shot, gather, translate per line."""

from __future__ import annotations

import weakref
from dataclasses import replace
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.blocks import time_states
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid
from spectramr.infrastructure.physics.degradations.volume import motion_rigid
from spectramr.infrastructure.physics.degradations.volume.motion_rigid import (
    rigid_multislice_kspace,
    rigid_volume_kspace,
    rotation_matrices,
)
from spectramr.infrastructure.physics.fft_ops import fftnc

M = AcquisitionMode
VOXEL = (2.0, 1.0, 1.0)
GEOMS = {
    "ms_sequential": AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=VOXEL, slice_order=SliceOrder.SEQUENTIAL
    ),
    "ms_interleaved": AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL),
    "volume": AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL),
}


def _phantom(shape=(1, 2, 6, 24, 24), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z, y, x = mm_grid(tuple(shape[-3:]), None)
    amp = 1.0 + torch.rand((*shape[:2], 1, 1, 1), generator=g)
    mag = torch.exp(-(x**2 + y**2 + z**2) / 0.3) + 0.5 * torch.cos(2.0 * x + y)
    return torch.polar(amp * mag.abs(), (1.5 * x - y + z).expand(shape))


def _apply(x, theta, geom, seed=1, **kw):
    return apply_degradation("rigid_motion", x, theta, seed=seed, geometry=GEOMS[geom], **kw)


def _rms(a, b) -> float:
    return float((a - b).abs().pow(2).mean().sqrt())


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_declared_native(mode):
    assert DEGRADATION_REGISTRY["rigid_motion"].realisation[mode] is Realisation.NATIVE


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
    g = replace(GEOMS[geom], voxel_mm=None)
    y = apply_degradation("rigid_motion", _phantom().abs(), 0.6, seed=1, geometry=g)
    assert y.is_complex() and torch.isfinite(torch.view_as_real(y)).all()


def test_rotation_matrices_are_rotations_and_reduce_to_2d():
    a = torch.rand(5, 3) - 0.5
    r = rotation_matrices(a)
    torch.testing.assert_close(
        r @ r.transpose(-1, -2), torch.eye(3).expand(5, 3, 3), atol=1e-6, rtol=0
    )
    inplane = rotation_matrices(torch.tensor([[0.3, 0.0, 0.0]]))[0]
    c, s = torch.cos(torch.tensor(0.3)), torch.sin(torch.tensor(0.3))
    expected = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(inplane, expected)


def _shots(geom, d, h):
    return shot_index(line_times(GEOMS[geom], d, h, tr_s=0.5), 8)


def test_volume_translation_per_shot_is_the_exact_fourier_shift():
    # Constant translation within each shot, no rotation: each shot's lines are those of the
    # whole volume circularly shifted by that shot's integer vector.
    x = _phantom()
    d, h = x.shape[2:4]
    shots = _shots("volume", d, h)
    vectors = torch.randint(-4, 5, (8, 3), generator=torch.Generator().manual_seed(0))
    shifts = vectors[shots].permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, D, H]
    k = rigid_volume_kspace(x, shots, None, shifts, VOXEL)
    expected = torch.zeros_like(k)
    for s in range(8):
        rolled = torch.roll(x, tuple(int(v) for v in vectors[s]), dims=(-3, -2, -1))
        sel = (shots == s)[None, None, :, :, None]
        expected = torch.where(sel, fftnc(rolled, dims=(-3, -2, -1)), expected)
    torch.testing.assert_close(k, expected, rtol=0, atol=1e-4)


def test_volume_equal_translations_are_the_whole_volume_circular_shift():
    x = _phantom()
    d, h = x.shape[2:4]
    shifts = torch.tensor([2.0, -3.0, 5.0]).view(1, 3, 1, 1).expand(1, 3, d, h)
    k = rigid_volume_kspace(x, _shots("volume", d, h), None, shifts, VOXEL)
    rolled = torch.roll(x, (2, -3, 5), dims=(-3, -2, -1))
    torch.testing.assert_close(k, fftnc(rolled, dims=(-3, -2, -1)), rtol=0, atol=1e-4)


def test_pure_translation_keeps_every_kspace_magnitude():
    # max_rotation = 0: only per-line phases, so |k| is untouched line by line.
    x = _phantom()
    y = _apply(x, 0.8, "volume", max_rotation=0.0)
    dims = (-3, -2, -1)
    torch.testing.assert_close(
        fftnc(y, dims=dims).abs(), fftnc(x, dims=dims).abs(), atol=1e-4, rtol=1e-4
    )
    assert _rms(x, y) > 0.05


def test_multislice_integer_dz_moves_each_shots_source_slice():
    x = _phantom((1, 1, 8, 16, 16))
    d, h = x.shape[2:4]
    geom = "ms_sequential"
    shots = _shots(geom, d, h)
    dz = torch.tensor([[1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, 3.0]])
    k = rigid_multislice_kspace(x, shots, torch.zeros(1, 8), dz, torch.zeros(1, 2, d, h), VOXEL)
    k_src = fftnc(x, dims=(-2, -1))
    for s in range(8):
        step = int(dz[0, s])
        for sl, ky in (shots == s).nonzero().tolist():
            src = sl + step
            if 0 <= src < d:
                torch.testing.assert_close(k[0, :, sl, ky], k_src[0, :, src, ky], atol=1e-4, rtol=0)
            else:
                assert float(k[0, :, sl, ky].abs().max()) < 1e-5  # moved out of the stack


def test_multislice_through_plane_uses_line_times():
    # Sequential and interleaved stacks put a slice's lines in different shots.
    x = _phantom()
    assert _rms(_apply(x, 0.6, "ms_sequential"), _apply(x, 0.6, "ms_interleaved")) > 1e-3


def _track_renders(monkeypatch, module):
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

    monkeypatch.setattr(module, "assemble_from_states", tracked)
    return refs, alive


@pytest.mark.parametrize("geom", ["ms_sequential", "volume"])
def test_shot_renders_stream_one_at_a_time(monkeypatch, geom):
    refs, alive = _track_renders(monkeypatch, motion_rigid)
    _apply(_phantom(), 0.6, geom)
    assert len(refs) == 8
    assert alive == [0] * 8
    assert all(r() is None for r in refs)


def test_rejects_negative_limits():
    with pytest.raises(ValueError, match=">= 0"):
        _apply(_phantom(), 0.5, "volume", max_translation=-1.0)


def test_inplane_volume_rotation_is_the_2d_primitives_rotation_per_slice():
    x = _phantom((1, 1, 4, 20, 20))
    a = 0.2
    out = motion_rigid._rotate_volume(x, torch.tensor([[a, 0.0, 0.0]]), (3.0, 1.0, 1.0))
    c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
    affine = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]]).expand(4, 2, 3)
    planes = x[0].permute(1, 0, 2, 3)  # [D, C, H, W]
    grid = F.affine_grid(affine, planes.shape, align_corners=False)
    ref = torch.complex(
        *(
            F.grid_sample(p, grid, padding_mode="zeros", align_corners=False)
            for p in (planes.real, planes.imag)
        )
    )
    torch.testing.assert_close(out[0], ref.permute(1, 0, 2, 3), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    ("mode", "fn"),
    [(M.VOLUME_3D, "rigid_volume_kspace"), (M.MULTISLICE_2D, "rigid_multislice_kspace")],
)
def test_translations_are_isotropic_in_mm(monkeypatch, mode, fn):
    # Same draws at (2, 1, 1) mm and (1, 1, 1) mm: through-plane voxels halve, in-plane stay.
    seen = []
    real = getattr(motion_rigid, fn)

    def capture(*args):
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(motion_rigid, fn, capture)
    x = _phantom()
    for vox in ((2.0, 1.0, 1.0), (1.0, 1.0, 1.0)):
        apply_degradation(
            "rigid_motion", x, 0.6, seed=1, geometry=AcquisitionGeometry(mode=mode, voxel_mm=vox)
        )
    aniso, iso = seen
    if mode is M.VOLUME_3D:
        torch.testing.assert_close(aniso[3][:, 0], iso[3][:, 0] / 2.0)
        torch.testing.assert_close(aniso[3][:, 1:], iso[3][:, 1:])
    else:
        torch.testing.assert_close(aniso[3], iso[3] / 2.0)  # dz per shot
        torch.testing.assert_close(aniso[4], iso[4])  # in-plane (dy, dx)
    assert float(iso[-2].abs().max()) > 5.0  # per-line draws span the declared 0.6 * 10 px
