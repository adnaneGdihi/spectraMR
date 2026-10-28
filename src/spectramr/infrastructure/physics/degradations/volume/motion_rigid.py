"""Native multislice and 3-D rigid_motion (D6): rotate, gather each shot's lines, translate per line.

As the 2-D primitive (``simulate_rigid_motion_kspace``): the lines are split into 8 shots of
consecutive acquisition time (``acquisition_timing``); each shot's pose is rendered once and its
lines gathered through ``assemble_from_states`` (one render alive at a time), then every line
gets its own translation as the exact Fourier phase ``exp(-2 pi i f . d)``. Translations are
patient motion, isotropic in mm (Ruling 34): ``U(-1, 1) * max_translation * voxel_mm[1]`` mm per
axis, converted to voxels of each axis; angles are ``U(-max_rotation, max_rotation)`` rad, drawn
per sample; rotations are rigid in mm. Without ``voxel_mm`` voxels are isotropic (px).

Multislice: each slice is its own 2-D acquisition at its own line times. A shot's pose is an
in-plane rotation plus a through-plane displacement ``dz`` that re-samples the excited
position along D; that partial-volume resampling is the whole through-plane effect here (spin
history is a separate catalogue axis).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import check_volume
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import MOTION_TR_S
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "N_SHOTS",
    "line_phase",
    "rigid_motion_multislice",
    "rigid_motion_volume",
    "rigid_multislice_kspace",
    "rigid_volume_kspace",
    "rotation_matrices",
]

_Voxel = tuple[float, float, float]
# The 2-D primitive's pose count.
N_SHOTS = 8
_VOL = (-3, -2, -1)
_SLICE = (-2, -1)


def rotation_matrices(angles: Tensor) -> Tensor:
    """``[..., 3, 3]`` rotations acting on ``(x, y, z)`` from ``angles[..., (a_z, a_y, a_x)]``.

    ``R = R_z(a_z) R_y(a_y) R_x(a_x)``; ``R_z`` is the 2-D primitive's in-plane rotation.
    """
    c, s = torch.cos(angles), torch.sin(angles)
    one, zero = torch.ones_like(c[..., 0]), torch.zeros_like(c[..., 0])

    def mat(*rows: Tensor) -> Tensor:
        return torch.stack(rows, dim=-1).unflatten(-1, (3, 3))

    rz = mat(c[..., 0], -s[..., 0], zero, s[..., 0], c[..., 0], zero, zero, zero, one)
    ry = mat(c[..., 1], zero, s[..., 1], zero, one, zero, -s[..., 1], zero, c[..., 1])
    rx = mat(one, zero, zero, zero, c[..., 2], -s[..., 2], zero, s[..., 2], c[..., 2])
    return rz @ ry @ rx


def line_phase(shifts: Tensor, shape: tuple[int, ...], device: torch.device, dtype) -> Tensor:
    """``exp(-2 pi i sum_a f_a d_a)`` ``[B, 1, D, H, W]`` for per-line shifts ``[B, n, D, H]``.

    The ``n`` channels are the trailing ``n`` axes in order (``(dz, dy, dx)`` or ``(dy, dx)``);
    the readout shift is the last.
    """
    n = shifts.shape[1]
    arg = torch.zeros((), device=device, dtype=dtype)
    for a, size in enumerate(shape[-n:]):
        view = [1] * 5
        view[5 - n + a] = size
        d = shifts[:, a].to(device=device, dtype=dtype)[:, None, :, :, None]
        arg = arg + centred_freqs(size, device, dtype=dtype).view(view) * d
    return torch.exp(-2j * math.pi * arg)


def _rotate_volume(x: Tensor, angles: Tensor, vox: _Voxel) -> Tensor:
    """Resample ``x`` [B, C, D, H, W] under rotations ``angles`` [B, 3], rigid in mm."""
    b, c, d, h, w = x.shape
    rdt = x.real.dtype
    # affine_grid works in [-1, 1] per axis; conjugating by the half-extents in mm makes it rigid.
    half = (w * vox[2], h * vox[1], d * vox[0])
    rot = rotation_matrices(angles.to(rdt))
    rows = [
        torch.stack([rot[:, i, j] * (half[j] / half[i]) for j in range(3)], -1) for i in range(3)
    ]
    theta = F.pad(torch.stack(rows, dim=-2), (0, 1))
    grid = F.affine_grid(theta, [b, 2 * c, d, h, w], align_corners=False)
    src = torch.view_as_real(x).movedim(-1, 2).reshape(b, 2 * c, d, h, w)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.view_as_complex(out.reshape(b, c, 2, d, h, w).movedim(2, -1).contiguous())


def rigid_volume_kspace(
    x: Tensor, shots: Tensor, angles: Tensor | None, shifts: Tensor, voxel_mm: _Voxel
) -> Tensor:
    """3-D k-space: shot ``shots[kz, ky]`` pose ``angles[:, shot]``, then the line's ``shifts``.

    ``shots`` [D, H] long (CPU); ``angles`` [B, S, 3] rad or ``None`` (no rotation); ``shifts``
    [B, 3, D, H] px ``(dz, dy, dx)``.
    """
    if angles is None:
        k = fftnc(x, dims=_VOL)
    else:
        k = assemble_from_states(
            angles.shape[1],
            shots,
            lambda p: _rotate_volume(x, angles[:, p], voxel_mm),
            encoding="volume",
        )
    return k * line_phase(shifts, x.shape, x.device, x.real.dtype)


def _inplane_displacement(
    angles: Tensor, dz: Tensor, shape: tuple[int, int, int], vox: _Voxel
) -> Tensor:
    """``[B, 3, D, H, W]`` pull-back displacement (voxels): in-plane rotation plus ``dz`` slices."""
    d, h, w = shape
    rdt = angles.dtype
    yy = (torch.arange(h, device=angles.device, dtype=rdt) - (h - 1) / 2.0) * vox[1]
    xx = (torch.arange(w, device=angles.device, dtype=rdt) - (w - 1) / 2.0) * vox[2]
    y, x = yy.view(1, h, 1), xx.view(1, 1, w)
    c, s = torch.cos(angles).view(-1, 1, 1), torch.sin(angles).view(-1, 1, 1)
    # Sample at R (x, y), the 2-D primitive's affine [[c, -s], [s, c]].
    dx = (c * x - s * y - x) / vox[2]
    dy = (s * x + c * y - y) / vox[1]
    full = (angles.shape[0], d, h, w)
    return torch.stack(
        [dz.view(-1, 1, 1, 1).expand(full), dy[:, None].expand(full), dx[:, None].expand(full)],
        dim=1,
    )


def rigid_multislice_kspace(
    x: Tensor, shots: Tensor, angles: Tensor, dz: Tensor, shifts: Tensor, voxel_mm: _Voxel
) -> Tensor:
    """Per-slice 2-D k-space: shot ``s`` pose ``(angles[:, s], dz[:, s])``, then line ``shifts``.

    ``shots`` [D, H] long (CPU); ``angles``, ``dz`` [B, S] (rad, slices); ``shifts`` [B, 2, D, H]
    px ``(dy, dx)``.
    """
    shape = tuple(x.shape[-3:])
    k = assemble_from_states(
        angles.shape[1],
        shots,
        lambda p: warp_volume(
            x, _inplane_displacement(angles[:, p], dz[:, p], shape, voxel_mm), padding_mode="zeros"
        ),
        encoding="slice",
    )
    return k * line_phase(shifts, x.shape, x.device, x.real.dtype)


def _shots(x: Tensor, geometry: AcquisitionGeometry) -> tuple[Tensor, int]:
    """Shot of every line (CPU ``[D, H]``) and the shot count: 8 groups of consecutive time."""
    d, h = x.shape[-3], x.shape[-2]
    n_shots = min(N_SHOTS, d * h)
    return shot_index(line_times(geometry, d, h, tr_s=MOTION_TR_S), n_shots), n_shots


def _to_voxels(shifts_px: Tensor, vox: _Voxel, axes: tuple[int, ...]) -> Tensor:
    """In-plane px (``voxel_mm[1]`` mm each) per channel -> voxels of ``axes``: isotropic in mm."""
    return torch.stack([shifts_px[:, i] * (vox[1] / vox[a]) for i, a in enumerate(axes)], dim=1)


def _limits(theta: float, max_translation: float | None, max_rotation: float | None):
    t = float(theta) * 10.0 if max_translation is None else float(max_translation)
    r = float(theta) * 0.1 if max_rotation is None else float(max_rotation)
    if t < 0 or r < 0:
        raise ValueError(f"rigid_motion: limits must be >= 0; got translation {t}, rotation {r}")
    return t, r


def rigid_motion_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_translation: float | None = None,
    max_rotation: float | None = None,
) -> Tensor:
    """D6 in 3-D: per-shot 3-D rotation, per-(kz, ky)-line 3-D translation."""
    if geometry.mode is not AcquisitionMode.VOLUME_3D:
        raise ValueError(f"rigid_motion_volume is the 3-D realisation; got {geometry.mode!r}")
    check_volume(x, geometry, "rigid_motion")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    t, r = _limits(theta, max_translation, max_rotation)
    shots, n_shots = _shots(x, geometry)
    vox = voxel_mm_or_unit(geometry.voxel_mm)
    b, _, d, h, _ = x.shape
    u = (
        per_sample_rand(
            b, 3 * n_shots + 3 * d * h, generator=_generator(x.device, seed), device=x.device
        )
        * 2.0
        - 1.0
    )
    angles = (u[:, : 3 * n_shots] * r).view(b, n_shots, 3) if r > 0.0 else None
    shifts = _to_voxels((u[:, 3 * n_shots :] * t).view(b, 3, d, h), vox, (0, 1, 2))
    return ifftnc(rigid_volume_kspace(x, shots, angles, shifts, vox), dims=_VOL)


def rigid_motion_multislice(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_translation: float | None = None,
    max_rotation: float | None = None,
) -> Tensor:
    """D6 in multislice: per-shot in-plane rotation and ``dz``, per-(slice, ky) in-plane shift."""
    if geometry.mode is not AcquisitionMode.MULTISLICE_2D:
        raise ValueError(f"rigid_motion_multislice is the multislice one; got {geometry.mode!r}")
    check_volume(x, geometry, "rigid_motion")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    t, r = _limits(theta, max_translation, max_rotation)
    shots, n_shots = _shots(x, geometry)
    vox = voxel_mm_or_unit(geometry.voxel_mm)
    b, _, d, h, _ = x.shape
    u = (
        per_sample_rand(
            b, 2 * n_shots + 2 * d * h, generator=_generator(x.device, seed), device=x.device
        )
        * 2.0
        - 1.0
    )
    angles, dz = u[:, :n_shots] * r, u[:, n_shots : 2 * n_shots] * (t * vox[1] / vox[0])
    shifts = _to_voxels((u[:, 2 * n_shots :] * t).view(b, 2, d, h), vox, (1, 2))
    return ifftnc(rigid_multislice_kspace(x, shots, angles, dz, shifts, vox), dims=_SLICE)
