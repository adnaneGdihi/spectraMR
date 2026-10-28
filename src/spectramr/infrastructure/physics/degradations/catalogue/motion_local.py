"""local_intermittent_motion: a regional smooth warp during one contiguous block of acquired lines.

Swallowing and other intermittent regional motion: a seeded ellipsoidal region is displaced
while a contiguous block of lines (consecutive in acquisition time, ``line_times``) is acquired,
and every other line sees the object at rest, so k-space consistency breaks over that block only
(intermittent motion: Zaitsev, Maclaren & Herbst, JMRI 42:887, 2015).

The displacement is ``peak * (1 - rho^2)^2 * n`` inside the region (``rho`` the ellipsoidal radius,
``n`` a seeded unit direction in mm) and exactly zero outside it, so the warp has compact
support. The region is centred on a voxel, which makes the declared peak the realised one.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    derive_axis_seed,
)
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import mm_to_px, warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    MOTION_TR_S,
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    ENCODING,
    ENCODING_DIMS,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "MAX_BLOCK_FRACTION",
    "PEAK_MM",
    "degrade_local_intermittent_motion",
    "local_intermittent_motion_volume",
    "motion_block",
    "regional_displacement_mm",
]

#: Peak regional displacement at theta = 1.
PEAK_MM = 10.0
#: Fraction of the acquired lines during which the region is displaced, at theta = 1.
MAX_BLOCK_FRACTION = 0.3


def regional_displacement_mm(
    batch: int,
    shape: tuple[int, int, int],
    seed: int,
    *,
    planar: bool,
    peak_mm: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[B, 3, D, H, W]`` ``(dz, dy, dx)`` regional displacement in mm, one region per sample.

    Per sample: centre on a voxel of the central half of each axis, semi-axes 0.2-0.4 of the
    half-FOV, direction uniform on the sphere (on the in-plane circle when ``planar``). An axis
    of size 1 does not bound the region.
    """
    device = torch.device("cpu") if device is None else device
    gen = _generator(device, derive_axis_seed(seed, "region"))
    u = per_sample_rand(batch, 8, generator=gen, device=device, dtype=dtype)
    phi = 2.0 * math.pi * u[:, 6]
    cos_b = torch.zeros_like(phi) if planar else 2.0 * u[:, 7] - 1.0
    sin_b = (1.0 - cos_b**2).clamp(min=0.0).sqrt()
    direction = torch.stack([cos_b, sin_b * torch.sin(phi), sin_b * torch.cos(phi)], dim=1)

    rho2 = torch.zeros((batch, 1, 1, 1, 1), device=device, dtype=dtype)
    for a, n in enumerate(shape):
        if n == 1:
            continue
        view = [1, 1, 1, 1, 1]
        view[2 + a] = n
        coord = 2.0 * centred_freqs(n, device, dtype=dtype).view(view)
        quarter = n // 4
        centre_idx = quarter + torch.floor(u[:, a] * (n - 2 * quarter)).clamp(max=n - quarter - 1)
        centre = (2.0 * (centre_idx - n // 2) / n).view(-1, 1, 1, 1, 1)
        semi = (0.2 + 0.2 * u[:, 3 + a]).view(-1, 1, 1, 1, 1)
        rho2 = rho2 + ((coord - centre) / semi) ** 2
    # Wendland-type bump: C1, peak 1 at rho = 0, exactly zero for rho >= 1.
    bump = (1.0 - rho2).clamp(min=0.0) ** 2
    return float(peak_mm) * bump * direction.view(batch, 3, 1, 1, 1)


def motion_block(
    geometry: AcquisitionGeometry, d: int, h: int, fraction: float, seed: int, *, tr_s: float
) -> Tensor:
    """``[D, H]`` bool (CPU): the ``round(fraction * D * H)`` lines consecutive in acquisition time.

    The block sits at a seeded relative position ``v`` (one CPU draw per call, so the schedule is
    static and shared by the batch): ``start = floor(v * (n_lines - n_block))``. A fixed ``v``
    nests the blocks of increasing ``fraction``, so the corrupted line set only grows with theta.
    """
    n_lines = d * h
    n_block = round(fraction * n_lines)
    gen = _generator(torch.device("cpu"), derive_axis_seed(seed, "block"))
    v = float(torch.rand(1, generator=gen, dtype=torch.float64))
    start = math.floor(v * (n_lines - n_block))
    rank = shot_index(line_times(geometry, d, h, tr_s=tr_s), n_lines)
    return (rank >= start) & (rank < start + n_block)


def _validate(peak_mm: float, max_block_fraction: float, tr_s: float) -> None:
    if not peak_mm >= 0:
        raise ValueError(f"local_intermittent_motion: peak_mm must be >= 0; got {peak_mm}")
    if not 0 < max_block_fraction <= 1:
        raise ValueError(
            f"local_intermittent_motion: max_block_fraction must be in (0, 1]; got {max_block_fraction}"
        )
    if not tr_s > 0:
        raise ValueError(f"local_intermittent_motion: tr_s must be > 0; got {tr_s}")


def local_intermittent_motion_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    peak_mm: float = PEAK_MM,
    max_block_fraction: float = MAX_BLOCK_FRACTION,
    tr_s: float = MOTION_TR_S,
) -> Tensor:
    """Intermittent regional motion of ``x`` ``(B, C, D, H, W)`` in any mode (``D == 1``: slice)."""
    check_input(x, geometry, "local_intermittent_motion")
    _validate(peak_mm, max_block_fraction, tr_s)
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    b, _, d, h, _ = x.shape
    block = motion_block(geometry, d, h, float(theta) * max_block_fraction, seed, tr_s=tr_s)
    if not bool(block.any()):
        return x
    disp_mm = regional_displacement_mm(
        b,
        (d, h, x.shape[-1]),
        seed,
        planar=geometry.mode is AcquisitionMode.SLICE_2D,
        peak_mm=float(theta) * peak_mm,
        device=x.device,
        dtype=x.real.dtype,
    )
    disp = mm_to_px(disp_mm, voxel_mm_or_unit(geometry.voxel_mm))
    k = assemble_from_states(
        2,
        block.long(),
        lambda p: warp_volume(x, disp) if p == 1 else x,
        encoding=ENCODING[geometry.mode],
    )
    return ifftnc(k, dims=ENCODING_DIMS[geometry.mode])


def degrade_local_intermittent_motion(
    x: Tensor,
    theta: float,
    seed: int = 0,
    voxel_mm: tuple[float, float, float] | None = None,
    **kwargs,
) -> Tensor:
    """The 2-D kernel: a ``(B, C, H, W)`` slice, lines timed per ``ky`` (1 mm pixels if unknown)."""
    return run_as_volume(local_intermittent_motion_volume, x, theta, seed, voxel_mm, **kwargs)
