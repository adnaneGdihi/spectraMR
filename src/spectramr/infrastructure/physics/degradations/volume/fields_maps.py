"""Native multislice and 3-D field maps: bias (D11), coil miscalibration (D18), concomitant (D23).

Each field is one realisation per subject on its ``(D, H, W)`` grid, so a multislice stack
samples the same 3-D field a volume does: the map depends on the voxel's position, never on
the mode. The 2-D cosine-mode families gain a slice-axis (z) mode at the same amplitude, so
the declared ranges hold unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.maxwell_concomitant import maxwell_concomitant_field_hz
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "bias_volume",
    "check_volume",
    "coil_miscal_volume",
    "concomitant_volume",
    "field_coords",
]

_MODES = frozenset({AcquisitionMode.MULTISLICE_2D, AcquisitionMode.VOLUME_3D})


def check_volume(x: Tensor, geometry: AcquisitionGeometry, name: str) -> None:
    """Raise unless ``x`` is ``(B, C, D, H, W)`` in a multislice or 3-D geometry."""
    if x.dim() != 5:
        raise ValueError(f"{name} expects (B, C, D, H, W), got {tuple(x.shape)}")
    if geometry.mode not in _MODES:
        raise ValueError(f"{name} is the multislice / 3-D realisation; got mode {geometry.mode!r}")


def field_coords(
    shape: tuple[int, int, int],
    voxel_mm: tuple[float, float, float] | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(z, y, x)`` grids, each ``[D, H, W]``, on which the smooth fields are drawn.

    Without ``voxel_mm`` each axis is normalised to its own FOV (``[-1, 1)``, as in 2-D). With
    it, positions are mm divided by the largest half-extent, so a field is isotropic in mm and
    thick slices sample it at their true spacing.
    """
    if voxel_mm is None:
        return mm_grid(shape, None, device=device, dtype=dtype)
    z, y, x = mm_grid(shape, voxel_mm, device=device, dtype=dtype)
    half = max((n // 2) * v for n, v in zip(shape, voxel_mm, strict=True)) or 1.0
    return z / half, y / half, x / half


def bias_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_log_b: float = 1.2,
) -> Tensor:
    """D11: ``exp(theta * max_log_b * cos(x-mode) cos(y-mode) cos(z-mode))`` per subject."""
    check_volume(x, geometry, "bias")
    if float(theta) <= 0.0:
        return x
    x = _ensure_complex(x)
    b = x.shape[0]
    z, y, xx = field_coords(tuple(x.shape[-3:]), geometry.voxel_mm, x.device)
    # Per sample: frequencies (fx, fy, fz) in [0, 2) half-cycles/FOV, then their phases.
    r = per_sample_rand(b, 6, 1, 1, 1, generator=_generator(x.device, seed), device=x.device)
    f, ph = r[:, :3] * 2.0, r[:, 3:] * (2.0 * math.pi)
    log_b = (
        float(theta)
        * max_log_b
        * torch.cos(f[:, 0] * math.pi * xx + ph[:, 0])
        * torch.cos(f[:, 1] * math.pi * y + ph[:, 1])
        * torch.cos(f[:, 2] * math.pi * z + ph[:, 2])
    )
    return x * torch.exp(log_b).unsqueeze(1)


def coil_miscal_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_relative_error: float = 1.0,
) -> Tensor:
    """D18: ratio map ``1 + theta * max_relative_error * mean_m cos(pi f_m . r + phi_m)``."""
    check_volume(x, geometry, "coil_miscal")
    if float(theta) <= 0.0:
        return x
    x = _ensure_complex(x)
    b = x.shape[0]
    z, y, xx = field_coords(tuple(x.shape[-3:]), geometry.voxel_mm, x.device)
    # Per sample: [fx/fy/fz/phase, 4 modes]; frequencies in [0, 3) as in 2-D.
    p = per_sample_rand(b, 4, 4, 1, 1, 1, generator=_generator(x.device, seed), device=x.device)
    f, ph = p[:, :3] * 3.0, p[:, 3] * (2.0 * math.pi)
    arg = math.pi * (f[:, 0] * xx + f[:, 1] * y + f[:, 2] * z) + ph
    err = torch.cos(arg).mean(dim=1, keepdim=True)
    return x * (1.0 + float(theta) * max_relative_error * err)


def concomitant_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    b0_T: float = 0.3,  # noqa: N803
    max_gradient_mT_per_m: float = 30.0,  # noqa: N803
    duration_ms: float = 5.0,
) -> Tensor:
    """D23: phase ``exp(+2 pi i f_c tau)`` of the Maxwell field, magnitude untouched.

    The 2-D primitive's ``gamma G^2 tau (x^2 + y^2) / (8 B0)`` is the Maxwell term of a
    slice-axis gradient, so ``(Gx, Gy, Gz) = (0, 0, theta * max_gradient)``: the phase grows
    as the in-plane ``r^2`` and is the same on every slice. Distances need ``voxel_mm``.
    """
    check_volume(x, geometry, "concomitant_phase")
    if geometry.b0_axis != "slice":
        raise ValueError(
            f"concomitant_phase supports b0_axis='slice' only, got {geometry.b0_axis!r}: the "
            "Maxwell field model fixes B0 along the slice axis (D)"
        )
    if geometry.voxel_mm is None:
        raise ValueError(
            f"concomitant_phase in {geometry.mode.value} needs geometry.voxel_mm: the phase "
            "depends on the physical distance from isocentre"
        )
    if float(theta) <= 0.0:
        return x
    x = _ensure_complex(x)
    grad = float(theta) * max_gradient_mT_per_m
    f_hz = maxwell_concomitant_field_hz(
        tuple(x.shape[-3:]),
        geometry.voxel_mm,
        (0.0, 0.0, grad),
        b0_T,
        device=x.device,
        dtype=x.real.dtype,
    )
    return x * torch.exp(1j * (2.0 * math.pi * duration_ms * 1e-3) * f_hz)
