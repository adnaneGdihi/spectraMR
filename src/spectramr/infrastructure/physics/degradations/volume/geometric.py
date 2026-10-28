"""Native multislice and 3-D geometric operators: geometric_warp (D20), gradient_nonlinearity (D21).

Each draws one 3-D displacement field per subject and resamples through ``warp_volume``, so a
multislice stack and a volume see the same field: in multislice the through-plane component
moves signal between thick slices, which is coarse by nature (one sample per slice).
"""

from __future__ import annotations

import math
from itertools import product

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import mm_to_px, warp_volume
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import (
    check_volume,
    field_coords,
)
from spectramr.infrastructure.physics.digital_twin_simulator import _GNL_REFERENCE_GRID
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand, per_sample_randn

__all__ = [
    "geometric_warp_volume",
    "gnl_displacement_px",
    "gradient_nonlinearity_volume",
    "warp_displacement_mm",
]

_Shape = tuple[int, int, int]
_Voxel = tuple[float, float, float]
# theta = 1 coefficient scale of the 2-D kernel (``_delegate_gradient_nl``).
_GNL_MAX_SEVERITY = 7e-5
# Monomials z^a y^b x^c of total order 2..3: order 0 is a translation and order 1 a gradient
# scale/shear error, neither of which is non-linearity.
_GNL_EXPONENTS = tuple(e for e in product(range(4), repeat=3) if 2 <= sum(e) <= 3)


def warp_displacement_mm(
    batch: int,
    shape: _Shape,
    voxel_mm: _Voxel | None,
    theta: float,
    seed: int,
    device: torch.device,
    *,
    max_amplitude_px: float = 16.0,
    n_modes: int = 5,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[B, 3, D, H, W]`` ``(dz, dy, dx)`` warp in mm, peak ``|d| = theta * max_amplitude_px * pe``.

    The 2-D mode family (sums of ``n_modes`` sin/cos modes up to 3 half-cycles) extended to
    ``(z, y, x)``, drawn on coordinates isotropic in mm. The declared unit is pixels of the
    in-plane grid, so the peak in mm is ``theta * max_amplitude_px * voxel_mm[1]``.
    """
    if n_modes < 1:
        raise ValueError(f"geometric_warp: n_modes must be >= 1; got {n_modes}")
    if max_amplitude_px < 0:
        raise ValueError(f"geometric_warp: max_amplitude_px must be >= 0; got {max_amplitude_px}")
    vox = voxel_mm_or_unit(voxel_mm)
    z, y, x = field_coords(shape, vox, device, dtype)
    p = per_sample_rand(
        batch, 4, n_modes, 1, 1, 1, generator=_generator(device, seed), device=device, dtype=dtype
    )
    fz, fy, fx, ph = p[:, 0] * 3.0, p[:, 1] * 3.0, p[:, 2] * 3.0, p[:, 3] * (2.0 * math.pi)
    pi = math.pi
    dx = torch.sin(pi * (fx * x + fy * y + fz * z) + ph).sum(1)
    dy = torch.cos(pi * (fx * x - fy * y + fz * z) + ph).sum(1)
    dz = torch.cos(pi * (fx * x + fy * y - fz * z) + ph).sum(1)
    # An axis of one sample cannot move, so it does not count towards the realised peak.
    d = torch.stack([c * float(n > 1) for c, n in zip((dz, dy, dx), shape, strict=True)], dim=1)
    peak = d.pow(2).sum(1, keepdim=True).sqrt().amax(dim=(2, 3, 4), keepdim=True)
    target = float(theta) * max_amplitude_px * vox[1]
    return d * (target / peak.clamp_min(torch.finfo(dtype).tiny))


def geometric_warp_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_amplitude_px: float = 16.0,
    n_modes: int = 5,
) -> Tensor:
    """D20: smooth 3-D warp, one realisation per subject, peak displacement as declared in mm."""
    check_volume(x, geometry, "geometric_warp")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    vox = voxel_mm_or_unit(geometry.voxel_mm)
    d_mm = warp_displacement_mm(
        x.shape[0],
        tuple(x.shape[-3:]),
        vox,
        theta,
        seed,
        x.device,
        max_amplitude_px=max_amplitude_px,
        n_modes=n_modes,
        dtype=x.real.dtype,
    )
    return warp_volume(x, mm_to_px(d_mm, vox))


def gnl_displacement_px(
    batch: int,
    shape: _Shape,
    theta: float,
    seed: int,
    device: torch.device,
    voxel_mm: _Voxel | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[B, 3, D, H, W]`` ``(dz, dy, dx)`` gradient non-linearity warp in voxels.

    An order 2..3 polynomial of position with every coefficient ``N(0, (theta * 7e-5 * 256)^2)``
    as in 2-D, so the displacement is a fixed fraction of the FOV at every grid size. With
    ``voxel_mm`` the position is in mm normalised by the largest half-extent (``field_coords``),
    so the field depends on where a voxel sits in mm, not on the slab's extent or slice
    thickness. Without it each axis is normalised to its own FOV (2 units over N voxels).
    """
    if voxel_mm is None:
        z, y, x = mm_grid(shape, None, device=device, dtype=dtype)
        to_px = [n / 2.0 for n in shape]
    else:
        z, y, x = field_coords(shape, voxel_mm, device, dtype)
        # Same half-extent as field_coords: normalised units -> mm -> voxels of each axis.
        half = max((n // 2) * v for n, v in zip(shape, voxel_mm, strict=True)) or 1.0
        to_px = [half / v for v in voxel_mm]
    scale = float(theta) * _GNL_MAX_SEVERITY * _GNL_REFERENCE_GRID
    coeffs = per_sample_randn(
        batch,
        3,
        len(_GNL_EXPONENTS),
        generator=_generator(device, seed),
        device=device,
        dtype=dtype,
    )
    coeffs = coeffs * scale
    d = torch.zeros(batch, 3, *shape, device=device, dtype=dtype)
    for t, (a, b, c) in enumerate(_GNL_EXPONENTS):
        d = d + coeffs[:, :, t].view(batch, 3, 1, 1, 1) * (z**a * y**b * x**c)
    return torch.cat([d[:, k : k + 1] * f for k, f in enumerate(to_px)], dim=1)


def gradient_nonlinearity_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    severity: float | None = None,
) -> Tensor:
    """D21: 3-D polynomial warp; multislice slices curve through-plane (the "potato chip")."""
    check_volume(x, geometry, "gradient_nonlinearity")
    if severity is not None:
        raise ValueError(
            "gradient_nonlinearity: the severity= override is 2-D only; in volume modes theta "
            "sets the coefficient scale (theta * 7e-5)"
        )
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    d = gnl_displacement_px(
        x.shape[0],
        tuple(x.shape[-3:]),
        theta,
        seed,
        x.device,
        voxel_mm=geometry.voxel_mm,
        dtype=x.real.dtype,
    )
    return warp_volume(x, d)
