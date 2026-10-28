"""gradwarp_residual: what the vendor gradient-nonlinearity correction leaves behind.

The scanner distorts by the displacement ``d_c(r)`` of ``gradient_nonlinearity`` (order 2..3 polynomial of
position, coefficients ``c``); the vendor correction undoes ``c (1 + eps)``, so the net displacement is
``-eps d_c(r)`` (Janke et al., MRM 52:115, 2004; Doran et al., Phys Med Biol 50:1343,
2005). It is applied as ONE resampling by the net displacement: distortion then correction as two
resamplings differs from it only at second order, ``eps (d_c . grad) d_c``, and would add a second
trilinear-interpolation blur that no scanner has. ``eps = theta eps_max``, eps_max = 0.5 (eps = 1 is no correction at all, so half the distortion is left, e.g. a wrong coefficient file); the seed draws the same ``c``
as ``gradient_nonlinearity`` at that seed, so the residual is relative to that axis's own field.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.degradations.volume.geometric import gnl_displacement_px

__all__ = ["gradwarp_residual_2d", "gradwarp_residual_volume", "residual_displacement_px"]


def residual_displacement_px(
    batch: int,
    shape: tuple[int, int, int],
    epsilon: float,
    seed: int,
    device: torch.device,
    voxel_mm: tuple[float, float, float] | None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[B, 3, D, H, W]`` ``(dz, dy, dx)`` net displacement ``-epsilon d_c`` in voxels."""
    d_c = gnl_displacement_px(batch, shape, 1.0, seed, device, voxel_mm=voxel_mm, dtype=dtype)
    return -float(epsilon) * d_c


def gradwarp_residual_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, eps_max: float = 0.5
) -> Tensor:
    """Resample ``x`` by the residual ``-eps d_c(r)``, ``eps = theta eps_max``; mm -> voxels by ``voxel_mm``."""
    check_input(x, geometry, "gradwarp_residual")
    if eps_max < 0:
        raise ValueError(f"gradwarp_residual: eps_max must be >= 0; got {eps_max}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    d = residual_displacement_px(
        x.shape[0],
        tuple(x.shape[-3:]),
        float(theta) * eps_max,
        seed,
        x.device,
        geometry.voxel_mm,
        x.real.dtype,
    )
    return warp_volume(x, d)


def gradwarp_residual_2d(x: Tensor, theta: float, seed: int = 0, voxel_mm=None, **kwargs) -> Tensor:
    return run_as_volume(gradwarp_residual_volume, x, theta, seed, voxel_mm, **kwargs)
