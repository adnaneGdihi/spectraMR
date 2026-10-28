"""Per-sample smooth random fields on a ``(D, H, W)`` grid, each with unit peak magnitude.

Every draw goes through ``per_sample_rng``, so a sample's field does not depend on the batch.
"""

from __future__ import annotations

import math
from itertools import product

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import centred_freqs
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand, per_sample_randn

__all__ = ["mm_grid", "polynomial_field", "smooth_random_field", "upsampled_noise_field"]

_Shape = tuple[int, int, int]


def mm_grid(
    shape: _Shape,
    voxel_mm: tuple[float, float, float] | None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(z, y, x)`` coordinate grids, each ``[D, H, W]``, zero at index ``N // 2`` per axis.

    With ``voxel_mm`` the coordinates are in mm (spacing = voxel size). With ``voxel_mm=None``
    no physical spacing is assumed: the coordinates are normalised to ``[-1, 1)``
    (``2 * centred_freqs(N)``), so one FOV spans 2 units along every axis.
    """
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError(f"shape must be three positive sizes (D, H, W); got {shape!r}")
    if voxel_mm is None:
        axes = [2.0 * centred_freqs(n, device=device, dtype=dtype) for n in shape]
    else:
        if len(voxel_mm) != 3 or any(v <= 0 for v in voxel_mm):
            raise ValueError(f"voxel_mm must be three positive sizes; got {voxel_mm!r}")
        axes = [
            (torch.arange(n, device=device, dtype=dtype) - n // 2) * v
            for n, v in zip(shape, voxel_mm, strict=True)
        ]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return z, y, x


def _unit_peak(field: torch.Tensor) -> torch.Tensor:
    """Scale each sample of a ``[B, 1, D, H, W]`` field to peak ``|value| = 1``."""
    peak = field.abs().amax(dim=(1, 2, 3, 4), keepdim=True)
    return field / peak.clamp_min(torch.finfo(field.dtype).tiny)


def polynomial_field(
    batch: int,
    shape: _Shape,
    *,
    order: int = 2,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random polynomial of total degree ``1..order`` in normalised coords, zero mean, unit peak."""
    if order < 1:
        raise ValueError(f"order must be >= 1; got {order}")
    z, y, x = mm_grid(shape, None, device=device, dtype=dtype)
    exponents = [e for e in product(range(order + 1), repeat=3) if 1 <= sum(e) <= order]
    coeffs = per_sample_randn(
        batch, len(exponents), generator=generator, device=device, dtype=dtype
    )
    # Term-by-term accumulation: a batched matmul would make sample b depend on the batch size.
    field = torch.zeros(batch, 1, *shape, device=device, dtype=dtype)
    for t, (a, b, c) in enumerate(exponents):
        field = field + coeffs[:, t].view(batch, 1, 1, 1, 1) * (z**a * y**b * x**c)
    field = field - field.mean(dim=(2, 3, 4), keepdim=True)
    return _unit_peak(field)


def smooth_random_field(
    batch: int,
    shape: _Shape,
    *,
    n_modes: int = 4,
    max_cycles: float = 1.5,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sum of ``n_modes`` cosines, each up to ``max_cycles`` cycles/FOV per axis, random phase."""
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}")
    if max_cycles <= 0:
        raise ValueError(f"max_cycles must be > 0; got {max_cycles}")
    z, y, x = mm_grid(shape, None, device=device, dtype=dtype)
    # Per mode: (fz, fy, fx) uniform in [-max_cycles, max_cycles], then a phase in [0, 2*pi).
    draws = per_sample_rand(batch, n_modes, 4, generator=generator, device=device, dtype=dtype)
    freqs = (2.0 * draws[..., :3] - 1.0) * max_cycles
    phases = 2.0 * math.pi * draws[..., 3]
    field = torch.zeros(batch, *shape, device=device, dtype=dtype)
    view = (batch, 1, 1, 1)
    for m in range(n_modes):
        # A normalised coordinate spans 2 per FOV, so f cycles/FOV is pi * f * u radians.
        arg = math.pi * (
            freqs[:, m, 0].view(view) * z
            + freqs[:, m, 1].view(view) * y
            + freqs[:, m, 2].view(view) * x
        )
        field = field + torch.cos(arg + phases[:, m].view(view))
    return _unit_peak(field.unsqueeze(1))


def upsampled_noise_field(
    batch: int,
    shape: _Shape,
    *,
    coarse: _Shape = (4, 8, 8),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A coarse Gaussian grid trilinearly upsampled to ``shape``, unit peak per sample."""
    if len(coarse) != 3 or any(n < 1 for n in coarse):
        raise ValueError(f"coarse must be three positive sizes; got {coarse!r}")
    noise = per_sample_randn(batch, 1, *coarse, generator=generator, device=device, dtype=dtype)
    field = F.interpolate(noise, size=tuple(shape), mode="trilinear", align_corners=True)
    return _unit_peak(field)
