"""The one owner of 3-D warping: pull-back resampling of a ``(B, C, D, H, W)`` volume."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

__all__ = ["PADDING_MODES", "device_constants", "mm_to_px", "warp_volume"]

PADDING_MODES = frozenset({"zeros", "border", "reflection"})


def device_constants(
    values: Sequence[float], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """A small constant vector built by fill kernels on ``device``, not a host-to-device copy.

    ``torch.tensor(values, device=cuda)`` copies from pageable host memory, which syncs the
    stream; the values are the same, so results stay bit-identical.
    """
    return torch.cat([torch.full((1,), float(v), device=device, dtype=dtype) for v in values])


def _identity_grid(
    shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """``[D, H, W, 3]`` identity grid in ``grid_sample`` units, channels ``(x, y, z)``."""
    axes = [torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flip(-1)


def warp_volume(
    x: torch.Tensor, displacement_px: torch.Tensor, *, padding_mode: str = "border"
) -> torch.Tensor:
    """Resample ``x`` at ``r + d(r)``: ``out(r) = x(r + d(r))`` (trilinear, pull-back).

    Args:
        x: ``[B, C, D, H, W]`` volume, real or complex.
        displacement_px: ``[B, 3, D, H, W]`` displacement, channels ``(dz, dy, dx)`` in voxels
            of ``(D, H, W)``. A +1 ``dx`` makes ``out[..., w] = x[..., w + 1]``.
        padding_mode: ``grid_sample`` padding for samples outside the volume.

    Returns:
        The warped volume, same shape and dtype as ``x``. ``align_corners=True`` makes a zero
        displacement the identity up to one rounding of the grid unnormalisation.
    """
    if padding_mode not in PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(PADDING_MODES)}; got {padding_mode!r}"
        )
    if x.ndim != 5:
        raise ValueError(f"x must be (B, C, D, H, W); got {tuple(x.shape)}")
    b, c, d, h, w = x.shape
    if displacement_px.shape != (b, 3, d, h, w):
        raise ValueError(
            f"displacement_px must be {(b, 3, d, h, w)}; got {tuple(displacement_px.shape)}"
        )
    is_complex = x.is_complex()
    real_dtype = x.real.dtype if is_complex else x.dtype
    # One voxel is 2 / (N - 1) grid units (divided, not multiplied by a rounded reciprocal);
    # an axis of size 1 has no extent to shift along.
    sizes = (w, h, d)
    steps = device_constants([max(n - 1, 1) for n in sizes], x.device, real_dtype)
    moves = device_constants([float(n > 1) for n in sizes], x.device, real_dtype)
    flow = displacement_px.to(device=x.device, dtype=real_dtype).flip(1).movedim(1, -1)
    grid = _identity_grid((d, h, w), x.device, real_dtype) + 2.0 * flow / steps * moves

    # Real and imaginary parts ride as extra channels of one grid_sample call.
    src = torch.view_as_real(x).movedim(-1, 2).reshape(b, 2 * c, d, h, w) if is_complex else x
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    if is_complex:
        out = torch.view_as_complex(out.reshape(b, c, 2, d, h, w).movedim(2, -1).contiguous())
    return out


def mm_to_px(displacement_mm: torch.Tensor, voxel_mm: tuple[float, float, float]) -> torch.Tensor:
    """Convert a ``[B, 3, D, H, W]`` ``(dz, dy, dx)`` displacement from mm to voxels."""
    if len(voxel_mm) != 3 or any(v <= 0 for v in voxel_mm):
        raise ValueError(f"voxel_mm must be three positive sizes (slice, pe, ro); got {voxel_mm!r}")
    if displacement_mm.ndim != 5 or displacement_mm.shape[1] != 3:
        raise ValueError(
            f"displacement_mm must be (B, 3, D, H, W); got {tuple(displacement_mm.shape)}"
        )
    scale = device_constants(voxel_mm, displacement_mm.device, displacement_mm.dtype)
    return displacement_mm / scale.view(1, 3, 1, 1, 1)
