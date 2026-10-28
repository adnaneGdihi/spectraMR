"""Native multislice and 3-D magic-angle T2 modulation (D22).

The 2-D model (``simulate_magic_angle``) generalised to a volume: ellipsoidal collagen patches
centred on the 3-D anatomy support, a collagen orientation field smooth in 3-D, and the
dipolar T2 lengthening near 54.7 degrees as a real positive (magnitude-only) boost. One
realisation per subject, shared by multislice and 3-D.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.degradations.volume.fields_maps import (
    check_volume,
    field_coords,
)

__all__ = ["anatomy_support_3d", "magic_angle_volume"]

# Patches per evaluation chunk: bounds the [chunk, D, H, W] ellipsoid buffer.
_PATCH_CHUNK = 16
_MAX_PATCHES = 512


def _box_volume(shape: tuple[int, int, int], voxel_mm: tuple[float, float, float] | None) -> float:
    """Volume of the FOV in the units of ``field_coords`` (8 when each axis spans 2 units)."""
    if voxel_mm is None:
        return 8.0
    half = max((n // 2) * v for n, v in zip(shape, voxel_mm, strict=True)) or 1.0
    return math.prod(n * v / half for n, v in zip(shape, voxel_mm, strict=True))


def anatomy_support_3d(
    x: Tensor, foreground_mask: Tensor | None = None, rel_threshold: float = 0.05
) -> Tensor:
    """``[B, D, H, W]`` support: magnitude in any channel above ``rel_threshold`` of the volume peak.

    As in 2-D, an all-zero sample falls back to the whole FOV before an explicit
    ``foreground_mask`` (``[B|1, (1,) D, H, W]``) is intersected in.
    """
    mag = x.abs().amax(dim=1)
    peak = mag.amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-8)
    support = (mag > rel_threshold * peak).to(mag.dtype)
    empty = support.flatten(1).sum(-1) < 1.0
    support = torch.where(empty.view(-1, 1, 1, 1), torch.ones_like(support), support)
    if foreground_mask is not None:
        fg = foreground_mask.to(dtype=support.dtype, device=support.device)
        if fg.dim() == 5:
            fg = fg.squeeze(1)
        if fg.dim() != 4 or fg.shape[-3:] != support.shape[-3:]:
            raise ValueError(
                f"magic_angle: foreground_mask must be [B|1, 1, D, H, W]; got {tuple(fg.shape)}"
            )
        support = support * fg
    return support


def _collagen_mask(
    sup: Tensor,
    coords: tuple[Tensor, Tensor, Tensor],
    box: float,
    collagen_fraction: float,
    g: torch.Generator,
) -> Tensor:
    """``[D, H, W]`` union of ellipsoids centred on support voxels, covering ~the fraction."""
    z, y, x = coords
    radius_scale = 1.0 + 2.0 * max(0.0, collagen_fraction - 0.05)
    # The 2-D in-plane radii (y, x), with z drawn like y.
    lo_r, span_r = (0.02, 0.02, 0.03), (0.1, 0.1, 0.15)
    avg_vol = (
        4.0
        / 3.0
        * math.pi
        * math.prod((a + 0.5 * s) * radius_scale for a, s in zip(lo_r, span_r, strict=True))
    )
    # The one host sync: the patch count follows this subject's support, as in 2-D.
    support_frac = float(sup.mean().clamp(min=1e-3))
    n = min(_MAX_PATCHES, max(1, round(collagen_fraction * support_frac * box / avg_vol)))
    idx = torch.multinomial(sup.reshape(-1) + 1e-12, n, replacement=True, generator=g)
    centres = torch.stack([c.reshape(-1)[idx] for c in coords], dim=1)  # [n, 3]
    u = torch.rand(n, 3, generator=g, device=sup.device).unbind(1)
    radii = torch.stack(
        [(a + s * v) * radius_scale for a, s, v in zip(lo_r, span_r, u, strict=True)], dim=1
    )
    inside = torch.zeros(sup.shape, dtype=torch.bool, device=sup.device)
    for start in range(0, n, _PATCH_CHUNK):
        c = centres[start : start + _PATCH_CHUNK].view(-1, 3, 1, 1, 1)
        r = radii[start : start + _PATCH_CHUNK].view(-1, 3, 1, 1, 1)
        rho = sum(((a - c[:, k]) / r[:, k]) ** 2 for k, a in enumerate((z, y, x)))
        inside |= rho.lt(1.0).any(0)
    return inside.to(sup.dtype) * sup


def magic_angle_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    te: float = 20.0,
    t2_base: float = 10.0,
    t2_magic: float = 30.0,
    collagen_fraction: float = 0.30,
    tissue_class: str = "msk",
    foreground_mask: Tensor | None = None,
) -> Tensor:
    """D22: ``|x|`` boosted by ``exp(-TE/T2_eff) / exp(-TE/T2_base)`` inside collagen patches."""
    check_volume(x, geometry, "magic_angle")
    x = _ensure_complex(x)
    # The effect needs ordered collagen: outside MSK it does not exist (as in 2-D).
    if float(theta) <= 0.0 or tissue_class != "msk":
        return x
    if not 0.0 < collagen_fraction <= 1.0:
        raise ValueError(
            f"magic_angle: collagen_fraction must be in (0, 1]; got {collagen_fraction}"
        )
    t2_max = t2_base + (float(t2_magic) - t2_base) * float(theta)
    b = x.shape[0]
    shape = tuple(x.shape[-3:])
    coords = field_coords(shape, geometry.voxel_mm, x.device)
    support = anatomy_support_3d(x, foreground_mask)
    box = _box_volume(shape, geometry.voxel_mm)
    g = _generator(x.device, seed)
    orient, masks = [], []
    for i in range(b):
        # Orientation field: a sinusoid of 1-4 half-cycles along a random 3-D direction.
        u = torch.randn(3, generator=g, device=x.device)
        u = u / u.norm().clamp_min(1e-8)
        freq, phase = torch.rand(2, generator=g, device=x.device).unbind()
        freq, phase = 3.0 * freq, (2.0 * math.pi) * phase
        arg = (freq + 1.0) * math.pi * (u[0] * coords[0] + u[1] * coords[1] + u[2] * coords[2])
        orient.append(math.pi / 2 * torch.sin(arg + phase))
        masks.append(_collagen_mask(support[i], coords, box, collagen_fraction, g))
    angle, col_mask = torch.stack(orient), torch.stack(masks)  # [B, D, H, W]
    dipolar = torch.abs(3.0 * torch.cos(angle) ** 2 - 1.0)
    t2_eff = t2_base + (t2_max - t2_base) * torch.exp(-3.0 * dipolar)
    boost = torch.exp(-te / t2_eff) / math.exp(-te / t2_base)
    modulation = 1.0 + (boost - 1.0) * col_mask
    return x * modulation.unsqueeze(1).to(x.real.dtype)
