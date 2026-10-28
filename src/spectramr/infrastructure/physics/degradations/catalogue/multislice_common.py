"""Shared pieces of the multislice / through-plane catalogue axes.

Slice positions are in units of the slice pitch ``voxel_mm[0]``. The slice gap is never a knob
(Ruling 49): it is ``voxel_mm[0] - slice_thickness_mm``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

__all__ = [
    "SIGMA_PER_THICKNESS",
    "hop_intervals",
    "intervals_at",
    "positive",
    "profile_response",
    "profile_taps",
    "slab_coverage",
    "thickness_mm",
    "thickness_over_pitch",
]


def positive(name: str, key: str, value: float) -> float:
    if not (isinstance(value, int | float) and not isinstance(value, bool) and value > 0):
        raise ValueError(f"{name}: {key} must be a positive number, got {value!r}")
    return float(value)


def thickness_over_pitch(geometry: AcquisitionGeometry, name: str) -> float:
    """Slice thickness in slice pitches; contiguous slices (thickness = pitch) when unset."""
    thk = geometry.slice_thickness_mm
    if thk is None:
        return 1.0
    if geometry.voxel_mm is None:
        raise ValueError(f"{name}: slice_thickness_mm needs voxel_mm to give the slice pitch")
    return thk / geometry.voxel_mm[0]


def thickness_mm(geometry: AcquisitionGeometry, name: str) -> float:
    """Slice thickness in mm: ``slice_thickness_mm``, or the pitch for contiguous slices."""
    if geometry.voxel_mm is None:
        raise ValueError(f"{name}: this needs voxel_mm (mm quantities along the slice axis)")
    return geometry.slice_thickness_mm or geometry.voxel_mm[0]


# Gaussian std of the real slice profile in slice thicknesses at theta = 1: sigma = theta * thk / 2.
SIGMA_PER_THICKNESS = 0.5


def profile_response(u: Tensor, thickness: float, sigma: float) -> Tensor:
    """Unit-height slice profile at offsets ``u``: a rect of width ``thickness`` convolved with a
    Gaussian of std ``sigma`` (same length unit), ``(erf((u + t/2)/(sqrt2 s)) - erf((u - t/2)/(sqrt2 s)))/2``
    (Bernstein et al., Handbook of MRI Pulse Sequences, 2004, sec. 2.3: finite-bandwidth profile)."""
    r = 1.0 / (math.sqrt(2.0) * sigma)
    return 0.5 * (torch.erf((u + thickness / 2.0) * r) - torch.erf((u - thickness / 2.0) * r))


def profile_taps(thickness: float, sigma: float, n_grid: int = 256) -> Tensor:
    """Weights of the slices (pitch = 1) a slice integrates: the profile averaged over each voxel
    extent, normalised to sum 1; ``[2J + 1]`` float64, symmetric, J covering 6 sigma."""
    half = math.ceil(thickness / 2.0 + 6.0 * sigma - 0.5) + 1
    j = torch.arange(-half, half + 1, dtype=torch.float64)
    u = (torch.arange(n_grid, dtype=torch.float64) + 0.5) / n_grid - 0.5
    w = profile_response(j[:, None] + u[None, :], thickness, sigma).mean(dim=-1)
    return w / w.sum()


def intervals_at(
    geometry: AcquisitionGeometry,
    d: int,
    h: int,
    tr_s: float,
    offsets: Sequence[int],
    t0: Tensor,
    cut: Tensor | None = None,
) -> Tensor:
    """Time (s) from the last excitation of slice ``k + o`` before ``cut[s]`` to ``t0[k, s]``.

    ``cut`` ``[S]`` (default ``t0``) is the time the spin left its source, so only excitations
    before it count. ``t0`` is ``[D, S]`` (s); returns ``[len(offsets), D, S]`` float64 on the CPU, ``inf`` where
    ``k + o`` is outside the stack. The acquisition repeats with its frame period, so a slice
    excited later in the frame was last excited in the previous one. ``o = 0`` is a slice's own
    previous excitation.
    """
    t = line_times(geometry, d, h, tr_s=tr_s).double()
    frame = (math.floor(float(t.max()) / tr_s + 1e-3) + 1) * tr_s
    off = torch.as_tensor(list(offsets), dtype=torch.long)
    src = torch.arange(d)[None, :] + off[:, None]
    cand = t[src.clamp(0, d - 1)]
    cand = torch.cat([cand, cand - frame], dim=-1)
    cut_ = t0[None, :, :, None] if cut is None else cut[None, None, :, None]
    gap = t0[None, :, :, None] - cand[:, :, None, :]
    gap = torch.where(cand[:, :, None, :] < cut_, gap, torch.full_like(gap, math.inf)).amin(dim=-1)
    return torch.where(((src >= 0) & (src < d))[:, :, None], gap, torch.full_like(gap, math.inf))


def hop_intervals(
    geometry: AcquisitionGeometry, d: int, h: int, tr_s: float, offsets: Sequence[int]
) -> Tensor:
    """:func:`intervals_at` on the central ky line, which sets the image contrast (Haacke et al.,
    MRI: Physical Principles and Sequence Design, 1999); ``[len(offsets), D]``."""
    t0 = line_times(geometry, d, h, tr_s=tr_s).double()[:, h // 2 : h // 2 + 1]
    return intervals_at(geometry, d, h, tr_s, offsets, t0)[..., 0]


def slab_coverage(shift_vox: Tensor, d: int, dim: int = 2) -> Tensor:
    """Fraction of each voxel along ``dim`` lying inside the slab ``[-0.5, d - 0.5]`` moved by
    ``shift_vox`` voxels (a rect slab sampled by unit voxels; exactly 1 at zero shift).

    ``shift_vox`` broadcasts against a ``(B, 1, D, H, W)`` volume; the voxel index runs on ``dim``.
    """
    k = torch.arange(d, device=shift_vox.device, dtype=shift_vox.dtype)
    view = [1] * 5
    view[dim] = d
    k = k.view(view)
    lo = torch.maximum(k - 0.5, shift_vox - 0.5)
    hi = torch.minimum(k + 0.5, shift_vox + (d - 0.5))
    return (hi - lo).clamp(0.0, 1.0)
