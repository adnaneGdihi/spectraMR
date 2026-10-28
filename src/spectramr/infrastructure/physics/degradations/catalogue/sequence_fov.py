"""fov_foldover: wrap-around of anatomy outside a prescribed field of view smaller than the object.

Pixel spacing is fixed and only the line count drops, so the reconstruction is the centred window of
width ``FOV'`` holding the aliased sum ``sum_k x(y + k FOV')`` (Fourier sampling theorem; Haacke et al.,
MRI: Physical Principles and Sequence Design, Wiley 1999, aliasing); the rest of the matrix is not acquired
(zero). ``FOV' = E (1 - 0.5 theta)`` with ``E`` the object's extent along the axis, so the wrap scales with the
anatomy, and the window is centred on the object's support, as a technologist centres the prescription. The replicas ``k = -2..2`` are fractional Fourier shifts (shift theorem ``fft(x) exp(-2 pi i f d)``
moves by ``+d`` px), which sum to the real filter ``1 + 2 cos a + 2 cos 2a``, ``a = 2 pi f FOV'``; ``K = 2``
is exact for ``FOV' >= E / 2``. The axis is zero-padded to three times its length so a shift never wraps content in.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc

__all__ = [
    "EXTENT_FRACTION",
    "MAX_REDUCTION",
    "N_REPLICAS",
    "degrade_fov_foldover",
    "fold_axis",
    "fov_foldover_volume",
    "object_extent",
]

#: FOV' is this fraction of the object's extent smaller at theta = 1.
MAX_REDUCTION = 0.5
#: The object's support is where the PE profile exceeds this fraction of its maximum.
EXTENT_FRACTION = 0.1
#: Replicas k = -N..N summed; exact for FOV' >= E / N.
N_REPLICAS = 2
_AXES = {"pe": (-2,), "slice": (-3,), "both": (-3, -2)}


def object_extent(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """``([B] centre, [B] extent)`` in pixels of the object's support along ``dim``.

    The projection ``sum |x|`` over every other axis (readout, coils, slices) is thresholded at
    ``EXTENT_FRACTION`` of its own maximum, so a noise floor of a few per cent does not widen it; the
    extent is the span of the support and the centre its midpoint (an array index).
    """
    xm = x.abs().movedim(dim, -1)
    profile = xm.sum(dim=tuple(range(1, xm.dim() - 1)))
    on = profile > EXTENT_FRACTION * profile.amax(dim=1, keepdim=True)
    idx = torch.arange(on.shape[1], device=x.device)
    last = (on * idx).amax(dim=1)
    first = torch.where(on, idx, on.shape[1]).amin(dim=1)
    return ((first + last) / 2.0).to(torch.float32), (last - first + 1).clamp(min=1).to(
        torch.float32
    )


def fold_axis(
    x: Tensor, dim: int, theta: float, support: tuple[Tensor, Tensor] | None = None
) -> Tensor:
    """Fold ``x`` along ``dim`` at FOV' = E (1 - 0.5 theta) pixels (per sample), window centred on the support."""
    n = x.shape[dim]
    if n == 1:
        return x
    centre, ext = object_extent(x, dim) if support is None else support
    fov = ext * (1.0 - MAX_REDUCTION * theta)  # [B]
    xm = x.movedim(dim, -1)
    view = (-1,) + (1,) * (xm.dim() - 1)
    f = centred_freqs(3 * n, x.device)
    a = 2.0 * math.pi * f * fov.view(view)
    gain = 1.0 + 2.0 * sum(torch.cos(k * a) for k in range(1, N_REPLICAS + 1))
    padded = F.pad(xm.to(torch.complex64), (n, n))
    folded = ifftnc(fftnc(padded, (-1,)) * gain, (-1,))[..., n : 2 * n]
    y = torch.arange(n, device=x.device, dtype=torch.float32) - centre.view(view)
    half = fov.view(view) / 2
    out = folded * ((y >= -half) & (y < half))
    out = out if x.is_complex() else out.real
    return out.movedim(-1, dim)


def fov_foldover_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, *, axes: str = "pe"
) -> Tensor:
    check_input(x, geometry, "fov_foldover")
    if axes not in _AXES:
        raise ValueError(f"axes must be one of {sorted(_AXES)}, got {axes!r}")
    if -3 in _AXES[axes] and geometry.mode is not AcquisitionMode.VOLUME_3D:
        raise ValueError(
            f"axes={axes!r} folds kz, which only a 3-D acquisition encodes; "
            f"mode {geometry.mode.value} can fold 'pe' only"
        )
    if theta == 0:
        return x
    supports = {dim: object_extent(x, dim) for dim in _AXES[axes]}
    for dim in _AXES[axes]:
        x = fold_axis(x, dim, theta, supports[dim])
    return x


def degrade_fov_foldover(x: Tensor, theta: float, seed: int = 0, *, axes: str = "pe") -> Tensor:
    """2-D slice ``(B, C, H, W)``: fold along phase encode (dim -2)."""
    return run_as_volume(fov_foldover_volume, x, theta, seed, axes=axes)
