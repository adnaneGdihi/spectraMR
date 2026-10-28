"""Slice-profile kernels and the 1-D convolution along the slice axis of a volume."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["SLICE_PROFILE_KINDS", "convolve_along_slices", "slice_profile_kernel"]

# Hand-tuned SLR profile — 33-tap symmetric kernel from a canonical
# 90°-pulse SLR design (BW * T = 4). Cached as a constant so we don't
# pull in scipy.signal at import time.
_SLR_KERNEL_33 = (
    -0.0014,
    -0.0028,
    -0.0027,
    0.0011,
    0.0083,
    0.0151,
    0.0150,
    0.0035,
    -0.0177,
    -0.0392,
    -0.0469,
    -0.0306,
    0.0085,
    0.0623,
    0.1209,
    0.1700,
    0.1908,
    0.1700,
    0.1209,
    0.0623,
    0.0085,
    -0.0306,
    -0.0469,
    -0.0392,
    -0.0177,
    0.0035,
    0.0150,
    0.0151,
    0.0083,
    0.0011,
    -0.0027,
    -0.0028,
    -0.0014,
)


def _sinc_gauss_kernel(fwhm_voxels: float, length: int) -> np.ndarray:
    """Apodised sinc profile, FWHM specified in voxels along the z axis."""
    half = (length - 1) / 2.0
    x = np.arange(length, dtype=np.float64) - half
    sigma = fwhm_voxels / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sinc = np.sinc(x / max(fwhm_voxels, 1e-3))
    gauss = np.exp(-(x**2) / (2.0 * sigma**2))
    k = sinc * gauss
    k = k - k.mean()
    k = k + abs(k.min())  # ensure non-negative
    return k / k.sum()


def _gauss_kernel(fwhm_voxels: float, length: int) -> np.ndarray:
    half = (length - 1) / 2.0
    x = np.arange(length, dtype=np.float64) - half
    sigma = fwhm_voxels / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def _slr_kernel() -> np.ndarray:
    return np.array(_SLR_KERNEL_33, dtype=np.float64)


_KERNELS: dict[str, Callable[[float, int], np.ndarray]] = {
    "sinc_gauss": _sinc_gauss_kernel,
    "gauss": _gauss_kernel,
    "slr": lambda _fwhm, _length: _slr_kernel(),
}
SLICE_PROFILE_KINDS = frozenset(_KERNELS)


def slice_profile_kernel(
    kind: Literal["sinc_gauss", "gauss", "slr"], fwhm_voxels: float, length: int
) -> np.ndarray:
    """Normalised 1-D slice profile; ``"slr"`` is a fixed 33-tap table and ignores the sizes."""
    if kind not in _KERNELS:
        raise ValueError(f"kind must be one of {sorted(_KERNELS)}; got {kind!r}")
    return _KERNELS[kind](fwhm_voxels, length)


def convolve_along_slices(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve ``x`` ``[B, C, D, H, W]`` with ``kernel`` ``[K]`` along D, replicate-padded.

    The output keeps the input shape and dtype; a complex input is convolved part by part.
    Even-length kernels put the extra tap of padding after the volume.
    """
    if x.ndim != 5:
        raise ValueError(f"x must be (B, C, D, H, W); got {tuple(x.shape)}")
    if kernel.ndim != 1 or kernel.numel() < 1:
        raise ValueError(f"kernel must be a non-empty 1-D tensor; got {tuple(kernel.shape)}")
    if x.is_complex():
        return torch.complex(
            convolve_along_slices(x.real, kernel), convolve_along_slices(x.imag, kernel)
        )
    b, c, d, h, w = x.shape
    k = kernel.numel()
    left = (k - 1) // 2
    weight = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, k, 1, 1)
    flat = x.reshape(b * c, 1, d, h, w)
    padded = F.pad(flat, (0, 0, 0, 0, left, k - 1 - left), mode="replicate")
    return F.conv3d(padded, weight).reshape(b, c, d, h, w)
