"""Reconstruction filters on acquired k-space: apodisation and zero-filled interpolation.

``recon_apodization`` multiplies k-space by a window mixed with the identity, ``w_theta = 1 - theta (1 - w)``
(a low-pass: Hamming, Hann or Fermi). ``zero_fill_interp`` acquires the central ``1 - f`` of every encoded
axis and zero-fills back to the full matrix, i.e. sinc interpolation with truncation ringing
(Bernstein, King & Zhou, Handbook of MRI Pulse Sequences, 2004, secs. 13.2 and 13.4). The windows are
separable over the encoded axes (ky, kx in 2-D and multislice; kz, ky, kx in 3-D), so a multislice stack is
exactly its slices run one by one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import check_input
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc

__all__ = [
    "WINDOWS",
    "apodization_2d",
    "apodization_volume",
    "apodization_window",
    "truncation_mask",
    "zero_fill_2d",
    "zero_fill_volume",
]

_FERMI_RADIUS, _FERMI_WIDTH = 0.75, 0.1

#: 1-D windows of the normalised k-space radius ``rho = |k| / k_max`` in [0, 1].
WINDOWS: Mapping[str, Callable[[Tensor], Tensor]] = {
    # Hamming w = 0.54 + 0.46 cos(pi rho); Hann w = 0.5 (1 + cos(pi rho)); Fermi w = 1 / (1 + e^((rho - r0) / T)).
    "hamming": lambda rho: 0.54 + 0.46 * torch.cos(math.pi * rho),
    "hann": lambda rho: 0.5 * (1.0 + torch.cos(math.pi * rho)),
    "fermi": lambda rho: torch.sigmoid(-(rho - _FERMI_RADIUS) / _FERMI_WIDTH),
}


def _axis_view(vec: Tensor, axis: int, ndim: int) -> Tensor:
    shape = [1] * ndim
    shape[axis] = -1
    return vec.view(shape)


def apodization_window(
    shape: tuple[int, ...],
    dims: tuple[int, ...],
    window: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """The separable window ``prod_a w(|2 f_a|)`` over ``dims`` of a tensor of ``shape``, broadcastable to it."""
    try:
        w1 = WINDOWS[window]
    except KeyError:
        raise ValueError(f"unknown window {window!r}; choose from {sorted(WINDOWS)}") from None
    out = torch.ones((), device=device, dtype=dtype)
    for d in dims:
        rho = 2.0 * centred_freqs(shape[d], device, dtype).abs()
        out = out * _axis_view(w1(rho), d % len(shape), len(shape))
    return out


def _apodize(x: Tensor, theta: float, window: str, dims: tuple[int, ...]) -> Tensor:
    # The window is built once: an unknown name raises here, at theta = 0 too.
    w = apodization_window(tuple(x.shape), dims, window, x.device, x.real.dtype)
    if float(theta) <= 0.0:
        return x
    return ifftnc(fftnc(x, dims) * (1.0 - float(theta) * (1.0 - w)), dims)


def apodization_2d(x: Tensor, theta: float, seed: int = 0, window: str = "hamming") -> Tensor:
    """recon_apodization on a ``(B, C, H, W)`` slice: in-plane (ky, kx) window."""
    x = _ensure_complex(x)
    return _apodize(x, theta, window, (-2, -1))


def apodization_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, window: str = "hamming"
) -> Tensor:
    """recon_apodization on a 3-D volume: the product of three 1-D windows over (kz, ky, kx)."""
    check_input(x, geometry, "recon_apodization")
    x = _ensure_complex(x)
    return _apodize(x, theta, window, (-3, -2, -1))


def truncation_mask(
    shape: tuple[int, ...], dims: tuple[int, ...], keep_fraction: float, device: torch.device
) -> Tensor:
    """Boolean mask of the central ``round(n * keep_fraction)`` samples along each of ``dims`` (DC at ``n // 2``)."""
    mask = torch.ones((), dtype=torch.bool, device=device)
    for d in dims:
        n = shape[d]
        keep = max(1, round(n * keep_fraction))
        idx = torch.arange(n, device=device) - n // 2
        mask = mask & _axis_view(
            (idx >= -(keep // 2)) & (idx < keep - keep // 2), d % len(shape), len(shape)
        )
    return mask


def _zero_fill(x: Tensor, theta: float, max_fraction: float, dims: tuple[int, ...]) -> Tensor:
    if not 0.0 <= max_fraction < 1.0:
        raise ValueError(f"zero_fill_interp: max_fraction must be in [0, 1); got {max_fraction}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    mask = truncation_mask(tuple(x.shape), dims, 1.0 - float(theta) * max_fraction, x.device)
    return ifftnc(fftnc(x, dims) * mask, dims)


def zero_fill_2d(x: Tensor, theta: float, seed: int = 0, max_fraction: float = 0.5) -> Tensor:
    """zero_fill_interp on a ``(B, C, H, W)`` slice: acquire the central ``1 - f`` of (ky, kx), ``f = theta max_fraction``."""
    return _zero_fill(x, theta, max_fraction, (-2, -1))


def zero_fill_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, max_fraction: float = 0.5
) -> Tensor:
    """zero_fill_interp on a 3-D volume: central box of (kz, ky, kx)."""
    check_input(x, geometry, "zero_fill_interp")
    return _zero_fill(x, theta, max_fraction, (-3, -2, -1))
