"""Compressed-sensing reconstruction artefacts: ISTA with an orthonormal Haar prior.

The scan is variable-density Cartesian undersampled at ``R = 1 + (r_max - 1) theta``; it is reconstructed
by a fixed number of ISTA iterations of ``min ||M F x - y||^2 / 2 + lambda ||Psi x||_1`` (Beck & Teboulle,
SIAM J. Imaging Sci. 2:183, 2009, Eq. 1.4 with unit step, since ``F`` is unitary and ``M`` a projector;
Lustig, Donoho & Pauly, MRM 58:1182, 2007), ``Psi`` the multilevel orthonormal Haar transform over the
encoded axes and the threshold ``lambda = lambda_max min(1, theta / lambda_ramp) p99(|x|)``: fixed for theta above a short ramp (continuous at 0+), so that only R changes along the axis. Haar is orthonormal, so the soft
threshold is the exact proximal map; the coarsest approximation band is not thresholded (no global
intensity bias). The object is edge-padded to a multiple of ``2^levels`` and the
result cropped. The (ky) or (kz, ky) mask is the variable-density one of ``cartesian_mask_3d``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _signal_level,
)
from spectramr.infrastructure.physics.degradations.blocks.kspace_masks_3d import cartesian_mask_3d
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import check_input
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc
from spectramr.infrastructure.physics.wavelet_ops import OrthogonalWavelet2D, complex_soft_threshold

__all__ = ["cs_recon", "cs_recon_2d", "cs_recon_volume", "haar_transform", "ista"]

_HAAR = OrthogonalWavelet2D("haar", 1)


def haar_transform(
    x: Tensor, dims: tuple[int, ...], levels: int, *, inverse: bool = False
) -> Tensor:
    """Multilevel separable orthonormal Haar over ``dims``, in the image-shaped coefficient layout."""
    out = x.clone()
    steps = range(levels - 1, -1, -1) if inverse else range(levels)
    for level in steps:
        idx = [slice(None)] * x.dim()
        for d in dims:
            idx[d] = slice(0, x.shape[d] >> level)
        block = out[tuple(idx)]
        for d in reversed(dims) if inverse else dims:
            block = (_HAAR.synthesise_axis if inverse else _HAAR.analyse_axis)(block, d)
        out[tuple(idx)] = block
    return out


def _norm(dims: tuple[int, ...], ndim: int) -> set[int]:
    return {d % ndim for d in dims}


def ista(
    y: Tensor,
    mask: Tensor,
    lam: Tensor | float,
    iterations: int,
    dims: tuple[int, ...],
    levels: int,
) -> Tensor:
    """``iterations`` ISTA steps from the zero-filled image; ``y`` is k-space, zero off ``mask``."""
    x = ifftnc(y, dims)
    ll = tuple(
        slice(0, x.shape[d] >> levels) if d in _norm(dims, x.dim()) else slice(None)
        for d in range(x.dim())
    )
    for _ in range(iterations):
        x = ifftnc(torch.where(mask, y, fftnc(x, dims)), dims)  # F^H (M y + (1 - M) F x)
        c = haar_transform(x, dims, levels)
        t = complex_soft_threshold(c, lam)
        t[ll] = c[ll]
        x = haar_transform(t, dims, levels, inverse=True)
    return x


def _pad(x: Tensor, dims: tuple[int, ...], multiple: int) -> Tensor:
    pad = [0] * 6  # (W, H, D) right pads; dims are always among the last three axes
    for d in dims:
        pad[2 * (x.dim() - 1 - d % x.dim()) + 1] = -x.shape[d] % multiple
    if not any(pad):
        return x
    return torch.complex(F.pad(x.real, pad, mode="replicate"), F.pad(x.imag, pad, mode="replicate"))


def cs_recon(
    x: Tensor,
    theta: float,
    seed: int,
    dims: tuple[int, ...],
    *,
    r_max: float = 6.0,
    lambda_max: float = 0.01,
    lambda_ramp: float = 0.05,
    iterations: int = 20,
    wavelet_levels: int = 2,
    center_fraction: float = 0.08,
    alpha: float = 3.0,
) -> Tensor:
    """CS-reconstruct ``x`` ``[B, C, D, H, W]`` acquired over ``dims`` (``(-2, -1)`` or ``(-3, -2, -1)``)."""
    if not (r_max >= 1 and lambda_max >= 0 and lambda_ramp > 0 and 0 < center_fraction <= 1):
        raise ValueError(
            "cs_recon_artifacts: need r_max >= 1, lambda_max >= 0, lambda_ramp > 0, 0 < center_fraction <= 1"
        )
    if not (isinstance(iterations, int) and iterations >= 1):
        raise ValueError(
            f"cs_recon_artifacts: iterations must be an integer >= 1; got {iterations!r}"
        )
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    shape, xp = x.shape, _pad(x, dims, 2**wavelet_levels)
    d, h = xp.shape[-3], xp.shape[-2]
    three_d = len(dims) == 3
    acs = (max(1, int(center_fraction * d)) if three_d else 1, max(1, int(center_fraction * h)))
    mask = cartesian_mask_3d(
        d if three_d else 1,
        h,
        1.0 + (r_max - 1.0) * float(theta),
        acs=acs,
        density="variable",
        alpha=alpha,
        generator=_generator(torch.device("cpu"), seed),
    ).to(x.device)
    mask = mask.view(1, 1, d if three_d else 1, h, 1)
    lam = (
        min(1.0, float(theta) / lambda_ramp)
        * lambda_max
        * _signal_level(x.abs()).view(-1, 1, 1, 1, 1)
    )
    out = ista(
        fftnc(xp, dims) * mask, mask, lam.to(xp.real.dtype), iterations, dims, wavelet_levels
    )
    return out[..., : shape[-3], : shape[-2], : shape[-1]]


def cs_recon_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, **kwargs
) -> Tensor:
    """cs_recon_artifacts on a 3-D volume: (kz, ky) mask, 3-D Haar."""
    check_input(x, geometry, "cs_recon_artifacts")
    if geometry.mode is not AcquisitionMode.VOLUME_3D:
        raise ValueError("cs_recon_volume is the 3-D realisation; slices run through cs_recon_2d")
    return cs_recon(x, theta, seed, (-3, -2, -1), **kwargs)


def cs_recon_2d(x: Tensor, theta: float, seed: int = 0, **kwargs) -> Tensor:
    """cs_recon_artifacts on a ``(B, C, H, W)`` slice (channels are separate images sharing one ky mask)."""
    return cs_recon(x.unsqueeze(2), theta, seed, (-2, -1), **kwargs).squeeze(2)
