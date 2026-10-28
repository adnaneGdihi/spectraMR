"""Native 3-D sampling: Cartesian (kz, ky) line masks, 3-D Bernoulli VD and partial Fourier.

Every operator takes ``(B, C, D, H, W)`` in ``VOLUME_3D`` and applies its mask in the 3-D
k-space ``fftnc(x, dims=(-3, -2, -1))``. Random masks are drawn from one seeded CPU
generator, one realisation per sample in batch order, shared by every coil; the readout kx
is fully sampled except where the mask is a 3-D point mask (``vd_undersamp``) or the
partial echo (``partial_fourier`` on ``kx``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _calibrate_bernoulli_scale,
    _ensure_complex,
    _generator,
)
from spectramr.infrastructure.physics.degradations.blocks.kspace_masks_3d import (
    cartesian_mask_3d,
    partial_fourier_mask_3d,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "cartesian_undersamp_volume",
    "partial_fourier_volume",
    "vd_cartesian_volume",
    "vd_undersamp_volume",
]

_DIMS = (-3, -2, -1)
_CPU = torch.device("cpu")
_DC_SIDE = 5  # side of the always-acquired DC block on each axis, as in the 2-D kernel
_PF_AXES = ("ky", "kz", "ky+kz", "kx")


def _check(x: Tensor, geometry: AcquisitionGeometry, name: str) -> None:
    if x.dim() != 5:
        raise ValueError(f"{name} expects (B, C, D, H, W), got {tuple(x.shape)}")
    if geometry.mode is not AcquisitionMode.VOLUME_3D:
        raise ValueError(f"{name} is the 3-D realisation; got mode {geometry.mode!r}")


def _acceleration(theta: float, r_max: float) -> float:
    if not r_max >= 1.0:
        raise ValueError(f"R_max must be >= 1, got {r_max}")
    return 1.0 + (r_max - 1.0) * float(theta)


def _apply_mask(x: Tensor, mask: Tensor) -> Tensor:
    """``mask`` is ``[B, D, H, W]`` or ``[B, D, H, 1]``; broadcast over coils."""
    return ifftnc(fftnc(x, _DIMS) * mask.to(x.device).unsqueeze(1), _DIMS)


def _cartesian(
    x: Tensor, theta: float, seed: int, r_max: float, center_fraction: float, **mask_kw
) -> Tensor:
    if not 0.0 < center_fraction <= 1.0:
        raise ValueError(f"center_fraction must be in (0, 1], got {center_fraction}")
    R = _acceleration(theta, r_max)
    if R <= 1.0:
        return x
    x = _ensure_complex(x)
    b, _, n_kz, n_ky, _ = x.shape
    # The 2-D kernels' ACS fraction applied to each encoded axis, at least one line each.
    acs = (max(1, int(n_kz * center_fraction)), max(1, int(n_ky * center_fraction)))
    g = _generator(_CPU, seed)
    masks = [cartesian_mask_3d(n_kz, n_ky, R, acs=acs, generator=g, **mask_kw) for _ in range(b)]
    return _apply_mask(x, torch.stack(masks).unsqueeze(-1).to(x.real.dtype))


def cartesian_undersamp_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    R_max: float = 8.0,  # noqa: N803
    center_fraction: float = 0.08,
) -> Tensor:
    """D3 in 3-D: uniform random (kz, ky) lines, exactly ``round(n_kz*n_ky/R)`` acquired."""
    _check(x, geometry, "cartesian_undersamp_volume")
    return _cartesian(x, theta, seed, R_max, center_fraction, density="uniform")


def vd_cartesian_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    R_max: float = 8.0,  # noqa: N803
    alpha: float = 3.0,
    center_fraction: float = 0.08,
) -> Tensor:
    """D4b in 3-D: (kz, ky) lines weighted ``(1+|k|)^-alpha``, exact line count."""
    _check(x, geometry, "vd_cartesian_volume")
    return _cartesian(x, theta, seed, R_max, center_fraction, density="variable", alpha=alpha)


def _dc_block(shape: tuple[int, int, int]) -> Tensor:
    block = torch.zeros(shape, dtype=torch.bool)
    idx = []
    for n in shape:
        m = min(_DC_SIDE, n)
        idx.append(slice(n // 2 - m // 2, n // 2 - m // 2 + m))
    block[tuple(idx)] = True
    return block


def vd_undersamp_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    R_max: float = 8.0,  # noqa: N803
    alpha: float = 3.0,
) -> Tensor:
    """D4 in 3-D: Bernoulli point mask over (kz, ky, kx), ``p ∝ (1+|k|)^-alpha``.

    The central ``5 x 5 x 5`` DC block (``min(5, n)`` per axis) is always kept and counted
    once; the outer probabilities are calibrated so the expected kept count is ``N / R``.
    """
    _check(x, geometry, "vd_undersamp_volume")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    R = _acceleration(theta, R_max)
    if R <= 1.0:
        return x
    x = _ensure_complex(x)
    b, _, d, h, w = x.shape
    kz, ky, kx = (2.0 * centred_freqs(n) for n in (d, h, w))
    radius = torch.sqrt(kz.view(-1, 1, 1) ** 2 + ky.view(1, -1, 1) ** 2 + kx.view(1, 1, -1) ** 2)
    p = (1.0 + radius) ** (-alpha)
    dc = _dc_block((d, h, w))
    n_total, n_dc = d * h * w, int(dc.sum())
    p_outer = p[~dc]
    target = max(0.0, (n_total / R - n_dc) / max(1, n_total - n_dc))
    p_keep = torch.ones_like(p)
    if p_outer.numel():
        p_keep[~dc] = (_calibrate_bernoulli_scale(p_outer, target) * p_outer).clamp(max=1.0)
    rand = per_sample_rand(b, d, h, w, generator=_generator(_CPU, seed))
    keep = (rand < p_keep) | dc
    return _apply_mask(x, keep.to(x.real.dtype))


def _pf_lines(n_kz: int, n_ky: int, n_kx: int, frac: float, axis: str) -> Tensor:
    return partial_fourier_mask_3d(n_kz, n_ky, frac, axis=axis).unsqueeze(-1)


def _pf_readout(n_kz: int, n_ky: int, n_kx: int, frac: float, axis: str) -> Tensor:
    # Partial echo along dim -1, cut from the high side exactly as the 2-D kernel does.
    mask = torch.ones(n_kz, n_ky, n_kx, dtype=torch.bool)
    cut = round(n_kx * (1.0 - frac))
    if cut > 0:
        mask[..., n_kx - cut :] = False
    return mask


_PF_MASK = {"ky": _pf_lines, "kz": _pf_lines, "ky+kz": _pf_lines, "kx": _pf_readout}


def partial_fourier_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    min_fraction: float = 0.625,
    axis: str = "ky",
) -> Tensor:
    """D24 in 3-D: keep the low-index ``frac`` of ``axis``; ``frac`` ramps 1 -> ``min_fraction``.

    ``ky`` / ``kz`` / ``ky+kz`` truncate the (kz, ky) line grid (``ky+kz`` both axes);
    ``kx`` is the readout partial echo.
    """
    _check(x, geometry, "partial_fourier_volume")
    make = _PF_MASK.get(axis)
    if make is None:
        raise ValueError(f"axis must be one of {_PF_AXES} in 3-D; got {axis!r}")
    if not 0.5 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in (0.5, 1], got {min_fraction}")
    if float(theta) <= 0.0:
        return x
    x = _ensure_complex(x)
    frac = 1.0 - (1.0 - min_fraction) * float(theta)
    mask = make(*x.shape[-3:], frac, axis)
    return _apply_mask(x, mask.unsqueeze(0).to(x.real.dtype))
