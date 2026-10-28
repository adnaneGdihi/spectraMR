"""Native 3-D k-space phase and window operators: Gibbs ringing, eddy current, I/Q ghost.

All take ``(B, C, D, H, W)`` for ``VOLUME_3D`` (one 3-D k-space, lines indexed by (kz, ky),
readout kx at dim -1) and reuse the 2-D kernels' theta -> parameter mappings.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_randn

__all__ = ["conjugate_mirror", "eddy_volume", "gibbs_volume", "iq_ghost_volume"]

_VOL_DIMS = (-3, -2, -1)


def _check_5d(x: Tensor, name: str) -> None:
    if x.dim() != 5:
        raise ValueError(f"{name} expects (B, C, D, H, W), got {tuple(x.shape)}")


def gibbs_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, **kw: float
) -> Tensor:
    """D9 in 3-D: keep k-space where ``r_n <= truncation``, ``r_n = |k_n| / sqrt(3)``.

    ``k_n = 2 * centred_freqs(n)`` per axis, so ``r_n`` spans [0, 1] with the grid corner at 1.
    Normalising each axis by its own extent keeps the same fraction of every axis, so the
    window is an ellipsoid in physical k when voxel sizes differ.
    """
    _check_5d(x, "gibbs")
    trunc = float(kw.get("truncation_fraction", 1.0 - float(theta) * 0.915))
    if trunc >= 1.0:
        return x
    x = _ensure_complex(x)
    d, h, w = x.shape[-3:]
    kz = (2.0 * centred_freqs(d, x.device)).view(-1, 1, 1)
    ky = (2.0 * centred_freqs(h, x.device)).view(1, -1, 1)
    kx = (2.0 * centred_freqs(w, x.device)).view(1, 1, -1)
    radius = torch.sqrt(kz**2 + ky**2 + kx**2) / math.sqrt(3.0)
    window = (radius <= trunc).to(x.real.dtype)
    return ifftnc(fftnc(x, dims=_VOL_DIMS) * window, dims=_VOL_DIMS)


def _eddy_coefficients(
    batch: int, strength: float, device: torch.device, seed: int
) -> tuple[Tensor, ...]:
    """Per-sample ``(alpha_y, beta_y, alpha_z, beta_z)``: alpha ~ N(0, s), beta ~ N(0, s / 2)."""
    g = _generator(device, seed)
    z = per_sample_randn(batch, 4, generator=g, device=device)
    scale = z.new_tensor([1.0, 0.5, 1.0, 0.5]) * strength
    a_y, b_y, a_z, b_z = (z * scale).unbind(dim=1)
    return a_y, b_y, a_z, b_z


def eddy_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, **kw: float
) -> Tensor:
    """D14 in 3-D: per-(kz, ky)-line phase ``pi * sum_axis(alpha * u + beta * u^2)``.

    ``u = 2 * centred_freqs`` on each phase-encode axis (0 on the DC line), so a linear term
    is an exact circular shift of alpha voxels along that axis; the readout is untouched.
    """
    _check_5d(x, "eddy")
    strength = float(kw.get("strength", float(theta) * 5.0))
    if strength == 0.0:
        return x
    x = _ensure_complex(x)
    d, h = x.shape[-3], x.shape[-2]
    a_y, b_y, a_z, b_z = (
        c.view(-1, 1, 1, 1, 1) for c in _eddy_coefficients(x.shape[0], strength, x.device, seed)
    )
    uz = (2.0 * centred_freqs(d, x.device)).view(1, 1, -1, 1, 1)
    uy = (2.0 * centred_freqs(h, x.device)).view(1, 1, 1, -1, 1)
    phase = a_y * uy + b_y * uy**2 + a_z * uz + b_z * uz**2
    return ifftnc(fftnc(x, dims=_VOL_DIMS) * torch.exp(1j * math.pi * phase), dims=_VOL_DIMS)


def conjugate_mirror(x: Tensor) -> Tensor:
    """``conj(x)(-r)`` about the centre voxel ``N // 2`` of dims (-3, -2, -1)."""
    m = x.conj().flip(dims=_VOL_DIMS)
    shifts = tuple(1 - n % 2 for n in x.shape[-3:])
    return torch.roll(m, shifts=shifts, dims=_VOL_DIMS)


def iq_ghost_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_dc_offset: float = 0.20,
    max_iq_imbalance: float = 0.35,
) -> Tensor:
    """D25 in 3-D: conjugate-mirror ghost through the volume centre plus a centre DC spike."""
    _check_5d(x, "iq_ghost")
    if float(theta) == 0.0:
        return x
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    d, h, w = x.shape[-3:]
    rand2 = torch.rand(2, generator=g, device=x.device)
    dc_amp = max_dc_offset * float(theta) * (1.0 + 0.1 * (rand2[0] - 0.5))
    iq_amp = max_iq_imbalance * float(theta) * (1.0 + 0.1 * (rand2[1] - 0.5))
    out = x + iq_amp * conjugate_mirror(x)
    peak = out.abs().amax(dim=(-4, -3, -2, -1))
    out[..., d // 2, h // 2, w // 2] = out[..., d // 2, h // 2, w // 2] + dc_amp * peak.view(-1, 1)
    return out
