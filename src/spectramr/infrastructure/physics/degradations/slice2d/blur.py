"""2-D blur kernels: readout relaxation (D12), slice-PVE proxy (D13), resolution + noise (D28)."""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _sample_rms,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c
from spectramr.infrastructure.physics.per_sample_rng import per_sample_normal_pair


def degrade_t2star_blur(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_readout_over_t2: float = 4.0,
    echo: str = "spin",
) -> torch.Tensor:
    r"""D12 — Relaxation during the readout: a Lorentzian PSF along readout only.

    Readout sample :math:`k_x` is acquired at :math:`t = \mathrm{TE} + f\,T_{\mathrm{read}}`,
    :math:`f \in [-\tfrac12, \tfrac12)` cycles/px, so k-space is weighted by the decay
    over that time, normalised to 1 at the echo (:math:`f = 0`):

    * ``echo="spin"`` (spin echo / TSE): the reversible dephasing refocuses at the echo,
      :math:`w(f) = e^{-r|f|}` with :math:`r = T_{\mathrm{read}}/T_2'`.
    * ``echo="gradient"``: nothing refocuses, :math:`w(f) = e^{-r f}` with
      :math:`r = T_{\mathrm{read}}/T_2^*`; the PSF is asymmetric and complex.

    The spin-echo PSF is Lorentzian with FWHM :math:`r/\pi` px. ``r`` ramps from 0 to
    ``max_readout_over_t2``. As a k-space multiply the blur is circular, as it is in a
    DFT reconstruction, and phase encode is untouched. True T2* varies per voxel; this
    uses one value for the image.
    """
    if echo not in ("spin", "gradient"):
        raise ValueError(f"echo must be 'spin' or 'gradient'; got {echo!r}.")
    r = max(0.0, float(theta) * max_readout_over_t2)
    if r <= 0.0:
        return x
    k = fft2c(_ensure_complex(x))
    f = centred_freqs(k.shape[-1], k.device).view(1, 1, 1, -1)
    weight = torch.exp(-r * (f.abs() if echo == "spin" else f))
    return ifft2c(k * weight)


def degrade_slice_thickness_pve(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_kernel_px: int = 20,
) -> torch.Tensor:
    """D13 — In-plane proxy for the through-plane partial-volume effect.

    A through-plane PVE mixes tissue from neighbouring depths and cannot be expressed
    on a single 2-D slice. This is a stand-in, not that physics: a one-axis Gaussian
    blur along readout (std ``theta * max_kernel_px / 4`` px), applied as a Gaussian
    k-space window so the blur is circular and phase encode is untouched.
    """
    sigma_px = float(theta) * (max_kernel_px / 4.0)
    if sigma_px < 1e-3:
        return x
    k = fft2c(_ensure_complex(x))
    f = centred_freqs(k.shape[-1], k.device).view(1, 1, 1, -1)
    # A Gaussian PSF of std sigma px is the window exp(-2 pi^2 sigma^2 f^2).
    return ifft2c(k * torch.exp(-2.0 * math.pi**2 * sigma_px**2 * f**2))


def degrade_resolution_snr(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    fwhm_max_px: float = 2.5,
    snr_floor_db: float = 20.0,
) -> torch.Tensor:
    r"""D28 — coupled effective-resolution loss + noise, no resampling.

    A single ``theta`` ramps BOTH the effective spatial resolution and the
    noise -- the two quantities the SNR/resolution literature holds inseparable
    (``SNR ∝ voxel_volume · sqrt(T_acq)``; Edelstein 1986; Larson). The
    acquisition matrix is unchanged: resolution is lost by a **smooth k-space
    apodization** that broadens the point-spread function (effective resolution
    = PSF FWHM), never by resampling.

    - *Resolution.* Multiply centred k-space by a Gaussian window whose
      image-domain PSF has ``FWHM = theta · fwhm_max_px``. A Gaussian PSF of
      std ``sigma_x`` px is the Fourier pair of ``exp(-2 pi^2 sigma_x^2 (f)^2)``
      in normalised frequency ``f = k / N``, so the window is
      ``W(f) = exp(-2 pi^2 sigma_x^2 f^2)`` with ``sigma_x = FWHM / 2.3548``.
    - *Coupled noise.* Complex Gaussian k-space noise, calibrated per-image to
      the clean k-space RMS, with sigma linear in ``theta`` from 0 (clean) to
      the ``snr_floor_db`` level at ``theta=1``.

    This is distinct from ``t2star_blur`` (a fixed-noise Gaussian blur) and from
    ``complex_gaussian`` (fixed-resolution noise): here the two co-vary on one
    axis, so a metric that trades sharpness for noise cannot hide behind an axis
    that only moves one of them. ``theta=0`` is an exact identity (no
    apodization, no noise).
    """
    x = _ensure_complex(x)
    device = x.device
    t = float(theta)
    k = fft2c(x)
    rms = _sample_rms(k)

    # Effective-resolution loss: smooth Gaussian k-space apodization (no resample).
    fwhm = t * fwhm_max_px
    if fwhm > 1e-6:
        sigma_x = fwhm / 2.35482  # FWHM -> Gaussian std (px)
        h, w = k.shape[-2], k.shape[-1]
        fy = (torch.arange(h, device=device, dtype=torch.float32) - h // 2) / h
        fx = (torch.arange(w, device=device, dtype=torch.float32) - w // 2) / w
        f2 = fy[:, None] ** 2 + fx[None, :] ** 2  # normalised radial frequency^2
        window = torch.exp(-2.0 * (math.pi**2) * (sigma_x**2) * f2)
        k = k * window

    # Coupled noise: sigma linear in theta, calibrated to the clean signal RMS.
    if t > 0.0:
        g = _generator(device, seed)
        sigma = t * rms / (10 ** (snr_floor_db / 20.0)) / (2.0**0.5)
        n_re, n_im = per_sample_normal_pair(*k.shape, generator=g, device=device)
        k = k + torch.complex(n_re, n_im) * sigma
    return ifft2c(k)


# ──────────────────────────────────────────────────────────────────────
# D27 (slice_profile) — REMOVED. It was a duplicate of D12 (t2star_blur).
#
# Issue #245. ``degrade_slice_profile`` was an isotropic depthwise Gaussian blur
# with ``max_sigma_px = 3.5``; ``degrade_t2star_blur`` (D12) is an isotropic
# depthwise Gaussian blur with ``max_sigma_px = 4.0``. Same operator, same
# family, differing only in a scalar: the cosine similarity of their residuals at
# theta = 1 was **0.998**. They were counted as two independent "orthogonal"
# physics axes, which double-counted the low-pass family in every per-family
# aggregation, inflated D in the Borda sum, and inflated the family count in the
# CDRS variance penalty.
#
# Its docstring claimed to be a *through-plane* effect ("the slice-select axis
# getting a non-rectangular profile"). A through-plane effect cannot be expressed
# on the single 2-D magnitude slices this sweep runs on — there is no z-neighbour
# to partial-volume with (pitfall #19: the hypothesis is untestable on this
# data). The honest move is to delete the axis, not to dress the same blur up
# twice. The anisotropic through-plane *proxy* survives as D13 (``slice_pve``);
# the isotropic in-plane PSF survives as D12 (``t2star_blur``).
#
# ``apply_degradation("slice_profile", ...)`` now raises with the list of valid
# names (pitfall #9: no silent fallback). Consumers must drop the axis — see the
# migration note in docs/degradation_simulator_correctness.rst.
# ──────────────────────────────────────────────────────────────────────
