"""Receive-chain noise axes: inter-coil correlated noise, EMI bursts, anti-alias roll-off.

Noise is referenced to the whole subject's k-space RMS (coils and every slice / kz plane), the
convention of ``complex_gaussian`` in ``volume/noise.py``: the severity is ``sigma / RMS``, the
noise RMS per complex sample over the signal RMS. ``m4raw_noise.M4RawNoiseSampler`` is not
reused: it draws one whole-tensor sample per call rather than one per sample (a sample's noise
would depend on the batch), takes a (..., C, H, W) layout and a mandatory support mask, and
defaults to the measured M4Raw matrix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.catalogue.receive_common import (
    acquisition_ranks,
    check_n_coils,
    complex_white,
    control_generator,
    positive,
    receive_op,
    two_d,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.degradations.volume.noise import _volume_rms
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "SYNTHETIC_COIL_RHO",
    "antialias_rolloff",
    "antialias_rolloff_2d",
    "butterworth_response",
    "coil_noise_factor",
    "correlated_coil_noise",
    "correlated_coil_noise_2d",
    "emi_burst",
    "emi_burst_2d",
]

#: Adjacent-coil noise correlation of the synthetic covariance Sigma_ij = rho^|i-j|.
SYNTHETIC_COIL_RHO = 0.3
_BUTTERWORTH_ORDER = 4
_CUTOFF_MIN = 0.6


def coil_noise_factor(coil_covariance: Sequence[Sequence[float]] | Tensor | None, n: int) -> Tensor:
    """Cholesky factor ``L`` (``L L^H = Sigma_n``) of the unit-mean-power coil covariance.

    ``Sigma_n = Sigma / mean(diag Sigma)`` so the total noise power per sample is the declared
    sigma^2 whatever units the covariance is in; ``None`` means ``Sigma_ij = rho^|i-j|``.
    """
    if coil_covariance is None:
        idx = torch.arange(n, dtype=torch.float64)
        cov = SYNTHETIC_COIL_RHO ** (idx[:, None] - idx[None, :]).abs()
        cov = cov.to(torch.complex128)
    else:
        cov = torch.as_tensor(coil_covariance).to(torch.complex128)
    if cov.shape != (n, n):
        raise ValueError(f"coil_covariance must be {n} x {n} for {n} coils, got {tuple(cov.shape)}")
    if not torch.allclose(cov, cov.mH):
        raise ValueError("coil_covariance must be Hermitian")
    cov = cov / cov.diagonal().real.mean()
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError as err:
        raise ValueError(f"coil_covariance must be positive definite: {err}") from err


@receive_op
def correlated_coil_noise(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
    *,
    coil_covariance: Sequence[Sequence[float]] | Tensor | None = None,
    n_coils: int | None = None,
    sigma_max: float = 0.3,
) -> Tensor:
    """Complex Gaussian k-space noise with inter-coil covariance: ``n = sigma L w``, ``L L^H = Sigma``.

    ``w ~ CN(0, I)`` so ``E[n n^H] = sigma^2 Sigma_n`` (Roemer et al., MRM 16:192, 1990, noise
    covariance Psi; colouring by the Cholesky factor, Golub & Van Loan, Matrix Computations).
    """
    check_n_coils(x, n_coils)
    sigma_max = positive("sigma_max", sigma_max)
    chol = coil_noise_factor(coil_covariance, x.shape[1])
    if theta <= 0.0:
        return x
    x = _ensure_complex(x)
    k = fftnc(x, dims)
    sigma = theta * sigma_max * _volume_rms(k)
    w = complex_white(tuple(k.shape), _generator(x.device, seed), k)
    z = torch.einsum("ij,bjdhw->bidhw", chol.to(device=k.device, dtype=k.dtype), w)
    return ifftnc(k + sigma * z, dims)


@receive_op
def emi_burst(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
    *,
    sigma_max: float = 0.5,
    frac_max: float = 0.2,
) -> Tensor:
    """Broadband interference over a contiguous run of acquired lines, white across kx.

    A transient RF interferer corrupts only the readouts acquired while it is on (Bernstein et
    al., Handbook of MRI Pulse Sequences, 2004, ch. 13); its spectrum is flat over the readout,
    so each corrupted line gains i.i.d. complex noise of RMS ``theta sigma_max`` x subject RMS,
    independent per coil. The run is ``ceil(theta frac_max N)`` lines of the ``N = D H`` lines in
    acquisition order, starting at a seeded line that does not depend on theta (runs are nested).
    """
    sigma_max, frac_max = positive("sigma_max", sigma_max), positive("frac_max", frac_max)
    if frac_max > 1.0:
        raise ValueError(f"frac_max must be <= 1, got {frac_max}")
    if theta <= 0.0:
        return x
    x = _ensure_complex(x)
    b, _, d, h, _ = x.shape
    n_lines = d * h
    n_max = max(1, math.ceil(frac_max * n_lines))
    n_burst = min(n_max, max(1, math.ceil(theta * frac_max * n_lines - 1e-9)))
    u = per_sample_rand(b, generator=control_generator(x.device, seed), device=x.device)
    start = (u * (n_lines - n_max + 1)).floor().long()
    rank, _ = acquisition_ranks(geometry, d, h, x.device)
    order = rank.flatten().argsort()  # line index at each acquisition rank
    lines = order[start.view(-1, 1) + torch.arange(n_burst, device=x.device)]  # [B, n_burst]
    k = fftnc(x, dims)
    c, w = k.shape[1], k.shape[-1]
    sigma = theta * sigma_max * _volume_rms(k)
    noise = complex_white((b, c, n_burst, w), _generator(x.device, seed), k) * sigma.view(
        b, 1, 1, 1
    )
    flat = k.reshape(b, c, n_lines, w)
    index = lines.view(b, 1, n_burst, 1).expand(b, c, n_burst, w)
    return ifftnc(flat.scatter_add(2, index, noise).reshape(k.shape), dims)


def butterworth_response(nu: Tensor, cutoff: float, order: int = _BUTTERWORTH_ORDER) -> Tensor:
    """Butterworth magnitude ``1 / sqrt(1 + (|nu| / cutoff)^(2 order))`` (Butterworth, Wireless
    Engineer 7:536, 1930); ``nu`` and ``cutoff`` are fractions of the Nyquist band edge."""
    return torch.rsqrt(1.0 + (nu.abs() / cutoff) ** (2 * order))


@receive_op
def antialias_rolloff(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    dims: tuple[int, ...],
    *,
    noise_sigma: float = 0.0,
) -> Tensor:
    """Receiver anti-alias filter: Butterworth (n = 4) magnitude over the readout band.

    The filter acts on the time-domain readout, and readout time maps to position through the
    gradient, so its frequency response H(nu) weights the IMAGE along the readout axis (nu = 1 is
    the band edge, the FOV border; Bernstein et al. 2004, ch. 13); in k-space that is a
    convolution, not a taper. The cutoff is ``x_c = 0.6 / theta`` in units of the band edge:
    infinite at theta -> 0+ (the response tends continuously to the identity) and 0.6 at
    theta = 1. Noise added before the filter (``noise_sigma`` x subject k-space RMS per complex
    sample, independent of theta) is coloured by it. Only the magnitude response is modelled.
    """
    if not (math.isfinite(noise_sigma) and noise_sigma >= 0.0):
        raise ValueError(f"noise_sigma must be a finite number >= 0, got {noise_sigma!r}")
    if theta <= 0.0 and noise_sigma == 0.0:
        return x
    x = _ensure_complex(x)
    if noise_sigma > 0.0:
        k = fftnc(x, dims)
        noise = complex_white(tuple(k.shape), _generator(x.device, seed), k)
        x = ifftnc(k + noise_sigma * _volume_rms(k) * noise, dims)
    if theta <= 0.0:
        return x
    nu = 2.0 * centred_freqs(x.shape[-1], x.device)
    gain = butterworth_response(nu, _CUTOFF_MIN / theta)
    return x * gain.to(x.real.dtype)


correlated_coil_noise_2d = two_d(correlated_coil_noise)
emi_burst_2d = two_d(emi_burst)
antialias_rolloff_2d = two_d(antialias_rolloff)
