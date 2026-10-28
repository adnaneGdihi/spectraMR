"""2-D noise kernels: complex Gaussian k-space noise (D1), Rician (D2), RF spikes (D15)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _isolated_rng,
    _sample_rms,
    _signal_level,
)
from spectramr.infrastructure.physics.digital_twin_simulator import simulate_spike_noise
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.per_sample_rng import per_sample_normal_pair


def degrade_complex_gaussian_noise(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    snr_max_db: float = 40.0,
    snr_min_db: float = 2.0,
) -> torch.Tensor:
    r"""D1 — complex Gaussian k-space noise → Rician magnitude noise.

    ``theta=0`` → ``snr_max_db`` (clean), ``theta=1`` → ``snr_min_db``.

    .. math::
        y = M F x + \eta,\quad \eta \sim \mathcal{CN}(0, \sigma^2 I)
    """
    x = _ensure_complex(x)
    device = x.device
    g = _generator(device, seed)
    snr_db = snr_max_db + (snr_min_db - snr_max_db) * float(theta)

    k = fft2c(x)
    rms = _sample_rms(k)
    # Complex noise eta = n_re + i*n_im with each channel ~ N(0, sigma^2) has
    # power E|eta|^2 = 2*sigma^2. To realise the *declared* SNR
    # (signal_power / noise_power = 10^(snr_db/10)) the per-channel std must carry
    # half the noise power, hence the 1/sqrt(2). Without it the realised SNR is
    # 3.01 dB hotter than declared (the severity axis the rankers calibrate to).
    sigma = rms / (10 ** (snr_db / 20.0)) / (2.0**0.5)

    n_re, n_im = per_sample_normal_pair(*k.shape, generator=g, device=device)
    return ifft2c(k + torch.complex(n_re, n_im) * sigma)


def degrade_rician_noise(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    sigma_max: float = 0.3,
) -> torch.Tensor:
    """D2 — image-space Rician noise on the magnitude image.

    Applied directly to ``|x|`` so the bias term ``sigma * sqrt(pi/2)``
    at zero signal is preserved.

    ``sigma_max`` is a **fraction of the image's own p99 signal**, matching the unit
    this axis declares in its registry severity. It used to be applied as a literal intensity,
    which made D2 the only noise axis that was not scale-invariant (D1 derives sigma
    from the k-space RMS at a declared SNR). ``sim2rank.py`` sweeps the RAW
    coil-combined magnitude (p99 ~ 1e-4 on fastMRI brain) and divides by p99 only
    *after* the sweep, so a literal ``sigma = theta * 0.3`` landed ~4 orders of
    magnitude above the signal: psnr pinned to its -30 dB floor, ssim to 0.0 and
    nema_snr to a constant at EVERY severity level, and the snapshot triptychs washed
    out to a uniform noise field.
    """
    if float(theta) <= 0.0:
        return x  # the clean anchor is the input itself, phase included
    mag = x.abs() if x.is_complex() else x
    g = _generator(mag.device, seed)
    sigma = float(theta) * sigma_max * _signal_level(mag)
    eta1, eta2 = per_sample_normal_pair(*mag.shape, generator=g, device=mag.device, dtype=mag.dtype)
    eta1, eta2 = eta1 * sigma, eta2 * sigma
    return torch.sqrt((mag + eta1) ** 2 + eta2**2)


def _delegate_spike(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        prob = float(kw.get("probability", float(theta) * 0.05))
        out = ifft2c(simulate_spike_noise(fft2c(x), probability=prob))
    return out
