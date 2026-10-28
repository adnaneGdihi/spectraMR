"""Native volume noise: complex Gaussian, Rician, resolution/SNR, EMI tones and 3-D spikes.

Every operator takes ``(B, C, D, H, W)`` and references its noise level to the whole
subject (coils and every slice / kz plane), so one subject gets one SNR. The encoded axes
follow the mode: ``MULTISLICE_2D`` is a 2-D k-space per slice (dims -2, -1) and
``VOLUME_3D`` one 3-D k-space (dims -3, -2, -1). Each operator draws the same stream as
its 2-D kernel, so a one-slice multislice stack reproduces the 2-D output.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _isolated_rng,
    _signal_level,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.digital_twin_simulator import simulate_spike_noise_volume
from spectramr.infrastructure.physics.emi_degradation import add_readout_tones
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_normal_pair

__all__ = [
    "apodization_window",
    "complex_gaussian_volume",
    "emi_volume",
    "resolution_snr_volume",
    "rician_volume",
    "spike_volume",
]

_ENCODED_DIMS: dict[AcquisitionMode, tuple[int, ...]] = {
    AcquisitionMode.MULTISLICE_2D: (-2, -1),
    AcquisitionMode.VOLUME_3D: (-3, -2, -1),
}


def _encoded_dims(x: Tensor, geometry: AcquisitionGeometry) -> tuple[int, ...]:
    if x.dim() != 5:
        raise ValueError(f"volume noise expects (B, C, D, H, W), got {tuple(x.shape)}")
    dims = _ENCODED_DIMS.get(geometry.mode)
    if dims is None:
        raise ValueError(f"volume noise has no encoding for mode {geometry.mode!r}")
    return dims


def _volume_rms(k: Tensor) -> Tensor:
    """RMS of each sample over coils and the whole k-space, ``[B, 1, 1, 1, 1]``."""
    return k.abs().pow(2).mean(dim=(-4, -3, -2, -1), keepdim=True).sqrt().clamp(min=1e-8)


def _complex_noise(k: Tensor, sigma: Tensor, g: torch.Generator) -> Tensor:
    n_re, n_im = per_sample_normal_pair(*k.shape, generator=g, device=k.device)
    return torch.complex(n_re, n_im) * sigma


def complex_gaussian_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    snr_max_db: float = 40.0,
    snr_min_db: float = 2.0,
) -> Tensor:
    """D1 on a volume: i.i.d. complex k-space noise at the declared whole-volume SNR.

    Added in the mode's k-space; by Parseval (ortho FFT) this is the same SNR in image space.
    """
    dims = _encoded_dims(x, geometry)
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    snr_db = snr_max_db + (snr_min_db - snr_max_db) * float(theta)
    k = fftnc(x, dims)
    # Each real channel carries half the complex noise power, hence the 1/sqrt(2).
    sigma = _volume_rms(k) / (10 ** (snr_db / 20.0)) / (2.0**0.5)
    return ifftnc(k + _complex_noise(k, sigma, g), dims)


def rician_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    sigma_max: float = 0.3,
) -> Tensor:
    """D2 on a volume: image-space Rician magnitude noise, sigma from the volume's p99."""
    _encoded_dims(x, geometry)
    if float(theta) <= 0.0:
        return x
    mag = x.abs() if x.is_complex() else x
    g = _generator(mag.device, seed)
    sigma = float(theta) * sigma_max * _signal_level(mag).view(-1, 1, 1, 1, 1)
    eta1, eta2 = per_sample_normal_pair(*mag.shape, generator=g, device=mag.device, dtype=mag.dtype)
    return torch.sqrt((mag + eta1 * sigma) ** 2 + (eta2 * sigma) ** 2)


def apodization_window(
    shape: tuple[int, ...], dims: tuple[int, ...], fwhm_px: float, device: torch.device
) -> Tensor:
    """Gaussian k-space window ``exp(-2 pi^2 sigma^2 |f|^2)`` over ``dims``, 1 elsewhere.

    ``f`` is cycles/px on each encoded axis, so the image PSF is a Gaussian of FWHM
    ``fwhm_px`` pixels along every encoded axis. Broadcastable to ``shape``.
    """
    sigma_x = fwhm_px / 2.35482
    ndim = len(shape)
    f2 = torch.zeros((1,) * ndim, device=device)
    for d in dims:
        view = [1] * ndim
        view[d % ndim] = shape[d]
        f2 = f2 + centred_freqs(shape[d], device).view(view) ** 2
    return torch.exp(-2.0 * (math.pi**2) * (sigma_x**2) * f2)


def resolution_snr_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    fwhm_max_px: float = 2.5,
    snr_floor_db: float = 20.0,
) -> Tensor:
    """D28 on a volume: Gaussian apodization on the encoded axes plus coupled noise.

    The noise is referenced to the CLEAN, un-apodized whole-volume k-space RMS.
    """
    dims = _encoded_dims(x, geometry)
    t = float(theta)
    if t <= 0.0:
        return x
    x = _ensure_complex(x)
    k = fftnc(x, dims)
    rms = _volume_rms(k)
    fwhm = t * fwhm_max_px
    if fwhm > 1e-6:
        k = k * apodization_window(tuple(k.shape), dims, fwhm, x.device)
    g = _generator(x.device, seed)
    sigma = t * rms / (10 ** (snr_floor_db / 20.0)) / (2.0**0.5)
    return ifftnc(k + _complex_noise(k, sigma, g), dims)


def emi_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    n_tones: int = 4,
    max_amp_ratio: float = 0.3,
) -> Tensor:
    """D32 on a volume: one interference source per subject, phase per readout line.

    The lines are (slice, ky) in multislice and (kz, ky) in 3-D; both fold to one line axis
    of length ``D * H`` for the 2-D tone model, so the RMS is the whole volume's.
    """
    dims = _encoded_dims(x, geometry)
    if n_tones < 1:
        raise ValueError(f"n_tones must be >= 1; got {n_tones}.")
    if float(theta) <= 0.0:
        return x
    x = _ensure_complex(x)
    b, c, d, h, w = x.shape
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))
    u = (torch.rand(n_tones, generator=g, device=x.device) - 0.5) * (0.9 * w)
    rel_amp = 0.5 + 0.5 * torch.rand(n_tones, generator=g, device=x.device)
    lines = fftnc(x, dims).reshape(b, c, d * h, w)
    k = add_readout_tones(lines, u, rel_amp, float(theta) * max_amp_ratio, g)
    return ifftnc(k.reshape(b, c, d, h, w), dims)


def spike_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    probability: float | None = None,
) -> Tensor:
    """D15 in 3-D: spikes at ``(kz, ky, kx)`` samples of the 3-D k-space, shared by coils."""
    dims = _encoded_dims(x, geometry)
    if geometry.mode is not AcquisitionMode.VOLUME_3D:
        raise ValueError(f"spike_volume is the 3-D realisation; got mode {geometry.mode!r}")
    prob = float(theta) * 0.05 if probability is None else float(probability)
    if prob <= 0.0:
        return x
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        return ifftnc(simulate_spike_noise_volume(fftnc(x, dims), probability=prob), dims)
