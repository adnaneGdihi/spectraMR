"""Electromagnetic interference (EMI) degradation for unshielded low-field scanners.

A ULF scanner without a Faraday cage picks up narrowband radiation from nearby electronics.
Each source is a tone at some frequency inside the receive bandwidth, so it is a complex
exponential along the readout (kx) axis of every phase-encode line. The tone is not locked
to the ADC clock, so its phase is independent from one line to the next. After the inverse
FFT this is a bright stripe along the phase-encode axis at the frequency-encode position of
the tone; a tone between two kx bins also leaks into neighbouring columns.

:func:`add_readout_tones` is the one tone model; :func:`degrade_emi` (the registry axis,
several seeded off-grid tones) and ``simulate_rf_zipper`` (one tone at a chosen position)
both call it. The registry's ``spike`` (isolated k-space samples) is a different artefact.
"""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = ["add_readout_tones", "degrade_emi", "readout_tone_field"]


def readout_tone_field(
    shape: tuple[int, int, int, int],
    positions_px: torch.Tensor,
    amplitudes: torch.Tensor,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Interference from narrowband readout tones, as centred k-space ``[B, C, H, W]``.

    Tone ``j`` sits ``positions_px[j]`` pixels from the frequency-encode centre (it may be
    fractional) with per-sample amplitude ``amplitudes[j]``. Its per-line phase is shared
    by every coil (one source, one time series); each coil couples to it with its own phase.
    """
    B, C, H, W = shape
    J = positions_px.numel()
    draws = per_sample_rand(B, H * J + C * J, generator=generator, device=device)
    line_phase = draws[:, : H * J].view(B, 1, H, J) * (2 * math.pi)
    coil_phase = draws[:, H * J :].view(B, C, 1, J) * (2 * math.pi)
    kx = centred_freqs(W, device)
    # exp(-2 pi i f x0) is the spectrum of a line at x0 (the shift-theorem sign), so a
    # positive position lands to the right of the readout centre.
    tones = torch.exp(-2j * math.pi * positions_px.view(-1, 1) * kx.view(1, -1))  # (J, W)
    weights = amplitudes.to(torch.complex64) * torch.exp(1j * (line_phase + coil_phase))
    return torch.einsum("bchj,jw->bchw", weights, tones)


def add_readout_tones(
    k: torch.Tensor,
    positions_px: torch.Tensor,
    rel_amp: torch.Tensor,
    rms_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add readout tones to ``k`` with summed RMS ``rms_ratio`` x each sample's k-space RMS.

    ``rel_amp`` sets the tones' relative amplitudes; see :func:`readout_tone_field`.
    """
    rel_amp = rel_amp / rel_amp.pow(2).sum().sqrt()
    field = readout_tone_field(tuple(k.shape), positions_px, rel_amp, generator, k.device)
    rms = k.abs().pow(2).mean(dim=(-3, -2, -1), keepdim=True).sqrt()
    return k + float(rms_ratio) * rms * field


def degrade_emi(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    n_tones: int = 4,
    max_amp_ratio: float = 0.3,
) -> torch.Tensor:
    r"""Add ``n_tones`` narrowband interference tones to every phase-encode line.

    ``theta = 0`` --- no interference (identity).
    ``theta = 1`` --- the summed interference has RMS ``max_amp_ratio`` times the k-space RMS.

    The tone frequencies, relative amplitudes and per-line phases are fixed by ``seed``
    and do not depend on ``theta``, so ``theta`` is the only free variable.
    """
    if n_tones < 1:
        raise ValueError(f"n_tones must be >= 1; got {n_tones}.")
    if not x.is_complex():
        x = torch.complex(x, torch.zeros_like(x))
    W = x.shape[-1]
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))

    # Tone position along the frequency-encode axis, in (fractional) columns from centre.
    u = (torch.rand(n_tones, generator=g, device=x.device) - 0.5) * (0.9 * W)
    rel_amp = 0.5 + 0.5 * torch.rand(n_tones, generator=g, device=x.device)
    return ifft2c(add_readout_tones(fft2c(x), u, rel_amp, float(theta) * max_amp_ratio, g))
