"""2-D k-space phase kernels: N/2 ghost (D8), eddy (D14), concomitant phase (D23), I/Q ghost
(D25), stimulated echo (D26)."""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _isolated_rng,
)
from spectramr.infrastructure.physics.digital_twin_simulator import (
    simulate_concomitant_phase,
    simulate_eddy_current,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c


def degrade_nyquist_ghost(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_phase_rad: float = math.pi,
) -> torch.Tensor:
    """D8 — N/2 ghost from EPI even/odd readout phase mismatch.

    Even readout lines receive zero phase, odd lines receive
    ``Δφ = theta * max_phase_rad``. This produces a ghost at FOV/2.

    Formula (per Bernstein et al.):
        K[k_y odd] *= exp(j Δφ)
    """
    x = _ensure_complex(x)
    H = x.shape[-2]
    device = x.device

    dphi = float(theta) * max_phase_rad
    line_phase = torch.zeros(H, device=device)
    line_phase[1::2] = dphi
    phase_ramp = torch.exp(1j * line_phase).view(1, 1, H, 1)

    return ifft2c(fft2c(x) * phase_ramp)


def degrade_iq_ghost(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_dc_offset: float = 0.20,
    max_iq_imbalance: float = 0.35,
) -> torch.Tensor:
    r"""Quadrature-receiver ghost: DC offset + I/Q amplitude/phase mismatch.

    Distinct from the N/2 EPI ghost (D8): the artefact is an additive
    central-FOV spike (DC) and a conjugate-symmetric mirror image
    (I/Q imbalance). Modelled as a constant added to k-space DC plus
    a small fraction of the conjugate-flipped image.
    """
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    H, W = x.shape[-2], x.shape[-1]
    cy, cx = H // 2, W // 2

    rand2 = torch.rand(2, generator=g, device=x.device)
    dc_amp = max_dc_offset * float(theta) * (1.0 + 0.1 * (rand2[0] - 0.5))
    iq_amp = max_iq_imbalance * float(theta) * (1.0 + 0.1 * (rand2[1] - 0.5))

    # Conjugate mirror x*(-r) about the FOV centre at N // 2. ``flip`` reflects about
    # (N - 1) / 2, so even axes need a one-sample roll to put the centre back on N // 2.
    mirror = x.conj().flip(dims=(-2, -1))
    mirror = torch.roll(mirror, shifts=(1 - H % 2, 1 - W % 2), dims=(-2, -1))
    out = x + iq_amp * mirror

    # Receiver DC offset -> the classic central-point artifact: a bright spike at the
    # FOV CENTRE. Physically a real constant added to every k-space ADC sample IFFTs to a
    # delta at the image centre; adding it to the SINGLE k-space DC sample instead (as
    # before) IFFTs to a flat image-wide pedestal, which is not the named central-FOV
    # spike (advisor). Realise the central point directly in image space.
    # Per-sample peak, shared by that sample's coils, so a bright neighbour in the batch
    # does not set this image's spike.
    peak = out.abs().amax(dim=(-3, -2, -1))
    out[..., cy, cx] = out[..., cy, cx] + dc_amp * peak.view(-1, 1)
    return out


def degrade_stimulated_echo(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_amp: float = 0.40,
    max_shift_px: float = 6.0,
) -> torch.Tensor:
    r"""Stimulated-echo ghost from imperfect refocusing in TSE / FSE.

    A small fraction of the signal is delayed by an integer number of
    echo spacings and re-imaged with a per-line phase ramp, producing
    a low-amplitude ghost shifted along the PE axis. Relevant for
    the TSE protocol used in M4Raw (Lyu et al. 2023).
    """
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    H, W = x.shape[-2], x.shape[-1]  # noqa: F841

    # Seeded jitter in [0.7, 1.0] of the declared shift, so max_shift_px is the true ceiling.
    rand = torch.rand(1, generator=g, device=x.device)
    shift_px = float(theta) * max_shift_px * (0.7 + 0.3 * rand)
    amp = float(theta) * max_amp

    ky = centred_freqs(H, x.device).view(1, 1, -1, 1)
    phase_ramp = torch.exp(-1j * 2 * math.pi * ky * shift_px)

    k = fft2c(x)
    return ifft2c(k + amp * k * phase_ramp)


def degrade_concomitant_phase(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    b0_T: float = 0.3,  # noqa: N803
    max_gradient_mT_per_m: float = 30.0,  # noqa: N803
    voxel_size_mm: float = 1.0,
    duration_ms: float = 5.0,
) -> torch.Tensor:
    """Min1 entry point for Maxwell concomitant phase as a D-axis."""

    x = _ensure_complex(x)
    grad = float(theta) * max_gradient_mT_per_m
    return simulate_concomitant_phase(
        x,
        im_size=x.shape[-2:],
        b0_T=b0_T,
        gradient_amplitudes_mT_per_m=grad,
        voxel_size_mm=voxel_size_mm,
        duration_ms=duration_ms,
    )


def _delegate_eddy(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        # Strength controls per-PE-line phase accumulation (radians).
        # Bumped 0.5 -> 5.0 so theta=1 produces a visible shearing
        # ghost rather than the prior near-imperceptible phase tilt.
        strength = float(kw.get("strength", float(theta) * 5.0))
        out = ifft2c(simulate_eddy_current(fft2c(x), strength=strength))
    return out
