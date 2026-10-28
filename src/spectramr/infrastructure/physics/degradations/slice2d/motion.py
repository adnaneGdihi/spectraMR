"""2-D motion kernels: rigid (D6), pulsatile (D7), drift (D29), sudden (D30), random (D31)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.digital_twin_simulator import simulate_rigid_motion_kspace
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c


def degrade_pulsatile_motion(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_amplitude_px: float = 9.0,
    base_freq_hz: float = 1.2,
    tr_s: float = 0.5,
    n_phase_bins: int = 16,
) -> torch.Tensor:
    r"""D7 --- Cardiac-like pulsatile non-rigid deformation.

    The pulsation amplitude is evaluated at the per-PE-line acquisition
    time :math:`t(k_y) = k_y \cdot \mathrm{TR} / N_{\mathrm{PE}}`, so
    different PE lines see different points of the cardiac cycle. This
    is what makes the artefact ghost in the PE direction; a single
    static-amplitude pulse (the prior implementation) would just be a
    static deformation and a reconstructor could invert it. The
    pulsation modulates the k-space along PE rather than the image as
    a whole (Bernstein, King & Zhou 2004, Ch. 11).

    Args:
        x: complex image ``[B, C, H, W]``.
        theta: severity in ``[0, 1]``.
        seed: deterministic per-call seed.
        max_amplitude_px: peak displacement at theta=1, in pixels.
        base_freq_hz: pulsation frequency (cardiac ~1.2 Hz default).
        tr_s: TR in seconds; PE line ``k_y`` is acquired at ``t = k_y * TR``.
        n_phase_bins: the cardiac phase is quantised into this many rendered poses.
    """
    x = _ensure_complex(x)
    B, C, H, W = x.shape
    device = x.device
    if float(theta) <= 0.0:
        return x
    g = _generator(device, seed)

    rdt = x.real.dtype
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=rdt),
        torch.linspace(-1, 1, W, device=device, dtype=rdt),
        indexing="ij",
    )
    # Radial displacement |d| = A * r exp(-r^2/0.5) / peak, whose peak (at r = 0.5) is A.
    envelope = torch.exp(-((X**2 + Y**2) / 0.5)) / (0.5 * math.exp(-0.5))
    amp = float(theta) * max_amplitude_px * 2.0 / max(max(H, W) - 1, 1)  # px -> grid units

    # Each PE line sees the cardiac phase at its own acquisition time. Sampling the
    # pulsation once per TR aliases it exactly as a real ungated scan does, which is what
    # sets where the ghosts land; nothing here caps or resamples the frequency.
    phase0 = torch.rand(1, generator=g, device=device) * (2 * math.pi)
    line_phase = (2 * math.pi * base_freq_hz * tr_s * torch.arange(H, device=device) + phase0) % (
        2 * math.pi
    )
    line_bin = (line_phase / (2 * math.pi) * n_phase_bins).long().clamp(max=n_phase_bins - 1)
    bin_centre = (torch.arange(n_phase_bins, device=device, dtype=rdt) + 0.5) * (
        2 * math.pi / n_phase_bins
    )
    pulse = torch.sin(bin_centre).view(-1, 1, 1)  # [P, 1, 1]

    disp = amp * pulse * envelope  # [P, H, W]
    grid = torch.stack([X + disp * X, Y + disp * Y], dim=-1)  # [P, H, W, 2]
    grid = grid.unsqueeze(1).expand(-1, B, -1, -1, -1).reshape(n_phase_bins * B, H, W, 2)
    rep = x.unsqueeze(0).expand(n_phase_bins, -1, -1, -1, -1).reshape(n_phase_bins * B, C, H, W)
    real = F.grid_sample(rep.real, grid, mode="bilinear", padding_mode="border", align_corners=True)
    imag = F.grid_sample(rep.imag, grid, mode="bilinear", padding_mode="border", align_corners=True)
    poses_k = fft2c(torch.complex(real, imag)).view(n_phase_bins, B, C, H, W)

    idx = line_bin.view(1, 1, 1, H, 1).expand(1, B, C, H, W)
    return ifft2c(torch.gather(poses_k, 0, idx).squeeze(0))


# ──────────────────────────────────────────────────────────────────────
# Motion types: coherent drift / abrupt jump / random jitter (D29-D31)
#
# Distinct motion TYPES beyond rigid_motion (D6, random per-PE-line SE(2)) and
# pulsatile_motion (D7, periodic non-rigid). Each displaces the object by d(k_y)
# px along READOUT while phase-encode line k_y is acquired -- a per-PE-line
# Fourier shift exp(-i 2pi k_x d(k_y)) on centred k-space. A per-line-VARYING d
# is what produces ghosting (a constant shift is globally recoverable). The
# schedule d(k_y) is what differs:
#   translation_drift : linear ramp   -> coherent directional smear
#   sudden_motion     : single step   -> discrete double-image ghost
#   random_motion     : i.i.d. jitter -> broadband ghosting (the pure-translation
#                       sibling of rigid_motion, no rotation)
# All are deterministic in (theta, seed): drift/step are closed-form; random_motion
# draws from a seeded generator.
# ──────────────────────────────────────────────────────────────────────


def _apply_pe_line_shift(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Displace the object by ``dx[k_y]`` px along readout while PE line ``k_y`` is
    acquired: multiply centred k-space by ``exp(-i 2pi k_x dx(k_y))`` (Fourier shift
    theorem, per line)."""
    k = fft2c(x)
    h, w = k.shape[-2], k.shape[-1]
    kx = centred_freqs(w, k.device).view(1, 1, 1, -1)
    dxv = dx.to(device=k.device, dtype=torch.float32).view(1, 1, h, 1)
    return ifft2c(k * torch.exp(-1j * 2.0 * math.pi * kx * dxv))


def degrade_translation_drift(
    x: torch.Tensor, theta: float, seed: int = 0, max_drift_px: float = 10.0
) -> torch.Tensor:
    """D29 -- coherent linear translational drift across the acquisition."""
    x = _ensure_complex(x)
    t = float(theta)
    if t <= 0.0:
        return x
    h = x.shape[-2]
    ky = torch.arange(h, device=x.device, dtype=torch.float32)
    dx = t * max_drift_px * (2.0 * ky / max(h - 1, 1) - 1.0)
    return _apply_pe_line_shift(x, dx)


def degrade_sudden_motion(
    x: torch.Tensor, theta: float, seed: int = 0, max_shift_px: float = 12.0
) -> torch.Tensor:
    """D30 -- single abrupt bulk shift partway through the acquisition."""
    x = _ensure_complex(x)
    t = float(theta)
    if t <= 0.0:
        return x
    h = x.shape[-2]
    dx = torch.zeros(h, device=x.device, dtype=torch.float32)
    dx[h // 2 :] = t * max_shift_px  # step at the k-space centre
    return _apply_pe_line_shift(x, dx)


def degrade_random_motion(
    x: torch.Tensor, theta: float, seed: int = 0, max_shift_px: float = 3.0
) -> torch.Tensor:
    """D31 -- i.i.d. per-PE-line random translational jitter (broadband ghosting)."""
    x = _ensure_complex(x)
    t = float(theta)
    if t <= 0.0:
        return x
    h = x.shape[-2]
    g = _generator(x.device, seed)
    dx = (torch.rand(h, generator=g, device=x.device) * 2.0 - 1.0) * (t * max_shift_px)
    return _apply_pe_line_shift(x, dx)


def _delegate_rigid_motion(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    kspace = simulate_rigid_motion_kspace(
        fft2c(x),
        max_translation=float(kw.get("max_translation", float(theta) * 10.0)),
        max_rotation=float(kw.get("max_rotation", float(theta) * 0.1)),
        generator=g,
    )
    return ifft2c(kspace)
