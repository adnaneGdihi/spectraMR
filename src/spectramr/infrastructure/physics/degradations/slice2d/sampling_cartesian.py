"""2-D Cartesian sampling kernels: PE-line (D3), variable-density lines (D4b), partial Fourier
(D24) and k-space truncation (D9)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.digital_twin_simulator import (
    outer_line_count,
    simulate_gibbs_ringing,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c


def degrade_cartesian_undersampling(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,  # noqa: N803
    center_fraction: float = 0.08,
) -> torch.Tensor:
    """D3 — Cartesian PE-line undersampling.

    Acceleration ramps from 1.0 (theta=0) to ``R_max`` (theta=1), and the
    **realised** rate matches the declared one: exactly ``round(H / R)`` PE lines
    are acquired, the ACS centre included in that budget, with the outer lines
    drawn from the non-ACS pool (issue #246 — the old form both over-budgeted the
    outer lines and drew them from a permutation of *all* H lines, so draws that
    landed back on the ACS were wasted; a declared R = 8 realised R = 5.3).
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x
    H = x.shape[-2]
    device = x.device
    g = _generator(device, seed)

    n_center = max(1, int(H * center_fraction))
    cs = H // 2 - n_center // 2
    ce = cs + n_center
    n_outer = outer_line_count(H, n_center, R)

    outer_pool = torch.cat(
        [
            torch.arange(0, cs, device=device),
            torch.arange(ce, H, device=device),
        ]
    )
    perm = torch.randperm(outer_pool.numel(), generator=g, device=device)
    outer_idx = outer_pool[perm[:n_outer]]

    mask = torch.zeros(1, 1, H, 1, device=device)
    mask[..., cs:ce, :] = 1.0
    mask[..., outer_idx, :] = 1.0
    return ifft2c(fft2c(x) * mask)


def degrade_variable_density_cartesian(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,  # noqa: N803
    alpha: float = 3.0,
    center_fraction: float = 0.08,
) -> torch.Tensor:
    r"""D4b --- Variable-density *Cartesian* PE-line sampling.

    Samples whole phase-encode *lines* (each readout is acquired or
    skipped together, the physical reality of Cartesian acquisition).
    Line probability decays radially as
    :math:`p(k_y) \propto (1 + |k_y|)^{-\alpha}` and the threshold is
    calibrated against the target acceleration by quantile so the
    realised count matches the declared ``R`` (this is what the prior
    2-D Bernoulli version did *not* honour).

    M5 sibling (digital-twin review).
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x
    H = x.shape[-2]
    device = x.device
    g = _generator(device, seed)

    n_center = max(1, int(H * center_fraction))
    cs = H // 2 - n_center // 2
    ce = cs + n_center

    ky = 2.0 * centred_freqs(H, device)  # [-1, 1), zero on the DC line
    p_lines = (1.0 + ky.abs()) ** (-alpha)
    # Mask out center lines so they don't compete with outer-line draws.
    p_lines_outer = p_lines.clone()
    p_lines_outer[cs:ce] = 0.0

    # Outer-line budget: the ACS lines count TOWARDS the declared acceleration,
    # they are not free (issue #246 — the old `(H - n_center) / R` budget realised
    # R = 5.3 at a declared R = 8).
    n_target = max(1, outer_line_count(H, n_center, R))
    # Incoherent variable-density draw: keep n_target DISTINCT outer lines sampled
    # WITHOUT replacement with probability proportional to p(k_y). ``topk`` would
    # deterministically keep the highest-p (central) lines -- a contiguous low-pass
    # block (resolution loss + Gibbs ringing), which is NOT the incoherent aliasing
    # that variable-density sampling is defined by (Lustig 2007). ``multinomial`` keeps
    # the EXACT declared count (honours #246) while distributing the kept lines
    # incoherently across the spectrum.
    top = torch.multinomial(p_lines_outer, n_target, replacement=False, generator=g)

    mask = torch.zeros(1, 1, H, 1, device=device)
    mask[..., cs:ce, :] = 1.0
    mask[..., top, :] = 1.0
    return ifft2c(fft2c(x) * mask)


def degrade_partial_fourier(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    min_fraction: float = 0.625,
    axis: str = "ky",
) -> torch.Tensor:
    r"""Partial-Fourier acquisition along the phase-encode axis.

    Acquires only ``frac`` of k-space lines (the high-frequency half is
    zeroed); a homodyne-style POCS reconstruction would invert this
    later but the digital twin emits the raw partially-acquired
    k-space. Used in TSE / FSE to shorten echo trains.

    ``theta = 0`` --- full acquisition (no truncation).
    ``theta = 1`` --- minimum fraction (default 5/8 = 0.625).
    """
    x = _ensure_complex(x)
    frac = 1.0 - (1.0 - min_fraction) * float(theta)
    H, W = x.shape[-2], x.shape[-1]
    device = x.device
    k = fft2c(x)
    mask = torch.ones(1, 1, H, W, device=device)
    # `mask[..., -cut:, :]` is a trap at cut == 0: -0 == 0, so the slice is [0:] -- the
    # WHOLE axis -- and the "truncate nothing" case zeroed the entire mask, returning a
    # BLACK image. theta=0 gives frac=1.0 gives cut=0, so the declared clean anchor of
    # this axis was an all-zero image (as was every theta < 1/(0.375*H): theta <= 0.02 at
    # H=128). Index from the front and skip the write when there is nothing to cut.
    if axis == "ky":
        cut = round(H * (1.0 - frac))
        if cut > 0:
            mask[..., H - cut :, :] = 0.0
    elif axis == "kx":
        cut = round(W * (1.0 - frac))
        if cut > 0:
            mask[..., W - cut :] = 0.0
    else:
        raise ValueError(f"axis must be 'ky' or 'kx'; got {axis!r}.")
    return ifft2c(k * mask)


def _delegate_gibbs(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    # theta=1 retains 8.5% of the k-space RADIUS, producing heavy truncation
    # ringing. The floor moved 0.12 -> 0.085 when the window's radius was
    # normalised by its true maximum (#246): 0.12 / sqrt(2) = 0.085 is the same
    # physical window as before, so the theta=1 severity is unchanged while
    # theta=0 becomes a genuine identity instead of a 22.8%-of-k-space pedestal.
    trunc = float(kw.get("truncation_fraction", 1.0 - float(theta) * 0.915))
    return ifft2c(simulate_gibbs_ringing(fft2c(x), truncation_fraction=trunc))
