"""Translation registration of M4Raw repetitions before they are averaged (#2849).

The NEX target is the phase-aligned mean of a group's repetitions. A subject who
moved between repetitions leaves a translation, a linear phase ramp in k-space,
that the per-coil global phase alignment cannot remove, so the mean superposes
shifted images. Measured on the kspace_filling training manifests (60 groups),
25 groups shift by more than 0.5 px (T1 median 0.73 px, maximum 1.14 px), and the
unregistered target differs from the registered one by 5-8 % (median per contrast).

The shift of each repetition is estimated once per group, by phase correlation of
coil-combined magnitude images summed over a fixed window of central slices, and
undone in k-space by the opposite phase ramp. A ramp multiplies every bin, so a
column that was never acquired stays exactly zero and the acquired-line support is
unchanged. Through-plane motion and rotation are not corrected.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import centred_freqs, ifft2c

__all__ = [
    "REGISTRATION_SLICES",
    "SHIFT_UPSAMPLE",
    "VALID_NEX_REGISTRATIONS",
    "estimate_shift",
    "group_shifts",
    "registration_window",
    "shift_kspace",
]

#: ``none`` averages repetitions as acquired; ``translation`` registers each to the anchor.
VALID_NEX_REGISTRATIONS: frozenset[str] = frozenset({"none", "translation"})

#: Central slices one group's shifts are estimated on. Edge slices hold little
#: anatomy, and one shift per group keeps every slice of the group consistent.
REGISTRATION_SLICES: int = 4

#: Sub-pixel grid of :func:`estimate_shift` (``1/SHIFT_UPSAMPLE`` px); stored shifts record it.
SHIFT_UPSAMPLE: int = 50


def registration_window(num_slices: int) -> range:
    """The central slices of a ``num_slices`` volume the shifts are estimated on."""
    lo = max(0, (int(num_slices) - REGISTRATION_SLICES) // 2)
    return range(lo, min(int(num_slices), lo + REGISTRATION_SLICES))


def _cross_power(reference: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    """``(C, H, W)`` per-coil cross-power ``sum_s ref * conj(moving)`` of ``(S, [C,] H, W)`` k-space."""
    if reference.shape != moving.shape or reference.dim() not in (3, 4):
        raise ValueError(
            "[M4Raw] registration needs two (S, [C,] H, W) k-space volumes of one shape, got "
            f"{tuple(reference.shape)} and {tuple(moving.shape)}."
        )
    if reference.dim() == 3:
        reference, moving = reference.unsqueeze(1), moving.unsqueeze(1)
    ref = reference.to(torch.complex128)
    return (ref * moving.to(torch.complex128).conj()).sum(dim=0)


def _centred_frequencies(n: int) -> torch.Tensor:
    """Cycles per pixel of each bin of a centred ``n``-point k-space axis, in float64."""
    return centred_freqs(n, dtype=torch.float64)


def estimate_shift(
    reference: torch.Tensor, moving: torch.Tensor, upsample: int = SHIFT_UPSAMPLE
) -> tuple[float, float]:
    """``(dy, dx)`` translation of ``moving`` against ``reference``, to ``1/upsample`` px.

    Both are centred ``(S, [C,] H, W)`` k-space over the same slices. A translation
    multiplies k-space by a phase ramp, so each coil's cross-power spectrum is a ramp
    times ``|F|^2`` times that coil's receiver phase between the two acquisitions; the
    modulus of its inverse transform peaks at minus the shift whatever that phase is.
    The coils' moduli are summed, the integer peak is located, and the peak is then
    refined on a grid ``upsample`` times finer by a matrix DFT around it
    (Guizar-Sicairos et al., Opt. Lett. 2008).
    """
    product = _cross_power(reference, moving)
    h, w = product.shape[-2:]
    surface = ifft2c(product).abs().sum(dim=0)
    py, px = divmod(int(torch.argmax(surface)), w)
    coarse_y, coarse_x = float(py - h // 2), float(px - w // 2)
    region = -(-3 * upsample // 2)
    grid = (torch.arange(region, dtype=torch.float64) - region // 2) / upsample
    ey = torch.exp(2j * torch.pi * (coarse_y + grid)[:, None] * _centred_frequencies(h)[None, :])
    ex = torch.exp(2j * torch.pi * (coarse_x + grid)[:, None] * _centred_frequencies(w)[None, :])
    fine = (ey @ product @ ex.T).abs().sum(dim=0)
    my, mx = divmod(int(torch.argmax(fine)), region)
    return -(coarse_y + float(grid[my])), -(coarse_x + float(grid[mx]))


def shift_kspace(kspace: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Centred k-space of the image translated by ``(dy, dx)`` pixels.

    The Fourier shift theorem on the centred grid: each bin is multiplied by
    ``exp(-2 pi i (ky dy + kx dx))``, so a zero bin stays exactly zero.
    """
    if dy == 0.0 and dx == 0.0:
        return kspace
    h, w = kspace.shape[-2:]
    ky = _centred_frequencies(h)[:, None]
    kx = _centred_frequencies(w)[None, :]
    ramp = torch.exp(-2j * torch.pi * (ky * dy + kx * dx)).to(kspace.dtype)
    return kspace * ramp.to(kspace.device)


def group_shifts(volumes: list[torch.Tensor]) -> list[tuple[float, float]]:
    """Each volume's shift against the first (``(0, 0)`` for the first itself).

    ``volumes`` are the :func:`registration_window` slices of one group's repetitions.
    """
    return [(0.0, 0.0)] + [estimate_shift(volumes[0], v) for v in volumes[1:]]
