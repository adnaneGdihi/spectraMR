"""Radial annuli over a centred k-space grid: the one owner of that partition.

A radial band is a ring of the frequency plane. Three things in this repository
want one and each had written its own: the spectral-transfer probe (which
measured the cohort's outer-band deficit), :func:`~spectramr.core.metrics.srf_bound.radial_band_energies`
and :func:`~spectramr.core.metrics.meta_evaluation.descriptors.radial_spectral_profile`.
Three definitions of "annulus" is three answers to "how much energy is in the
outer band", and a mechanism that corrects the band must be measured against the
same partition it acts on or the two cannot be compared (non-negotiable 17).

This module lives under ``infrastructure/physics`` deliberately. ``models/`` may
import it -- that is the one carve-out in the layering direction
(``scripts/ci/check_layering.sh:126``) -- so a model-side band mechanism and a
validation-side band probe can share this code without an upward import.

Nothing here touches ``torch.fft``: the grid is built from ``arange`` and
compared by radius, so non-negotiable 2 does not apply and the module is not a
second FFT owner.
"""

from __future__ import annotations

import torch

__all__ = ["band_counts", "band_reduce", "radial_bins"]

_EPS = 1e-12


def radial_bins(
    height: int, width: int, n_bins: int, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(bin_index, inside_nyquist, edges)`` over a centred (H, W) frequency grid.

    Coordinates run ``-1 .. +1`` with zero at index ``N // 2`` -- identical to
    ``fftshift(fftfreq(N)) * 2`` (verified exactly) but written from ``arange``
    so nothing here reads as an FFT of MRI data (non-negotiable 2).

    Corners carry ``r`` up to ``sqrt(2)`` and are **excluded**: an annulus that
    exists only along the diagonals is not the same population as the bins below
    it, and pooling it into the last bin quietly mixes the two.

    Bins are equal-width in ``r``. That is a choice the probe made and this
    module keeps, because the mechanism and the measurement must partition
    identically -- log-spacing the mechanism alone would make its bands
    unreadable by the probe that grades it.

    Args:
        height: Grid rows.
        width: Grid columns.
        n_bins: Number of annuli; at least 2.
        device: Where to build the index.

    Returns:
        ``index`` ``[H, W]`` long, ``inside`` ``[H, W]`` bool (``r <= 1``), and
        ``edges`` ``[n_bins + 1]`` in ``r/Nyquist``.

    Raises:
        ValueError: If ``n_bins < 2``.
    """
    if n_bins < 2:
        raise ValueError(f"radial_bands: n_bins must be >= 2, got {n_bins}.")
    yy = (torch.arange(height, device=device, dtype=torch.float32) - height // 2) / (height / 2.0)
    xx = (torch.arange(width, device=device, dtype=torch.float32) - width // 2) / (width / 2.0)
    radius = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    inside = radius <= 1.0
    index = torch.clamp((radius * n_bins).long(), max=n_bins - 1)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    return index, inside, edges


def band_counts(index: torch.Tensor, inside: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Bins per annulus, ``[n_bins]``.

    An empty annulus makes every per-band statistic above it a division by zero
    dressed as a number, so callers check this rather than clamping it away.
    """
    flat = index[inside]
    return torch.zeros(n_bins, device=index.device).index_add_(
        0, flat, torch.ones_like(flat, dtype=torch.float32)
    )


def band_reduce(
    plane: torch.Tensor, index: torch.Tensor, inside: torch.Tensor, n_bins: int
) -> torch.Tensor:
    """Sum a ``[H, W]`` per-bin quantity into ``[n_bins]`` annuli.

    ``plane`` is already a real per-bin scalar (an amplitude or a power), not a
    complex field: reducing complex values would cancel across an annulus and
    report a near-zero "energy" for a band that is full of signal.
    """
    if plane.is_complex():
        raise ValueError(
            "radial_bands.band_reduce needs a real per-bin quantity; a complex "
            "field cancels across an annulus. Pass .abs() or .abs()**2."
        )
    return torch.zeros(n_bins, device=plane.device, dtype=torch.float32).index_add_(
        0, index[inside], plane[inside].float()
    )
