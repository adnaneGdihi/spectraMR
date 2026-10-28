"""Band-energy envelope of measured k-space, and the divisor that enforces it.

A reverse step writes the model's estimate onto coefficients the measurement
does not cover. Nothing in the network output bounds the ENERGY of those writes:
an estimate that overshoots by a factor ``g`` feeds the next model input, whose
estimate overshoots again, so a loop that re-reads its own writes compounds the
error with the number of calls. The per-coefficient magnitude clamp caps each
coefficient but flattens every one above the cap to the same modulus, erasing
the relative structure inside a band.

This module enforces the bound at the level of a radial band instead. The
reference is the root-mean-square physical magnitude of the MEASURED
coefficients in each band; the written coefficients of a band are divided by

.. math::

    d_b = \\max\\left(1, \\frac{\\mathrm{RMS}_b(\\hat{x}_0)}{\\rho\\,\\mathrm{RMS}_b(y)}\\right),

a single phase-preserving factor per band. A band whose estimate is within
``rho`` of the measured energy is left untouched, so the adjuster can only
remove energy, never add it. Relative magnitudes inside the band survive because
the whole band shares one divisor.

Every measured Cartesian line crosses most radial bands (a line spans the whole
readout), so the envelope is populated even at the highest rung; the corner
bands a line does not reach inherit the nearest populated inner band.
Observed bands are NOT forced monotone: a per-band reference below an inner one
is data, and forcing it down is what makes the max-based
``band_local_magnitude_ceiling`` clip true coefficients at R = 32.

Magnitudes are compared in PHYSICAL units: under ``log1p`` compression the
modulus is expanded with ``expm1`` before squaring and compressed again after
division, as ``apply_ceiling_ratio`` does for the per-coefficient ceiling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

#: Radial bands across the half-diagonal; matches ``band_local_magnitude_ceiling``.
ENVELOPE_NUM_BANDS = 16


def per_coefficient_support(support: Tensor, x: Tensor) -> Tensor:
    """Reduce a support mask to one entry per complex coefficient of ``x``.

    Supports arrive channel-broadcast (``[B, 1, H, W]``) or per channel. A
    per-channel support on real-stacked ``x`` is paired down to its Re channels;
    the Im channel of an interleaved pair always carries the same flag.
    """
    if support.shape[-3] == 1 or torch.is_complex(x):
        return support
    return support[..., 0::2, :, :]


def _physical(magnitude: Tensor, log_scaled: bool) -> Tensor:
    return torch.expm1(magnitude) if log_scaled else magnitude


def _band_rms(
    magnitude: Tensor, support: Tensor, band: Tensor, num_bands: int, log_scaled: bool
) -> tuple[Tensor, Tensor]:
    """Per-sample, per-band physical RMS over ``support``; ``[B, nb]`` value and count."""
    phys = _physical(magnitude, log_scaled)
    weight = support.expand_as(phys).to(phys.dtype)
    energy = (phys * phys * weight).sum(dim=-3).flatten(1)  # [B, H*W]
    count = weight.sum(dim=-3).flatten(1)
    index = band.flatten().unsqueeze(0).expand(energy.shape[0], -1)
    zeros = energy.new_zeros((energy.shape[0], num_bands))
    band_energy = zeros.scatter_add(1, index, energy)
    band_count = zeros.clone().scatter_add(1, index, count)
    return torch.sqrt(band_energy / band_count.clamp_min(1.0)), band_count


def observed_band_envelope(
    measurement: Tensor,
    obs: Tensor,
    *,
    log_scaled: bool,
    num_bands: int = ENVELOPE_NUM_BANDS,
    eps: float = 1e-8,
) -> Tensor:
    """Physical RMS magnitude of the measured coefficients per radial band, ``[B, nb]``.

    Bands with no measured coefficient take the value of the nearest populated
    band, searching inward first and then outward, so a sample with any measured
    coefficient yields a strictly positive envelope in every band.
    """
    from spectramr.models.diffusion.kspace_process import _radial_band_partition, paired_magnitude

    mag = paired_magnitude(measurement)
    band = _radial_band_partition(mag.shape[-2], mag.shape[-1], num_bands, eps, mag.device)
    observed = per_coefficient_support(obs, measurement) > 0
    rms, count = _band_rms(mag, observed, band, num_bands, log_scaled)
    populated = count > 0
    seen = populated.to(torch.int8).cummax(dim=1).values.bool()
    filled = rms.clone()
    carry = rms[:, 0]
    for b in range(num_bands):
        carry = torch.where(populated[:, b], rms[:, b], carry)
        filled[:, b] = carry
    carry = rms[:, -1]
    for b in range(num_bands - 1, -1, -1):
        carry = torch.where(populated[:, b], rms[:, b], carry)
        filled[:, b] = torch.where(seen[:, b], filled[:, b], carry)
    return filled.clamp_min(eps)


def normalize_band_energy(
    x0: Tensor,
    envelope: Tensor,
    support: Tensor,
    *,
    ratio: float,
    log_scaled: bool,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Divide each band of ``x0`` on ``support`` by ``max(1, RMS_b / (ratio * envelope_b))``.

    Args:
        x0: estimate ``[B, C, H, W]``, complex or interleaved real-stacked.
        envelope: ``[B, nb]`` from :func:`observed_band_envelope`.
        support: coefficients this step writes; zero elsewhere, where ``x0`` is
            returned unchanged.
        ratio: allowed physical energy ratio over the measured envelope.
        log_scaled: whether ``x0`` holds ``log1p``-compressed magnitudes.

    Returns:
        The adjusted estimate and the ``[B, nb]`` divisors (1 where untouched).
    """
    from spectramr.models.diffusion.kspace_process import _radial_band_partition, paired_magnitude

    num_bands = envelope.shape[-1]
    mag = paired_magnitude(x0)
    band = _radial_band_partition(mag.shape[-2], mag.shape[-1], num_bands, eps, mag.device)
    written = per_coefficient_support(support, x0) > 0
    rms, _ = _band_rms(mag, written, band, num_bands, log_scaled)
    divisor = (rms / (ratio * envelope)).clamp_min(1.0)

    per_pixel = divisor[:, band.flatten()].reshape(divisor.shape[0], 1, *band.shape)
    return divide_physical_magnitude(x0, per_pixel, written, log_scaled=log_scaled, eps=eps), divisor


def divide_physical_magnitude(
    x0: Tensor, divisor: Tensor, written: Tensor, *, log_scaled: bool, eps: float = 1e-8
) -> Tensor:
    """Divide the PHYSICAL modulus of ``x0`` by ``divisor`` on ``written``, keeping the phase.

    ``divisor`` and ``written`` broadcast against the per-coefficient modulus
    ``[B, C//2, H, W]``; under log compression the division happens after
    ``expm1`` and the result is compressed again.
    """
    from spectramr.models.diffusion.kspace_process import paired_magnitude

    mag = paired_magnitude(x0)
    phys = _physical(mag, log_scaled) / divisor
    target = torch.log1p(phys) if log_scaled else phys
    factor = torch.where(written & (mag > eps), target / mag.clamp_min(eps), torch.ones_like(mag))
    if not torch.is_complex(x0):
        factor = factor.repeat_interleave(2, dim=-3)
    return x0 * factor


__all__ = [
    "ENVELOPE_NUM_BANDS",
    "divide_physical_magnitude",
    "normalize_band_energy",
    "observed_band_envelope",
    "per_coefficient_support",
]
