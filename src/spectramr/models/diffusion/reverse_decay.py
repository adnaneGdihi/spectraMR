"""Power-law decay envelope of the measured bins, and the bound it puts on the rest.

Algorithm F of ``docs/explanation/kspace_reverse_processes.md``. The band
envelope of ``reverse_envelope`` bounds a written band by the measured energy
OF THAT BAND, which at R = 32 is measured on a handful of central lines and
inherited outward. The rows a high rung leaves unmeasured are the outer ones,
and their true energy is far below any inner reference: bounding them by an
inner band admits the blind extrapolation the loop was supposed to stop.

Natural-image k-space decays as a power of the distance from the centre, so the
measured bins fix the whole profile. A bin is a phase-encode row (distance
``|ky|``) or a shell of radial ``|k|`` (``reverse_decay_axis``). Per sample,
over measured bins at distance ``>= 1`` whose power clears twice the noise floor,

.. math::

    \\log S(d) = \\log a + s \\log d, \\qquad
    S^2 = P_{\\mathrm{bin}} - \\sigma^2,

where ``sigma^2`` is the noise power of the ``noise_adaptive`` data-consistency
estimator, the one owner of that estimate. Every UNMEASURED coefficient in a bin
at distance ``>= 1`` is then bounded twice, both bounds only ever removing energy:

* the bin is divided by ``max(1, RMS_bin / (rho E))`` with
  ``E = sqrt((a d^s)^2 + sigma^2)``, so a bin keeps its relative structure
  and its energy stays within ``rho`` of the fitted profile;
* each coefficient is capped at ``rho * kappa * E``, where the crest factor
  ``kappa`` is the largest max/RMS ratio of any fitted measured bin.

Each sample resolves to one of ``RESOLUTIONS``. ``fitted`` is the fit above.
A sample whose fit cannot claim a decay gets the weakest bound its measurement
supports instead of a refusal, because both shapes are ordinary slices at low
field: a small object's spectrum is flat over the measured bins (its width goes
as one over its size), and a slice whose every measured bin sits under twice the
noise power has no signal to fit.

* ``flat``: one distinct fitted distance, or a slope shallower than
  ``SLOPE_RANGE[1]``. Slope 0 at the loudest fitted bin, so no decay is assumed.
* ``noise_floor``: no fitted bin. ``S = 0`` and ``E = sigma``.

A slope steeper than ``SLOPE_RANGE[0]`` still raises: that bound would damp true
detail, and noise flattens a measured spectrum rather than steepening it.
``batch_pooled`` shares one slope across the fitted samples
(``sum cov_b / sum var_b``) and keeps a per-sample amplitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from spectramr.models.diffusion.reverse_decay_axis import DecayAxis, build_decay_axis, row_index
from spectramr.models.diffusion.reverse_envelope import (
    _physical,
    divide_physical_magnitude,
    per_coefficient_support,
)

if TYPE_CHECKING:
    from torch import Tensor

    from spectramr.infrastructure.physics.data_consistency import NoiseAdaptiveDataConsistency

#: Admissible fitted slope of the bin magnitude against its distance from the centre.
SLOPE_RANGE: tuple[float, float] = (-3.0, -0.25)

#: A bin enters the fit only when its power exceeds this multiple of the noise power.
_FIT_SNR = 2.0

#: How a sample's envelope was resolved; ``DecayEnvelope.resolution`` indexes this.
RESOLUTIONS: tuple[str, ...] = ("fitted", "flat", "noise_floor")


@dataclass(frozen=True)
class DecayEnvelope:
    """Fitted profile, per sample: ``[B]`` tensors in physical units, over ``axis``."""

    slope: Tensor
    log_amp: Tensor
    sigma2: Tensor
    crest: Tensor
    #: ``[B, nbins]`` bool: the measured bins the fit used.
    fitted_bins: Tensor
    axis: DecayAxis
    #: ``[B]`` index into ``RESOLUTIONS``.
    resolution: Tensor

    def unfitted(self) -> Tensor:
        """``[B]`` bool: samples bounded by a ``flat`` or ``noise_floor`` envelope."""
        return self.resolution != RESOLUTIONS.index("fitted")

    def bin_rms(self) -> Tensor:
        """``E(d)`` per bin, ``[B, nbins]``; bins at distance ``< 1`` are unbounded (``inf``)."""
        d = self.axis.coord
        signal = torch.exp(
            self.log_amp[:, None] + self.slope[:, None] * torch.log(d.clamp_min(1.0))
        )
        rms = torch.sqrt(signal**2 + self.sigma2[:, None])
        return torch.where(d >= 1, rms, torch.full_like(rms, float("inf")))


def measured_power(
    measurement: Tensor, obs: Tensor, log_scaled: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Physical modulus ``[B, C', H, W]``, coil-mean power ``[B, 1, H, W]`` and measured support."""
    from spectramr.models.diffusion.kspace_process import paired_magnitude

    phys = _physical(paired_magnitude(measurement), log_scaled)
    support = (per_coefficient_support(obs, measurement) > 0).any(dim=-3, keepdim=True)
    return phys, (phys * phys).mean(dim=-3, keepdim=True), support


def _regress(x: Tensor, y: Tensor, w: Tensor, pooled: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted least squares per sample: ``(slope, intercept, var)``; NaN-free on empty rows."""
    n = w.sum(dim=1)
    n_safe = n.clamp_min(1.0)
    mx = (w * x).sum(dim=1) / n_safe
    my = (w * y).sum(dim=1) / n_safe
    cov = (w * (x - mx[:, None]) * (y - my[:, None])).sum(dim=1)
    var = (w * (x - mx[:, None]) ** 2).sum(dim=1)
    if pooled:
        pooled_var = var.sum()
        common = cov.sum() / pooled_var.clamp_min(1e-12)
        slope = torch.where(pooled_var > 0, common, torch.zeros_like(common)).expand_as(mx)
        var = pooled_var.expand_as(mx)
    else:
        slope = torch.where(var > 0, cov / var.clamp_min(1e-12), torch.zeros_like(cov))
    return slope, my - slope * mx, var


def fit_decay_envelope(
    measurement: Tensor,
    obs: Tensor,
    *,
    log_scaled: bool,
    pooled: bool,
    axis: str,
    noise: NoiseAdaptiveDataConsistency,
    eps: float = 1e-8,
) -> DecayEnvelope:
    """Fit the per-sample decay of ``measurement`` over the bins of ``axis`` on its observed support."""
    phys, power, support = measured_power(measurement, obs, log_scaled)
    bins = build_decay_axis(axis, *measurement.shape[-2:], measurement.device)
    sigma2 = noise.estimate_noise_variance(power, support.float()).flatten()
    count = bins.reduce_sum(support).float()
    bin_power = bins.reduce_sum(power * support) / count.clamp_min(1.0)
    d = bins.coord
    measured = (count > 0) & (d >= 1)
    usable = measured & (bin_power > _FIT_SNR * sigma2[:, None])
    log_s = 0.5 * torch.log((bin_power - sigma2[:, None]).clamp_min(eps))
    log_d = torch.log(d.clamp_min(1.0)).expand_as(log_s)
    slope, log_amp, var = _regress(log_d, log_s, usable.float(), pooled)
    lo, hi = SLOPE_RANGE
    has_bin = usable.any(dim=1)
    steep = has_bin & (var > 0) & (slope < lo)
    if bool(steep.any()):
        raise ValueError(
            f"reverse_decay_envelope: sample(s) {steep.nonzero().flatten().tolist()} fit slope(s) "
            f"{slope[steep].tolist()} steeper than {lo}; that bound would damp true detail."
        )
    flat = has_bin & ((var <= 0) | (slope > hi))
    loudest = torch.where(usable, log_s, torch.full_like(log_s, -torch.inf)).amax(dim=1)
    # S = sqrt(eps) * sigma: zero against the noise at any data scale, and finite.
    floor = 0.5 * (math.log(eps) + torch.log(sigma2))
    slope = torch.where(flat | ~has_bin, torch.zeros_like(slope), slope)
    log_amp = torch.where(flat, loudest, torch.where(has_bin, log_amp, floor))
    resolution = torch.full_like(slope, RESOLUTIONS.index("fitted"), dtype=torch.long)
    resolution[flat] = RESOLUTIONS.index("flat")
    resolution[~has_bin] = RESOLUTIONS.index("noise_floor")
    peak = bins.reduce_max(phys * support)
    # A noise-floor sample has no fitted bin, so its crest is read off the measured
    # noise bins; a crest of 0 would cap every unmeasured coefficient at 0.
    crest_bins = torch.where(has_bin[:, None], usable, measured)
    crest = torch.where(crest_bins, peak / bin_power.clamp_min(eps).sqrt(), torch.zeros_like(peak))
    crest = crest.amax(dim=1)
    if bool((crest <= 0).any()):
        raise ValueError(
            "reverse_decay_envelope: a sample measures no bin at distance >= 1, so neither a "
            "decay nor a noise floor can be read from it."
        )
    return DecayEnvelope(
        slope=slope,
        log_amp=log_amp,
        sigma2=sigma2,
        crest=crest,
        fitted_bins=usable,
        axis=bins,
        resolution=resolution,
    )


def unmeasured_decay_bins(obs: Tensor, x: Tensor, axis: DecayAxis) -> Tensor:
    """``[B, 1, H, W]`` bool: coefficients no channel measured, in bins at distance ``>= 1``."""
    measured = (per_coefficient_support(obs, x) > 0).any(dim=-3, keepdim=True)
    return ~measured & (axis.coord_map() >= 1)


def decay_ceiling(envelope: DecayEnvelope, ratio: float, *, log_scaled: bool) -> Tensor:
    """Per-bin magnitude cap ``rho * kappa * E``, broadcastable to ``[B, 1, H, W]``, in the sampler's domain."""
    cap = ratio * envelope.crest[:, None] * envelope.bin_rms()
    cap = torch.log1p(cap) if log_scaled else cap
    return envelope.axis.spread(cap)


def tighten_ceiling(
    ceil: Tensor, envelope: DecayEnvelope, obs: Tensor, x: Tensor, *, ratio: float, log_scaled: bool
) -> Tensor:
    """``ceil`` lowered to the decay cap on unmeasured coefficients at distance ``>= 1`` only.

    Measured coefficients keep ``ceil`` exactly, so a loop's existing clamp never
    moves an acquired line further through this change.
    """
    cap = decay_ceiling(envelope, ratio, log_scaled=log_scaled)
    target = unmeasured_decay_bins(obs, x, envelope.axis)
    ceil = ceil.expand(x.shape[0], 1, *x.shape[-2:])
    return torch.where(target, torch.minimum(ceil, cap), ceil)


def bound_bins_by_decay(
    x0: Tensor,
    envelope: DecayEnvelope,
    obs: Tensor,
    *,
    ratio: float,
    log_scaled: bool,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Divide each bin's unmeasured coefficients by ``max(1, RMS_bin / (rho E))``; return them and the ``[B, nbins]`` divisors."""
    from spectramr.models.diffusion.kspace_process import paired_magnitude

    axis = envelope.axis
    target = unmeasured_decay_bins(obs, x0, axis)
    phys = _physical(paired_magnitude(x0), log_scaled)
    weight = target.expand_as(phys).to(phys.dtype)
    count = axis.reduce_sum(weight)
    rms = torch.sqrt(axis.reduce_sum(phys * phys * weight) / count.clamp_min(1.0))
    divisor = (rms / (ratio * envelope.bin_rms())).clamp_min(1.0)
    divisor = torch.where(count > 0, divisor, torch.ones_like(divisor))
    out = divide_physical_magnitude(
        x0, axis.spread(divisor), target, log_scaled=log_scaled, eps=eps
    )
    return out, divisor


__all__ = [
    "RESOLUTIONS",
    "SLOPE_RANGE",
    "DecayEnvelope",
    "bound_bins_by_decay",
    "decay_ceiling",
    "fit_decay_envelope",
    "measured_power",
    "row_index",
    "tighten_ceiling",
    "unmeasured_decay_bins",
]
