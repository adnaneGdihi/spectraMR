"""K-Space Undersampling Diffusion Process.

Physics-informed degradation operator that models MRI acquisition
as deterministic undersampling rather than Gaussian noise.

This implements the forward process q(x_t | x_0) = x_0 * M_t where M_t
is a sampling mask with acceleration factor R(t) that increases with t.

Reference: Bansal et al., "Cold Diffusion: Inverting Arbitrary Image Transforms"
           Applied to MRI: Degradation is undersampling, not noise.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.data_consistency import VALID_DC_METHODS
from spectramr.infrastructure.physics.line_axis import line_axis_for
from spectramr.models.diffusion.reverse_additive_normalized import sample_additive_normalized
from spectramr.models.diffusion.reverse_hooks import (
    bound_estimate,
    prepare_reverse,
    refine_estimate,
)
from spectramr.models.diffusion.reverse_integrity import (
    check_coverage,
    check_step,
    validate_integrity_check,
)
from spectramr.models.diffusion.reverse_physics import resolve_reverse_physics
from spectramr.models.diffusion.reverse_refine import (
    sample_reveal_refine,
    validate_refine_alpha,
)
from spectramr.models.diffusion.reveal_plan import plan_reveals
from spectramr.models.diffusion.reverse_single_call import sample_single_call
from spectramr.models.diffusion.reverse_support_ordered import sample_support_ordered

#: Reverse-process variants for the cold-diffusion k-space sampler.
#:
#:   additive             Bansal Algorithm 2, ``x_{s-1} = x_s - D(x0, s) + D(x0, s-1)``.
#:                        With ``D(x, s) = M_s * x`` and nested masks this reduces to
#:                        ``x_s + x0 * (M_{s-1} - M_s)``: the correction term that makes
#:                        Algorithm 2 robust for blur or noise cancels for a projection.
#:                        Kept as the reproduction baseline; no DC, no magnitude bound.
#:   additive_normalized  the additive structure with the state re-pinned by DC, each
#:                        step's writes divided by the measured band-energy envelope and
#:                        clamped, and every coefficient the mean of its writes
#:                        (``reverse_additive_normalized.py``).
#:   replace_freeze       write each revealed coefficient once, then freeze; observed
#:                        support hard-pinned; magnitude-bounded.
#:   replace_freeze_dc    as above, but the observed support follows ``dc_method``.
#:   replace_freeze_dc_t0 as above, but the terminal call (t=0) rewrites every
#:                        unobserved bin from its own estimate instead of freezing them.
#:   reveal_refine_dc     reveal on the same schedule but keep refining revealed
#:                        coefficients (``reverse_refine.py``).
#:   single_call_dc       one call at the measurement's own rung, then data
#:                        consistency (``reverse_single_call.py``).
#:   support_ordered_dc   commit unmeasured lines nearest the measured ones first and
#:                        rewrite the rest every step (``reverse_support_ordered.py``).
VALID_REVERSE_MODES: frozenset[str] = frozenset(
    {
        "additive",
        "additive_normalized",
        "replace_freeze",
        "replace_freeze_dc",
        "replace_freeze_dc_t0",
        "reveal_refine_dc",
        "single_call_dc",
        "support_ordered_dc",
    }
)

#: Reverse modes whose reveal at step ``i`` keys off ``mask(schedule[i])`` -- the
#: rung the model is CALLED at -- rather than ``mask(schedule[i+1])``.
#:
#: Every mode reveals at step ``i`` the rung of ``schedule[i+1]``, so the call at
#: ``schedule[i]`` sees exactly ``M(schedule[i])`` (input support = label). The
#: terminal call at t=0 then sees the full plane; under ``replace_freeze_dc_t0``
#: it is the decisive write, which is why it may overwrite committed bins.
#: The call count is one more than the earlier t0 keying (the head call now runs).
TERMINAL_OVERWRITE_REVERSE_MODES: frozenset[str] = frozenset({"replace_freeze_dc_t0"})

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor


from spectramr.infrastructure.training.utils.kspace_masks import (
    create_kspace_mask_generator,
)

#: How a predicted coefficient's magnitude ceiling is referenced.
#:   global_max — ``ratio * max|measured|`` over the whole tensor (legacy).
#:   band_local — ``ratio * max|measured|`` within the coefficient's own radial
#:                frequency band, with a monotone non-increasing envelope filling
#:                bands the measurement never sampled.
VALID_CLIP_REFERENCES: frozenset[str] = frozenset({"global_max", "band_local"})

#: C6 determinism contract: how the reverse loop picks which lines to reveal.
#: Only ``fixed`` exists — reveals come from the process' fixed-seed mask cascade,
#: making the sampler a deterministic function of (measurement, mask, sampler_seed).
#: The knob is declared (and validated) so a future stochastic rule must be opted
#: into explicitly rather than slipping in as an unvalidated kwarg.
VALID_SELECTION_RULES: frozenset[str] = frozenset({"fixed"})


def validate_sampler_determinism(sampler_sigma: float, selection_rule: str) -> None:
    """Fail-loud validation of the C6 sampler knobs, shared by the two samplers.

    Used by both ``PhysicsInformedColdDiffusion.__init__`` and
    ``ColdDiffusionInferenceStrategy.__init__`` so the same YAML keys are
    rejected identically at construction on either path (pitfall #9/#15).
    """
    sigma = float(sampler_sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sampler_sigma must be finite and >= 0, got {sampler_sigma!r}.")
    if selection_rule not in VALID_SELECTION_RULES:
        raise ValueError(
            f"Unknown selection_rule {selection_rule!r} for the cold-diffusion "
            f"sampler. Valid: {sorted(VALID_SELECTION_RULES)}."
        )


#: Reverse steps when neither ``training.diffusion.sampling_steps`` nor a ``model_kwargs``
#: copy declares a count, capped at the horizon (#566, #1176).
DEFAULT_SAMPLING_STEPS = 50


def resolve_sampling_steps(declared: int | None, num_timesteps: int) -> int:
    """The reverse step count, shared by the generator, the sampler and predict.

    An undeclared count takes :data:`DEFAULT_SAMPLING_STEPS` capped at the horizon and says
    so at WARNING; an explicit count below 1 raises rather than falling back (#1176).
    """
    if declared is None:
        steps = min(DEFAULT_SAMPLING_STEPS, int(num_timesteps))
        logger.warning(
            "No sampling_steps declared: the reverse loop runs %d steps over a %d-step "
            "horizon. Declare training.diffusion.sampling_steps to choose the budget.",
            steps,
            num_timesteps,
        )
        return steps
    if isinstance(declared, bool) or int(declared) != declared or int(declared) < 1:
        raise ValueError(f"sampling_steps must be an integer >= 1, got {declared!r}.")
    return int(declared)


def inject_reverse_step_noise(
    x0: Tensor,
    sigma: float,
    generator: torch.Generator,
    exclude_support: Tensor | None = None,
) -> Tensor:
    """Add σ-scaled Gaussian reverse-step noise from a dedicated generator (C6).

    One shared implementation for every reverse loop (the three
    ``PhysicsInformedColdDiffusion`` modes and the inference strategy's own
    loop) so σ means the same thing on every path. The draw happens on the CPU
    generator's stream and is then moved to ``x0``'s device, so the noise
    sequence is identical across devices; global RNG state is never touched.

    ``exclude_support`` (the observed / data-consistent support) zeroes the
    noise there, so σ>0 can never violate data consistency on measured lines.
    Callers gate on ``sigma > 0`` — at σ=0 no generator exists and no draw
    happens, so the deterministic path is bit-for-bit unchanged.
    """
    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device="cpu").to(x0.device)
    if exclude_support is not None:
        noise = noise * (1.0 - exclude_support)
    return x0 + sigma * noise


#: Training-time ("hot") forward noise schedules: name -> f(t) in [0, 1] with
#: sigma_t = forward_noise_sigma * f(t). Both vanish at t = 0, so the clean level stays clean.
FORWARD_NOISE_SCHEDULES: dict[str, Any] = {
    "linear_t": lambda t, horizon: t / horizon,
    "sqrt_t": lambda t, horizon: torch.sqrt(t / horizon),
}


def validate_forward_noise(sigma: float, schedule: str) -> tuple[float, str]:
    """Fail-loud validation of ``model.model_kwargs.forward_noise_*`` (non-negotiable 3)."""
    value = float(sigma)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"forward_noise_sigma must be finite and >= 0, got {sigma!r}.")
    if schedule not in FORWARD_NOISE_SCHEDULES:
        raise ValueError(
            f"Unknown forward_noise_schedule {schedule!r}. "
            f"Valid: {sorted(FORWARD_NOISE_SCHEDULES)}."
        )
    return value, schedule


def validate_forward_noise_exclusive(forward_sigma: float, sampler_sigma: float) -> None:
    """Refuse both reverse-side noise knobs at once: one owner of the reverse-step noise."""
    if float(forward_sigma) > 0 and float(sampler_sigma) > 0:
        raise ValueError(
            "forward_noise_sigma > 0 and sampler_sigma > 0 both inject noise into the reverse "
            "trajectory; declare one. forward_noise_sigma re-noises the unmeasured bins the "
            "denoiser sees, sampler_sigma perturbs the reverse step after data consistency."
        )


def forward_noise_levels(
    sigma: float, schedule: str, t: Tensor, num_timesteps: int
) -> Tensor:
    """Per-sample noise level ``sigma_t`` (relative), ``[B]``, for timesteps ``t``."""
    horizon = float(max(int(num_timesteps) - 1, 1))
    fraction = FORWARD_NOISE_SCHEDULES[schedule](t.to(torch.float32), horizon)
    return float(sigma) * fraction


def hot_noise_increment(x: Tensor, measured: Tensor, level: Tensor, eps: Tensor) -> Tensor:
    """``level * RMS(x over measured bins) * eps`` on the unmeasured bins, zero elsewhere.

    The scale is the RMS of the measured coefficients of ``x`` per sample, detached, so
    sigma is relative to the data and therefore comparable across contrasts. Real-stacked
    ``x`` holds (Re, Im) channel pairs: its per-entry power is doubled so the complex power
    matches, and each real entry gets ``1/sqrt(2)`` of the complex std.
    """
    with torch.no_grad():
        xd = x.detach()
        power = xd.abs() ** 2 if torch.is_complex(xd) else 2.0 * xd * xd
        known = measured.detach().expand_as(power).to(power.dtype)
        rms = ((power * known).flatten(1).sum(1) / known.flatten(1).sum(1).clamp_min(1.0)).sqrt()
        factor = 1.0 if torch.is_complex(x) else 1.0 / math.sqrt(2.0)
        # One real-valued coefficient map, so the full-size complex tensor is touched once.
        coef = (level.to(rms) * rms).view(-1, 1, 1, 1) * factor * (1.0 - known.to(rms.dtype))
    return eps * coef.to(eps.real.dtype)


def paired_magnitude(x: Tensor, eps: float = 1e-8) -> Tensor:
    """TRUE complex modulus of k-space held as interleaved real/imag channels.

    Real-stacked complex k-space in this codebase is INTERLEAVED —
    ``[Re0, Im0, Re1, Im1, ...]`` — which is the layout the generator itself
    parses with ``x[:, 0::2]`` / ``x[:, 1::2]`` before every ``fft2c``
    (``kspace_cold_diffusion_generator`` lines ~1532 and ~3129, the latter only
    ~75 lines above the training-path clamp this helper now feeds).

    ``Tensor.abs()`` on that layout is the ELEMENTWISE absolute value, i.e.
    ``max(|Re|, |Im|)``-ish per channel rather than ``sqrt(Re^2 + Im^2)`` per
    coefficient. It therefore UNDER-READS the true modulus by up to sqrt(2) (at
    ``|Re| == |Im|``) and by 1.07x on real M4Raw data even away from that
    diagonal — which is why a clamp built on it admits ``sqrt(2) * ratio``
    instead of ``ratio`` and rotates phase toward the diagonal (issue #1281).

    Args:
        x: ``[..., C, H, W]``. Complex dtype is returned as ``x.abs()``
            unchanged (already the true modulus). Real dtype is read as
            ``C // 2`` interleaved pairs.
        eps: unused; accepted so callers can pass their local epsilon
            uniformly alongside the clamp helper below.

    Returns:
        ``[..., C // 2, H, W]`` for real input (one modulus per complex
        coefficient), ``[..., C, H, W]`` for complex input.

    Raises:
        ValueError: odd channel count on a real tensor — the interleaved
            reading is then undefined. Mirrors the existing loud refusal at
            ``kspace_cold_diffusion_generator`` line ~1518 (CLAUDE.md #9)
            rather than silently degrading to the elementwise reading.
    """
    if torch.is_complex(x):
        return x.abs()
    return torch.sqrt(paired_squared_magnitude(x))


def paired_squared_magnitude(x: Tensor) -> Tensor:
    """``|z|^2`` per complex coefficient, on the layout :func:`paired_magnitude` reads.

    The squared modulus is smooth at zero and the modulus is not: the gradient of
    ``sqrt`` there is infinite. A term that needs a floored modulus on a
    gradient-carrying tensor therefore floors THIS and takes the root afterwards.

    Args:
        x: ``[..., C, H, W]``, complex or interleaved real-stacked.

    Returns:
        ``[..., C // 2, H, W]`` for real input, ``[..., C, H, W]`` for complex.

    Raises:
        ValueError: fewer than 3 dims, or an odd channel count on a real tensor.
    """
    if torch.is_complex(x):
        return x.real * x.real + x.imag * x.imag
    if x.dim() < 3:
        raise ValueError(f"paired_magnitude expects at least [C, H, W], got {tuple(x.shape)}.")
    c = x.shape[-3]
    if c % 2 != 0:
        raise ValueError(
            "paired_magnitude: interleaved Re/Im layout requires an even channel "
            f"count but got {tuple(x.shape)} (C={c}). A real-stacked complex "
            "k-space tensor always has C = 2 * n_coils."
        )
    re = x[..., 0::2, :, :]
    im = x[..., 1::2, :, :]
    return re * re + im * im


def paired_complex(x: Tensor) -> Tensor:
    """Complex view of k-space held as interleaved real/imag channels.

    Same ``[Re0, Im0, Re1, Im1, ...]`` convention and the same refusal on an odd
    channel count as :func:`paired_magnitude`, which is the reference for both;
    ``paired_complex(x).abs()`` and ``paired_magnitude(x)`` agree by
    construction and a unit test pins that. Kept separate rather than folded
    into one helper because the magnitude path runs inside the reverse loop and
    must not allocate a complex tensor it would immediately discard.

    Phase-sensitive diagnostics need the argument, not just the modulus, so they
    cannot be served by ``paired_magnitude``.

    Args:
        x: ``[..., C, H, W]``. Complex dtype is returned unchanged; real dtype
            is read as ``C // 2`` interleaved pairs.

    Returns:
        ``[..., C // 2, H, W]`` complex for real input, ``x`` for complex input.

    Raises:
        ValueError: fewer than 3 dims, or an odd channel count on a real tensor.
    """
    if torch.is_complex(x):
        return x
    if x.dim() < 3:
        raise ValueError(f"paired_complex expects at least [C, H, W], got {tuple(x.shape)}.")
    c = x.shape[-3]
    if c % 2 != 0:
        raise ValueError(
            "paired_complex: interleaved Re/Im layout requires an even channel "
            f"count but got {tuple(x.shape)} (C={c}). A real-stacked complex "
            "k-space tensor always has C = 2 * n_coils."
        )
    return torch.complex(x[..., 0::2, :, :], x[..., 1::2, :, :])


def clamp_to_magnitude_ceiling(x: Tensor, ceiling: Tensor, eps: float = 1e-8) -> Tensor:
    """Genuinely phase-preserving magnitude clamp (radial, not box).

    Scales each complex coefficient by ONE shared factor derived from its true
    modulus, so the constraint region is the disc ``|z| <= ceiling`` and the
    argument of ``z`` is exactly invariant.

    The elementwise predecessor — ``x * (ceil / x.abs()).clamp(max=1)`` — bounds
    ``|Re|`` and ``|Im|`` INDEPENDENTLY. That is a square of half-width
    ``ceiling``, whose corner sits at ``sqrt(2) * ceiling``, and because the two
    components are scaled by different factors it MOVES the phase (measured:
    5.71 deg -> 26.57 deg). Both effects are corrected here.

    Args:
        x: ``[..., C, H, W]``, complex or interleaved real-stacked.
        ceiling: broadcastable against the modulus. Both production ceilings are
            channel-broadcast (``[B, 1, H, W]`` from
            :func:`band_local_magnitude_ceiling`, ``[B, 1, 1, 1]`` from
            ``_magnitude_ceiling``'s ``global_max`` branch), so no new ceiling
            shape is required by this change. A ``+inf`` entry means unbounded:
            that coefficient passes through with scale 1 and no gradient from the
            clamp, which is how :func:`measured_write_ceiling` exempts hard-DC bins.
        eps: modulus floor, guarding 0/0 on unwritten coefficients.

    Returns:
        Same shape and dtype as ``x``.
    """
    # Floor the SQUARED modulus, then take the root. Flooring the modulus after
    # `sqrt` guards the forward divide but not the backward: at an exact zero
    # (hard DC copies the readout window's zeros into the output) `clamp_min`
    # returns 0 into sqrt's infinite gradient, 0 * inf = NaN, and every weight
    # goes NaN at the first optimizer step.
    modulus = torch.sqrt(paired_squared_magnitude(x).clamp_min(eps * eps))
    # An infinite ceiling must never reach the divide: its backward is 0 * inf = NaN
    # even where the clamp does not bind, so it is swapped out before and after.
    bounded = ~torch.isposinf(ceiling)
    ratio = torch.where(bounded, ceiling, torch.ones_like(ceiling)) / modulus
    scale = torch.where(bounded, ratio.clamp(max=1.0), torch.ones_like(ratio))
    if torch.is_complex(x):
        return x * scale
    # One scale per (Re, Im) pair -> expand back over the interleaved axis so
    # both components of a coefficient move together (this is what preserves
    # the phase).
    return x * scale.repeat_interleave(2, dim=-3)


def ceiling_binding_stats(x: Tensor, ceiling: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """``(fraction, mean overshoot)`` of the bounded coefficients ``clamp_to_magnitude_ceiling`` binds.

    Above the ceiling the clamp passes no magnitude gradient, so a post-DC loss
    cannot pull an overshoot down there; the fraction says how much of the plane
    is in that dead zone. ``+inf`` entries are unbounded and excluded. The
    overshoot is the mean of ``|x| / ceiling`` over the bound coefficients, 1.0
    when none bind. Both are 0-dim device tensors, computed without a gradient
    or a host sync, so they can ride the training metrics.
    """
    with torch.no_grad():
        modulus = torch.sqrt(paired_squared_magnitude(x).clamp_min(eps * eps))
        bounded = (~torch.isposinf(ceiling)).expand_as(modulus)
        safe_ceiling = torch.where(bounded, ceiling.expand_as(modulus), torch.ones_like(modulus))
        ratio = modulus / safe_ceiling.clamp_min(eps)
        binds = bounded & (ratio > 1.0)
        n_bounded = bounded.sum().clamp_min(1)
        n_binds = binds.sum()
        fraction = n_binds / n_bounded
        overshoot = torch.where(
            n_binds > 0, (ratio * binds).sum() / n_binds.clamp_min(1), torch.ones_like(fraction)
        )
    return fraction.float(), overshoot.float()


def measured_write_ceiling(
    ceil: Tensor, measurement: Tensor, measured: Tensor, dc_method: str
) -> Tensor:
    """The ceiling a clamp applied after data consistency may enforce, per coefficient.

    ``ceil`` off the measured support. On it, no bound under hard DC, where the bin already
    equals ``y``, and ``max(ceil, |y|)`` under any blend, so no clamp rewrites an acquired
    coefficient: ``band_local``'s running-minimum envelope sits below an outer band's
    measured maximum, and a whole-plane clamp cut those bins (#2310).

    Args:
        ceil: the magnitude ceiling, broadcastable against the modulus.
        measurement: ``[B, 2C, H, W]`` interleaved (or complex) measured k-space.
        measured: boolean support, broadcastable against the modulus (``[B, 1, H, W]``).
        dc_method: the ``dc_method`` that wrote the measured bins before the clamp.
    """
    magnitude = paired_magnitude(measurement)
    own = (
        torch.full_like(magnitude, float("inf"))
        if dc_method == "hard"
        else torch.maximum(ceil, magnitude)
    )
    return torch.where(measured, own, ceil)


def apply_ceiling_ratio(reference: Tensor, ratio: float, log_scaled: bool) -> Tensor:
    """Apply a magnitude ``ratio`` to a reference IN PHYSICAL UNITS.

    ``reverse_clip_ratio: 1.3`` is a statement about physical k-space magnitude:
    "no written coefficient may exceed 1.3x its reference". When
    ``data.processing.enable_log_scaling`` is on, the sampler runs on
    ``log1p(|k| / scale)``, and multiplying a COMPRESSED value by 1.3 is not a
    1.3x physical bound — it is ``expm1(1.3 * u) / expm1(u)``, which grows
    exponentially in the dynamic range ``u``. Measured on this arm's real M4Raw
    data (``u_max = 4.03``): a declared 1.3 realised **29.8x** (issue #1281).

    The correct compressed-domain ceiling is the compression of the physical
    ceiling, which is what this returns::

        physical:   ceil = ratio * ref
        log1p:      ceil = log1p(ratio * expm1(ref))

    Args:
        reference: reference magnitude, in whatever domain the sampler holds.
        ratio: the declared physical multiplier.
        log_scaled: whether ``reference`` is ``log1p``-compressed. Required —
            it cannot be inferred from the data, and defaulting it would let a
            log-scaled arm silently keep the exponential bound (CLAUDE.md #3).

    Returns:
        Ceiling in the same domain as ``reference``.
    """
    if not log_scaled:
        return ratio * reference
    return torch.log1p(ratio * torch.expm1(reference))


@functools.lru_cache(maxsize=32)
def _radial_band_partition(
    h: int, w: int, num_bands: int, eps: float, device: torch.device
) -> Tensor:
    """``[H, W]`` radial band index — a function of shape alone.

    Depends on nothing per-call: not the measurement's values, not the batch,
    not which bins are observed. A training run and a validation trajectory
    both call ``band_local_magnitude_ceiling`` with the SAME ``(h, w,
    num_bands, device)`` on every step, so this is the compile-time constant
    to cache rather than rebuild on the accelerated-run hot path
    (non-negotiable 9, issue #536's follow-on). ``lru_cache`` needs every
    argument hashable, which is why this takes the plain scalars rather than
    the ``measurement`` tensor.
    """
    yy = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2.0
    xx = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0
    radius = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    radius = radius / radius.max().clamp_min(eps)
    return (radius * num_bands).long().clamp_(0, num_bands - 1)  # [H, W]


def band_local_magnitude_ceiling(
    measurement: Tensor,
    ratio: float,
    num_bands: int = 16,
    eps: float = 1e-8,
    *,
    log_scaled: bool,
) -> Tensor:
    """Per-coefficient magnitude ceiling referenced to its own frequency band.

    ``ratio * max|measured|`` over the WHOLE tensor is not a bound in any useful
    sense: ``max|measured|`` is the k-space DC peak. Measured on the exp_11 run's
    step-1 snapshot, ``abs_max/std = 3.52/0.095 = 37``, so a ratio of 1.3 permits
    every one of the ~88-92% unobserved coefficients to be written at ~48 sigma —
    filling that much of k-space at 48x the RMS coefficient produces exactly the
    coherent high-amplitude banding the arm's validation images show (issue #536).

    This references each coefficient to the largest measured magnitude in its own
    radial band instead. k-space magnitude falls steeply with radius, so a band-
    local ceiling is tight where it matters (the periphery) while still permitting
    the true dynamic range at the centre. Bands with no measured sample inherit the
    nearest inner band's ceiling, giving a monotone non-increasing envelope — the
    physically correct prior for an unobserved high-frequency band.

    Args:
        measurement: measured k-space ``[B, C, H, W]`` (complex or real-stacked);
            zeros are treated as unobserved. Real-stacked input is read as
            INTERLEAVED Re/Im pairs via :func:`paired_magnitude`, so the band
            reference is a true complex modulus rather than the elementwise
            ``.abs()`` that under-read it by up to sqrt(2) (issue #1281).
        ratio: multiplier on the band reference, in PHYSICAL units.
        num_bands: radial bands across the half-diagonal.
        eps: magnitude floor.
        log_scaled: whether ``measurement`` is ``log1p``-compressed
            (``data.processing.enable_log_scaling``). Required and keyword-only:
            the ratio is applied through :func:`apply_ceiling_ratio`, and getting
            this wrong is the difference between a 1.3x and a 29.8x realised
            bound on this arm's data.

    Returns:
        Ceiling broadcastable against ``measurement`` (``[B, 1, H, W]``).
    """
    if measurement.dim() < 3:
        raise ValueError(
            f"band_local_magnitude_ceiling expects at least [C, H, W], got "
            f"{tuple(measurement.shape)}."
        )
    # TRUE complex modulus per coefficient, not the elementwise |Re|,|Im|.
    # Channels collapse to C//2 here; the per-band reduction below is over the
    # channel axis anyway, so the returned [B, 1, H, W] shape is unchanged.
    mag = paired_magnitude(measurement)
    h, w = mag.shape[-2], mag.shape[-1]
    # Radial index from the k-space centre (matching fft2c's centred convention);
    # cached on shape alone, see ``_radial_band_partition``.
    band = _radial_band_partition(h, w, num_bands, eps, mag.device)

    # Per-sample, per-band max over the OBSERVED support (nonzero measurement).
    # One ``scatter_reduce`` over the whole partition rather than a per-band
    # Python loop: a per-band ``bool(...)`` test or boolean-mask gather is a
    # device->host synchronization on the training hot path (non-negotiable
    # 9) for a reduction whose SHAPE never depends on this call's data (issue
    # #536 follow-on). A band with no observed entry keeps ``ceilings``' zero
    # initialisation (``include_self=True``), so an empty band needs no
    # emptiness test at all.
    per_pixel = mag.amax(dim=1, keepdim=True) if mag.dim() >= 4 else mag.unsqueeze(1)
    flat = per_pixel.flatten(2)  # [B, 1, H*W]
    band_flat = band.flatten()  # [H*W]
    batch = flat.shape[0]
    observed_vals = flat * (flat > eps)
    index = band_flat.view(1, 1, -1).expand(batch, 1, -1)
    ceilings = flat.new_zeros((batch, 1, num_bands)).scatter_reduce(
        dim=-1, index=index, src=observed_vals, reduce="amax", include_self=True
    )

    # Monotone non-increasing envelope: an unsampled band inherits the nearest
    # inner band's ceiling rather than collapsing to zero (which would forbid ANY
    # prediction there and hard low-pass the output).
    running = ceilings[..., 0].clone()
    for b in range(num_bands):
        current = ceilings[..., b]
        running = torch.where(current > eps, torch.minimum(running, current), running)
        ceilings[..., b] = running

    ceiling_map = ceilings[..., band_flat].reshape(batch, 1, h, w)
    return apply_ceiling_ratio(ceiling_map, ratio, log_scaled).clamp_min(eps)


def _as_mapping(source: Any) -> dict[str, Any]:
    """Normalise an acceleration config to a plain mapping.

    ``model_builder`` hands the generator a live ``AccelerationConfigSchema``
    while ``ModelFactory`` hands it a ``model_dump()`` of the same object, and
    tests/scripts pass raw dicts. All three must read identically; a pydantic
    model has no ``.get``, so without this the object path is an AttributeError
    waiting on whichever call site skips the factory.
    """
    if source is None:
        return {}
    if hasattr(source, "model_dump"):
        return dict(source.model_dump())
    if isinstance(source, dict):
        return dict(source)
    return dict(getattr(source, "__dict__", {}) or {})


def _first_declared(*values: Any, default: Any = None) -> Any:
    """Return the first value that is not ``None``, else ``default``.

    Presence, not truthiness: ``0`` and ``0.0`` are legal seeds and fractions,
    and an ``or`` chain silently replaced them with the fallback (issue #690).
    """
    for value in values:
        if value is not None:
            return value
    return default


def _plain(value: Any) -> Any:
    """Unwrap a ``str``-mixin enum to its value.

    ``AccelerationSchedule`` is a ``(str, Enum)``, so ``str(AccelerationSchedule
    .STEP)`` is ``'AccelerationSchedule.STEP'``, not ``'step'``. Every consumer
    downstream compares against the lowercase value.
    """
    return getattr(value, "value", value)


def resolve_undersampling_kwargs(
    acceleration_config: Any = None,
    overrides: dict[str, Any] | None = None,
    *,
    phase_encode_axis: int | None = None,
) -> dict[str, Any]:
    """Resolve ``KSpaceUndersamplingProcess`` kwargs from an acceleration config.

    SSOT for "what undersampling process does this config actually build?".
    ``KSpaceColdDiffusionGenerator`` and
    ``scripts/ci/check_acceleration_ladder_realisable.py`` both call this, so the
    gate can no longer certify a ladder the runtime does not build. Before this
    existed the two re-derived the same kwargs independently with *different*
    defaults (``center_fraction`` 0.0325 vs the schema's 0.08,
    ``base_acceleration`` 1.0 vs 4.0), and the gate read raw YAML, so it never
    saw keys the schema drops (issue #550).

    Precedence is ``acceleration_config`` > ``overrides`` (the generator's
    ``**kwargs``) > default, and only ``None`` counts as unset: a declared
    ``mask_seed: 0`` or ``center_fraction: 0.0`` is honoured (issue #690).

    Args:
        acceleration_config: ``AccelerationConfigSchema``, its ``model_dump()``,
            or a raw mapping. ``None`` is treated as empty.
        overrides: Secondary source, typically the generator's ``model_kwargs``.
        phase_encode_axis: ``data.acquisition.phase_encode_axis`` for callers
            that hold the data block; the generator passes it in ``overrides``.

    Raises:
        ValueError: ``mask_direction`` is ``phase``/``readout`` and no axis is
            declared, or the axis is declared twice with different values.

    Returns:
        Keyword arguments for ``KSpaceUndersamplingProcess`` covering
        ``max_acceleration``, ``base_acceleration``, ``center_fraction``,
        ``min_center_fraction``, ``mask_type``, ``seed``, ``schedule_type``,
        ``schedule_kwargs``, ``enable_dynamic_mask`` and
        ``train_identity_rung``.
    """
    accel = _as_mapping(acceleration_config)
    over = dict(overrides or {})

    nested_schedule_kwargs = accel.get("schedule_kwargs") or over.get("schedule_kwargs")
    schedule_kwargs: dict[str, Any] = dict(nested_schedule_kwargs or {"density_power": 1.6})

    # ``AccelerationConfigSchema.density_power`` (default 2.0, "Power for
    # variable density sampling") is a real, audit-advertised schema field, but
    # the schema carries no ``schedule_kwargs`` mapping at all, so the read
    # above could never see a declaration made through it: every arm silently
    # ran the literal 1.6 above regardless of what it declared (pitfall 15). A
    # schema DUMP always carries a value for ``density_power`` (the default
    # fills in), so truthiness alone cannot tell "declared" from "unset" --
    # ``model_fields_set`` is this repo's own idiom for that distinction
    # (``models/losses/weights.py``, ``config/overrides.py``); a schema default
    # is not a declaration, and the 63 ``density_nested`` cohort arms that never
    # mention this knob must keep computing the same 1.6 they do today.
    declared_fields = getattr(acceleration_config, "model_fields_set", frozenset())
    if "density_power" in declared_fields:
        schema_density_power = accel["density_power"]
        # Look up the nested spelling only in what was ACTUALLY provided --
        # ``schedule_kwargs`` above may hold the ``{"density_power": 1.6}``
        # fallback rather than a real declaration, and comparing against that
        # literal would raise a false NN17 conflict for every arm that never
        # touched the nested spelling at all.
        nested_density_power = (
            nested_schedule_kwargs.get("density_power")
            if nested_schedule_kwargs is not None
            else None
        )
        if nested_density_power is not None and nested_density_power != schema_density_power:
            raise ValueError(
                "undersampling.density_power is declared twice with different "
                f"values: the schema field says {schema_density_power!r}, "
                "model.model_kwargs.schedule_kwargs.density_power says "
                f"{nested_density_power!r}. Declare it in one place (NN17)."
            )
        schedule_kwargs["density_power"] = schema_density_power

    # The curve exponent of ``power_law``/``polynomial``, forwarded on the same
    # declared-not-defaulted reading as ``density_power``; without it a declared
    # 0.5 is dropped and the ramp runs the accelerator's own 2.0.
    if "schedule_power" in declared_fields:
        schema_schedule_power = accel["schedule_power"]
        nested_schedule_power = (nested_schedule_kwargs or {}).get("schedule_power")
        if nested_schedule_power is not None and nested_schedule_power != schema_schedule_power:
            raise ValueError(
                "undersampling.schedule_power is declared twice with different "
                f"values: the schema field says {schema_schedule_power!r}, "
                "model.model_kwargs.schedule_kwargs.schedule_power says "
                f"{nested_schedule_power!r}. Declare it in one place (NN17)."
            )
        schedule_kwargs["schedule_power"] = schema_schedule_power

    # Without this the step schedule falls back to [1.0, max_acceleration], so
    # R(0)=1 is fully sampled and t=0 degenerates to the identity.
    accel_range = accel.get("acceleration_range")
    if accel_range is not None and "acceleration_range" not in schedule_kwargs:
        schedule_kwargs["acceleration_range"] = list(accel_range)

    resolved: dict[str, Any] = {
        "max_acceleration": _first_declared(
            accel.get("max_acceleration"), over.get("max_acceleration"), default=32.0
        ),
        "base_acceleration": _first_declared(
            accel.get("base_acceleration"), over.get("base_acceleration"), default=1.0
        ),
        "center_fraction": _first_declared(
            accel.get("center_fraction"), over.get("center_fraction"), default=0.0325
        ),
        "min_center_fraction": _first_declared(
            accel.get("min_center_fraction"), over.get("min_center_fraction")
        ),
        "mask_type": _first_declared(
            _plain(accel.get("acceleration_type")),
            over.get("acceleration_type"),
            default="variable_density",
        ),
        "seed": _first_declared(accel.get("mask_seed"), over.get("seed"), default=42),
        "schedule_type": _first_declared(
            _plain(accel.get("schedule_type")), over.get("schedule_type"), default="power_law"
        ),
        "schedule_kwargs": schedule_kwargs,
        "enable_dynamic_mask": bool(
            accel.get("enable_dynamic_mask", over.get("enable_dynamic_mask", False))
        ),
        "train_identity_rung": bool(
            accel.get("train_identity_rung", over.get("train_identity_rung", False))
        ),
        # Nesting enforcement and the mask axis both used to stop here: the
        # accelerator knows how to honour them, but this resolver never forwarded
        # them, so `mask_direction` was inert (issue #948) and enforcement was
        # unreachable from YAML. Forwarded explicitly rather than by **kwargs, so
        # a typo in the block still fails at the schema.
        "enforce_nested": bool(accel.get("enforce_nested", over.get("enforce_nested", False))),
        "nested_tolerance": float(
            _first_declared(
                accel.get("nested_tolerance"), over.get("nested_tolerance"), default=0.5
            )
        ),
    }
    # Only forward the axis when one was actually declared. Passing None would be
    # mapped onto ``line_axis=None`` downstream, which the line-based
    # accelerators reject; absent means "keep the accelerator's own default".
    direction = _first_declared(_plain(accel.get("mask_direction")), over.get("mask_direction"))
    if direction is not None:
        resolved["mask_direction"] = direction
    # The dataset's phase-encode axis (``data.acquisition``) arrives from the
    # generator's kwargs or from a caller holding the data block; two sources
    # that disagree are one fact stated twice.
    over_axis = over.get("phase_encode_axis")
    if phase_encode_axis is not None and over_axis is not None and over_axis != phase_encode_axis:
        raise ValueError(
            f"phase_encode_axis is declared twice: {phase_encode_axis!r} from "
            f"data.acquisition and {over_axis!r} from model_kwargs. Declare it once."
        )
    axis = _first_declared(phase_encode_axis, over_axis)
    if axis is not None:
        resolved["phase_encode_axis"] = axis
    if direction is not None:
        # Resolved here only to raise at build: an absent axis under a
        # ``phase``/``readout`` mask names the measurement script (#2160).
        line_axis_for(direction, axis)
    return resolved


def build_accelerator_kwargs(
    *,
    max_acceleration: float,
    base_acceleration: float,
    center_fraction: float,
    min_center_fraction: float | None = None,
    seed: int | None = 42,
    schedule_type: str = "linear",
    schedule_kwargs: dict[str, Any] | None = None,
    enforce_nested: bool = False,
    nested_tolerance: float = 0.5,
    mask_direction: str | None = None,
    phase_encode_axis: int | None = None,
) -> dict[str, Any]:
    """Map resolved undersampling settings onto ``create_kspace_accelerator`` kwargs.

    SSOT for the *second* half of the translation. ``resolve_undersampling_kwargs``
    answers "what undersampling settings does this config declare?"; this answers
    "what does an accelerator constructor call them?". The two vocabularies are
    genuinely different — ``mask_seed``/``seed``, ``schedule_type``/
    ``acceleration_schedule``, ``mask_type`` naming the accelerator rather than
    being a kwarg of it, ``schedule_kwargs`` flattened rather than nested — and
    every call site that re-derived the mapping by hand got some of it wrong.

    An **allowlist**, deliberately, not a filter over ``model_dump()``. A dump
    emits every schema field including unset defaults, so a denylist silently
    grows a new junk kwarg each time the schema gains a field; that is how
    ``mixins/kspace.py`` came to forward seventeen names the accelerator does not
    read, and why it never translated ``mask_seed`` (issue #1059: ``seed=None``
    → global RNG → a cascade that is no longer nested, which cold diffusion's
    forward process assumes). Filtering against
    ``sampling._accelerator_kwarg_vocabulary()`` would not fix that — it drops
    ``mask_seed`` too, silently, which is pitfall #9 rather than a repair.

    Args:
        max_acceleration: Acceleration at ``t=T-1``, the last timestep.
        base_acceleration: Acceleration at ``t=0``.
        center_fraction: ACS fraction at low acceleration.
        min_center_fraction: ACS fraction at max acceleration, forwarded as
            declared (issue #534: an unforwarded value made every rung above
            R≈12 realise the same mask). ``None`` stays ``None``: every family
            already reads it as "the core is ``center_fraction``", and
            substituting that value here made the accelerators' build-time
            floor check treat an undeclared floor as a declared one (#2516).
        seed: Accelerator seed. Already the accelerator spelling — callers
            holding YAML's ``mask_seed`` must translate first, which
            ``resolve_undersampling_kwargs`` does.
        schedule_type: Acceleration schedule; forwarded as
            ``acceleration_schedule``.
        schedule_kwargs: Schedule parameters, flattened into the result.
        enforce_nested: Coerce ``M_{t+1} ⊆ M_t``.
        nested_tolerance: Minimum share of the family's own raw draw that the
            enforced mask must retain. NOT a share of ``1 / declared_R`` --
            see ``ColdDiffusionAccelerator`` for why that denominator was
            wrong.
        mask_direction: ``phase``/``readout``; omitted entirely when ``None``
            because the compat mapping rejects a non-axis.
        phase_encode_axis: ``data.acquisition.phase_encode_axis``, which
            ``phase``/``readout`` resolve through; omitted when ``None``.

    Returns:
        Keyword arguments accepted by ``create_kspace_accelerator``.
    """
    return {
        "max_acceleration": max_acceleration,
        "base_acceleration": base_acceleration,
        "center_fraction": center_fraction,
        "min_center_fraction": min_center_fraction,
        "seed": seed,
        "enforce_nested": enforce_nested,
        "nested_tolerance": nested_tolerance,
        **({} if mask_direction is None else {"mask_direction": mask_direction}),
        **({} if phase_encode_axis is None else {"phase_encode_axis": phase_encode_axis}),
        "acceleration_schedule": schedule_type,
        **(schedule_kwargs or {}),
    }


def accelerator_kwargs_from_config(
    acceleration_config: Any = None,
    overrides: dict[str, Any] | None = None,
    *,
    phase_encode_axis: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve an acceleration config straight to ``(pattern, accelerator_kwargs)``.

    Convenience composition of ``resolve_undersampling_kwargs`` and
    ``build_accelerator_kwargs`` for the call sites that construct a
    ``KSpaceMaskGenerator`` directly instead of going through
    ``KSpaceUndersamplingProcess``.

    ``mask_type`` becomes the generator's ``default_pattern`` (it names *which*
    accelerator, so it is not one of its kwargs), and ``enable_dynamic_mask`` /
    ``train_identity_rung`` are consumed here rather than forwarded: per-sample
    seed jitter is implemented by ``KSpaceUndersamplingProcess.q_sample`` and is
    training-only by design, and the timestep floor is a property of the
    diffusion process, not of a mask pattern — an accelerator has nothing to do
    with either. Dropping them is a translation, not a silent discard: the
    forwarded set below is enumerated explicitly, so the accelerator gate would
    reject an unknown kwarg rather than absorb it.

    Args:
        acceleration_config: ``AccelerationConfigSchema``, its ``model_dump()``,
            or a raw mapping.
        overrides: Secondary source, as in ``resolve_undersampling_kwargs``.
        phase_encode_axis: ``data.acquisition.phase_encode_axis``.

    Returns:
        ``(default_pattern, accelerator_kwargs)``.
    """
    resolved = resolve_undersampling_kwargs(
        acceleration_config, overrides, phase_encode_axis=phase_encode_axis
    )
    pattern = resolved["mask_type"]
    return pattern, build_accelerator_kwargs(
        max_acceleration=resolved["max_acceleration"],
        base_acceleration=resolved["base_acceleration"],
        center_fraction=resolved["center_fraction"],
        min_center_fraction=resolved["min_center_fraction"],
        seed=resolved["seed"],
        schedule_type=resolved["schedule_type"],
        schedule_kwargs=resolved["schedule_kwargs"],
        enforce_nested=resolved["enforce_nested"],
        nested_tolerance=resolved["nested_tolerance"],
        mask_direction=resolved.get("mask_direction"),
        phase_encode_axis=resolved.get("phase_encode_axis"),
    )


class KSpaceUndersamplingProcess(nn.Module):
    """Deterministic undersampling process for k-space cold diffusion.

    Delegates to KSpaceMaskGenerator to ensure SSOT for sampling patterns.
    Guarantees Nested Property: M_{t+1} subset of M_t.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = 8.0,
        base_acceleration: float = 1.0,
        center_fraction: float = 0.0325,
        min_center_fraction: float | None = None,
        mask_type: str = "variable_density",
        seed: int | None = 42,
        schedule_type: str = "linear",
        schedule_kwargs: dict | None = None,
        enable_dynamic_mask: bool = False,
        train_identity_rung: bool = False,
        enforce_nested: bool = False,
        nested_tolerance: float = 0.5,
        mask_direction: str | None = None,
        phase_encode_axis: int | None = None,
        device: torch.device | str | None = None,
        forward_noise_sigma: float = 0.0,
        forward_noise_schedule: str = "linear_t",
    ) -> None:
        """Initialize k-space undersampling process.

        Args:
            num_timesteps: Number of diffusion timesteps.
            max_acceleration: Maximum acceleration factor, reached at t=T-1.
            base_acceleration: Minimum acceleration factor, at t=0.
            center_fraction: ACS fraction at low acceleration (R≈1).
            min_center_fraction: ACS fraction at max acceleration.
                If None, uses center_fraction (static behavior).
            mask_type: Sampling pattern type.
            seed: Random seed for reproducibility.
            schedule_type: Acceleration schedule type.
            schedule_kwargs: Additional schedule parameters.
            enforce_nested: Coerce the cascade so ``M_{t+1}`` is a subset of
                ``M_t`` at every timestep, which is what cold diffusion's
                forward process assumes. Applies to the fixed-seed cascade only;
                see ``enable_dynamic_mask`` below, which deliberately varies the
                pattern per sample and is left unenforced.
            nested_tolerance: Minimum share of the family's OWN raw draw at each
                timestep that the enforced mask must retain before construction
                raises. The denominator is the raw draw rather than the
                continuous ``1 / declared_R`` -- line quantisation means the two
                can never coincide, which used to make ``1.0`` unsatisfiable for
                every family. Declared-R drift is :meth:`declared_ladder_defects`.
            mask_direction: ``phase`` or ``readout``; mapped onto the
                accelerator's ``line_axis`` through ``phase_encode_axis``.
            phase_encode_axis: ``data.acquisition.phase_encode_axis`` (-1 or
                -2). Required by a ``phase``/``readout`` direction; there is
                no default (#2160).
            enable_dynamic_mask: When True, ``q_sample`` draws a fresh
                accelerator seed *per sample* (during training only), so the
                model sees many distinct undersampling PATTERNS per acceleration
                level instead of one fixed pattern. The acceleration FRACTION is
                unchanged — only which lines are kept varies. ``False`` (default)
                keeps the single seeded pattern, preserving legacy behaviour and
                reproducible validation. Wires the previously-dead
                ``acceleration.enable_dynamic_mask`` config knob (pitfall #15).
            train_identity_rung: When True, :meth:`min_meaningful_timestep`
                returns 0 even at ``R(0) == 1``, so the fully-sampled rung is
                trained instead of excluded. No effect when ``R(0) > 1`` — the
                floor is already 0 there. See that method for why the default
                excludes it and when overriding is correct.
            device: Device the mask generator serves masks from, inherited from
                the run configuration by ``resolve_generator_kwargs`` (step 3d).
                Gates the device-resident mask table: left ``None`` the
                generator is CPU-pinned and every ``q_sample`` pays a host sync
                (#1508). Not resolved here -- pass an already-resolved device
                (non-negotiable 9b); ``None`` means "CPU, as before".
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        # Training-time noise on the unmeasured bins of x_t (see ``q_sample``); 0 is off.
        self.forward_noise_sigma, self.forward_noise_schedule = validate_forward_noise(
            forward_noise_sigma, forward_noise_schedule
        )
        self.max_accel = max_acceleration
        self.base_acceleration = base_acceleration
        self.center_fraction = center_fraction
        self.min_center_fraction = (
            min_center_fraction if min_center_fraction is not None else center_fraction
        )
        self.mask_type = mask_type
        self.seed = seed
        self.enable_dynamic_mask = enable_dynamic_mask
        self.train_identity_rung = train_identity_rung
        self.enforce_nested = enforce_nested
        self.nested_tolerance = nested_tolerance
        self.mask_direction = mask_direction
        self.phase_encode_axis = phase_encode_axis
        # First-batch line-axis verdict per split, read by the snapshot
        # provenance; see ``check_line_axis``.
        self.line_axis_gate: dict[str, dict[str, Any]] = {}
        # Plane shape -> the fixed line ranking the acquired-line budget reads,
        # and the shapes whose ACS core was checked against a batch's budget.
        self._line_rankings: dict[tuple[int, int], Any] = {}
        self._core_checked: set[tuple[int, int]] = set()

        # Use SSOT Mask Generator
        # We pass max_acceleration and base_acceleration into the generator's
        # default config so the accelerator uses the correct range [base, max].
        # ``min_center_fraction`` was stored on this process and never forwarded,
        # so the accelerator fell back to a STATIC ``center_fraction``. A static
        # 8% ACS is the entire sampling budget at R=12.5, so every nominally-
        # higher rung realised the same ~8%-of-k-space mask: the exp_11 ladder
        # declared [2,4,8,10,12,16,32] and topped out at an effective 12.2x,
        # with R=16 and R=32 identical (issue #534). The mapping now lives in
        # ``build_accelerator_kwargs`` so the strategy-side and builder-side
        # generators cannot drift away from it again.
        # The mask device is INHERITED from the run configuration, never sniffed
        # from a tensor at call time. ``KSpaceMaskGenerator`` serves masks from a
        # device-resident ``[T, 1, H, W]`` table only when BOTH its own device and
        # the incoming ``timesteps`` are non-CPU; constructed device-less this
        # process pinned the generator to CPU, so every ``q_sample`` on the
        # training path fell back to the ``timesteps.to("cpu").tolist()`` host
        # sync the table exists to remove (#1508). ``None`` keeps the historical
        # CPU behaviour -- resolution is the caller's job (non-negotiable 9b), and
        # resolving here would raise on every legitimately CPU-only construction
        # (the witness schedule check, unit tests, CI wiring).
        self.mask_device = None if device is None else torch.device(device)
        self.mask_generator = create_kspace_mask_generator(
            num_timesteps=num_timesteps,
            default_pattern=mask_type,
            device=self.mask_device,
            accelerator_kwargs=build_accelerator_kwargs(
                max_acceleration=max_acceleration,
                base_acceleration=base_acceleration,
                center_fraction=center_fraction,
                min_center_fraction=min_center_fraction,
                seed=seed,
                schedule_type=schedule_type,
                schedule_kwargs=schedule_kwargs,
                enforce_nested=enforce_nested,
                nested_tolerance=nested_tolerance,
                mask_direction=mask_direction,
                phase_encode_axis=phase_encode_axis,
            ),
        )

    def min_meaningful_timestep(self) -> int:
        """Lowest timestep that is a real degradation, not the identity.

        SSOT for the training/sampling timestep floor (issue #535). The training
        sampler and the reverse schedule MUST agree on it: the reverse loop
        evaluates its final, decisive step at this timestep, and if training never
        draws it the model's time embedding is extrapolating exactly where it
        matters most.

        Returns 0 when ``R(0) > 1`` (``base_acceleration > 1``, so t=0 still
        undersamples and is worth training), else 1 — at ``R(0) == 1`` the input is
        the fully-sampled target and the task degenerates to the identity, which is
        why t=0 was excluded from training in the first place.

        ``train_identity_rung`` overrides that exclusion. "Degenerates to the
        identity" is a statement about the DEGRADATION (``q_sample`` at t=0 is
        bit-exact ``x_start``), not about the SUPERVISION: an arm whose loss
        target differs from its model input — NEX phase-aligned averaging,
        denoising, cross-contrast translation — has a real fully-sampled task at
        t=0, and excluding it means the reverse schedule's terminal, decisive
        step lands on a timestep the time embedding never saw. Opt in per arm;
        the default stays 1 so the exclusion remains the behaviour for arms
        where t=0 genuinely is a no-op.
        """
        if self.train_identity_rung:
            return 0
        accelerator = self.mask_generator._get_accelerator(self.mask_type)
        return 0 if float(accelerator.get_acceleration_factor(0)) > 1.0 + 1e-6 else 1

    def describe_ladder(self, image_shape: tuple[int, int]) -> list[tuple[int, float, float, int]]:
        """Realised acceleration per timestep for a given matrix size.

        Returns ``(t, R_nominal, R_effective, bins_kept)`` per timestep, where
        ``R_effective = total_bins / bins_kept`` — acceleration is one over the
        SAMPLED FRACTION, which is the only definition valid for every mask
        geometry (a line count is meaningful for equispaced/1-D masks but not for
        variable-density or radial patterns, where nearly every row carries some
        sample while the sampled fraction is still 1/R).

        Nominal and realised diverge whenever the always-sampled ACS band consumes
        the budget: at ``center_fraction=0.08`` on a 256 matrix the centre alone is
        8% of k-space, which is the entire budget at R=12.2, so every
        nominally-higher rung realises ~12x.

        Goes straight to the mask generator's fixed-seed path rather than through
        ``q_sample``, which is also the ladder validation reverse-samples (the
        per-sample randomisation of ``enable_dynamic_mask`` applies to training
        only).
        """
        accelerator = self.mask_generator._get_accelerator(self.mask_type)
        h, w = image_shape
        out: list[tuple[int, float, float, int]] = []
        for t in range(self.num_timesteps):
            r_nominal = float(accelerator.get_acceleration_factor(t))
            mask = self.mask_generator.generate_batch_masks(
                batch_size=1,
                timesteps=torch.tensor([t]),
                image_shape=(h, w),
                pattern=self.mask_type,
            )
            m = mask[0, 0].float()
            kept = max(int(m.sum()), 1)
            out.append((t, r_nominal, m.numel() / kept, kept))
        return out

    def declared_ladder_defects(
        self, image_shape: tuple[int, int], tolerance: float = 0.25
    ) -> list[str]:
        """Defects in an explicitly-declared discrete acceleration ladder.

        Scoped deliberately to ``schedule_type: step`` with an explicit
        ``acceleration_range``. That is the only configuration where "each rung is
        a distinct, realised acceleration" is a well-defined claim — and it is the
        configuration that failed silently in issue #534.

        A *continuous* schedule (linear / power_law) over T timesteps and a finite
        line count must produce duplicate line counts by pigeonhole (T=1000
        timesteps cannot map to 256 distinct masks), so duplicates there are not a
        defect and are not reported.

        Two defects are reported for a discrete ladder:

        * a declared rung whose realised ``R`` differs by more than ``tolerance``
          — the ``val_*_{R}x`` metric column then measures a different
          acceleration than its label claims (nominal 32x realised 12.2x),
        * two declared rungs realising the SAME line count — the ladder has fewer
          rungs than it advertises and a stretch of the timestep axis is
          degenerate, so a timestep-conditioned net spends gradient distinguishing
          physically identical inputs.

        Returns an empty list when the ladder is sound or the check does not apply.
        """
        accelerator = self.mask_generator._get_accelerator(self.mask_type).accelerator
        if str(getattr(accelerator, "acceleration_schedule", "")) != "step":
            return []
        if not getattr(accelerator, "_acceleration_range_explicit", False):
            return []
        declared = [float(r) for r in getattr(accelerator, "acceleration_range", [])]
        if len(declared) < 2:
            return []

        ladder = self.describe_ladder(image_shape)
        # One representative timestep per declared rung.
        per_rung: dict[float, tuple[int, float, int]] = {}
        for t, nom, eff, kept in ladder:
            if nom in declared and nom not in per_rung:
                per_rung[nom] = (t, eff, kept)

        defects: list[str] = []

        # A rung realised at NO timestep never enters ``per_rung``, so both checks
        # below would skip it and the gate would report "defects: none" for a
        # ladder that silently drops rungs (issue #1171). The forward index is
        # ``min(int(t/(T-1) * K), K - 1)``, which takes at most ``num_timesteps``
        # distinct values — so declaring MORE rungs than timesteps guarantees some
        # are unreachable. This is the config-time counterpart of the raise in
        # ``KSpaceAccelerator.timestep_for_acceleration``: same declared-vs-realised
        # confusion, caught before launch instead of mid-validation.
        unreachable = [nom for nom in declared if nom not in per_rung]
        if unreachable:
            defects.append(
                f"declared rung(s) R={sorted(unreachable)} are realised at NO "
                f"timestep: acceleration_range has {len(declared)} entries but "
                f"the schedule has only {self.num_timesteps} timesteps to index "
                f"them with, so the step index skips entries. Declare at most one "
                f"rung per timestep."
            )
        for nom in sorted(per_rung):
            t, eff, _ = per_rung[nom]
            if nom > 1.0 and abs(eff - nom) / nom > tolerance:
                defects.append(
                    f"declared rung R={nom:g} realises R={eff:.2f} at t={t} "
                    f"(off by {abs(eff - nom) / nom:.0%} > {tolerance:.0%})"
                )
        by_lines: dict[int, list[float]] = {}
        for nom, (_, _, kept) in per_rung.items():
            by_lines.setdefault(kept, []).append(nom)
        for lines, rungs in sorted(by_lines.items()):
            if len(rungs) > 1:
                defects.append(
                    f"declared rungs R={sorted(rungs)} all realise the same "
                    f"{lines}-bin mask ({lines} of {image_shape[0] * image_shape[1]} bins)"
                )
        return defects

    def assert_ladder_realisable(
        self, image_shape: tuple[int, int], tolerance: float = 0.25
    ) -> None:
        """Raise when an explicitly-declared acceleration rung is unrealisable.

        See :meth:`declared_ladder_defects` for the scope and rationale.

        NOT called from ``q_sample``: 62 of the 66 discrete-ladder arms in the
        corpus are defective today (issue #534 and its follow-up), so raising
        mid-run would take them all offline at once. The corpus is instead
        ratcheted pre-launch by
        ``scripts/ci/check_acceleration_ladder_realisable.py``, which carries a
        baseline of the known-defective arms so new ones cannot be added. Call
        this directly from a test or a probe when you want the hard assertion.
        """
        parts = self.declared_ladder_defects(image_shape, tolerance=tolerance)
        if not parts:
            return
        raise ValueError(
            "Acceleration ladder is not realisable at "
            f"{image_shape[0]}x{image_shape[1]}: "
            + " | ".join(parts)
            + f". The always-sampled ACS band (center_fraction="
            f"{self.center_fraction:g}, min_center_fraction="
            f"{self.min_center_fraction:g}) consumes the line budget. Lower "
            "acceleration.min_center_fraction so the ACS shrinks with R, or "
            "truncate acceleration.acceleration_range to the rungs that are "
            "actually reachable (issue #534)."
        )

    def mask_at(self, t: int, image_shape: tuple[int, int]) -> Tensor:
        """The validation-path kept-set at one timestep, as a boolean ``[H, W]``.

        The fixed-seed generator, never ``q_sample`` — the same path
        :meth:`describe_ladder` and :meth:`_cascade_masks` take, so a caller
        reasoning about the reverse trajectory sees the cascade validation
        actually reverse-samples (``enable_dynamic_mask`` randomisation applies
        to training draws only). Single-timestep primitive so a consumer that
        needs a handful of rungs does not rebuild the whole ladder.
        """
        h, w = image_shape
        # `device=` is load-bearing, not tidiness. The memoised cascade in
        # `generate_batch_masks` is gated on BOTH the timestep tensor and the
        # generator being off-CPU (kspace_masks.py: "a host-side table would
        # need the index moved back, reintroducing this very sync"), so a bare
        # `torch.tensor([t])` lands on CPU and misses the table every call --
        # rebuilding the mask instead of indexing one. The attribution path
        # calls this once per scheduled step, per rung, per batch, and the
        # cache's own note records a Scalene profile charging 24.24% of a run
        # to the host copy it exists to avoid.
        mask = self.mask_generator.generate_batch_masks(
            batch_size=1,
            timesteps=torch.tensor([int(t)], device=self.mask_generator.device),
            image_shape=(h, w),
            pattern=self.mask_type,
        )
        return mask[0, 0].bool()

    @property
    def line_axis(self) -> str | None:
        """The built accelerator's ``line_axis`` (``y`` rows, ``x`` columns), or None.

        Read off the family that was constructed rather than re-derived from
        ``mask_direction``, so a family's own default is reported too. None for
        a 2-D point pattern.
        """
        accelerator = self.mask_generator._get_accelerator(self.mask_type)
        return getattr(getattr(accelerator, "accelerator", accelerator), "line_axis", None)

    @property
    def line_dim(self) -> int | None:
        """The plane dimension the mask's lines index (-2 rows, -1 columns), or None."""
        axis = self.line_axis
        return None if axis is None else (-2 if axis == "y" else -1)

    def acquired_lines(self, kspace: Tensor) -> Tensor | None:
        """``[B, N]`` acquired lines of a plane-last batch, or None without a line axis (#2160)."""
        from spectramr.models.diffusion.acquired_line_budget import acquired_lines_of

        dim = self.line_dim
        return None if dim is None else acquired_lines_of(kspace, dim)

    def _line_ranking(self, image_shape: tuple[int, int]) -> Any:
        """The fixed ranking of this plane's full-support cascade, built once per shape."""
        from spectramr.models.diffusion.acquired_line_budget import rank_lines

        key = (int(image_shape[0]), int(image_shape[1]))
        ranking = self._line_rankings.get(key)
        if ranking is None:
            across = -2 if self.line_dim == -1 else -1
            cascade = torch.stack(
                [self.mask_at(t, key).any(dim=across) for t in range(self.num_timesteps)]
            )
            ranking = rank_lines(cascade)
            self._line_rankings[key] = ranking
        return ranking

    def _check_core_fits(self, ranking: Any, acquired: Tensor, key: tuple[int, int]) -> None:
        """Raise once per shape if a sample's ACS core is not acquired or exceeds its top budget.

        One host read on the first batch of a shape. A core larger than the top
        rung's budget would be cut by the budget, and the declared ACS would not
        survive; the fix is a smaller ``undersampling.min_center_fraction``.
        """
        if key in self._core_checked:
            return
        core = ranking.core.to(acquired.device)
        n_acquired = acquired.sum(-1)
        top = int(ranking.kept[-1])
        budget = torch.round(top * n_acquired / ranking.num_lines)
        n_core = int(core.sum())
        outside = bool((core[None] & ~acquired).any())
        too_wide = bool((budget < n_core).any())
        if outside or too_wide:
            raise ValueError(
                f"The {n_core}-line ACS core of this ladder "
                + ("lies partly on never-acquired lines" if outside else "")
                + (" and " if outside and too_wide else "")
                + (
                    f"exceeds the top rung's budget of {int(budget.min())} acquired lines "
                    f"(R={self.max_accel:g} over {int(n_acquired.min())} acquired lines)"
                    if too_wide
                    else ""
                )
                + ". Lower undersampling.min_center_fraction so the core fits (#2160)."
            )
        self._core_checked.add(key)

    def forward_noise_record(self) -> dict[str, Any]:
        """The declared training-time noise, stamped beside ``mask_scope`` in provenance."""
        return {"sigma": self.forward_noise_sigma, "schedule": self.forward_noise_schedule}

    def mask_scope(self) -> dict[str, Any]:
        """Which masks the nesting flag and the seed govern, as applied (#1240).

        Both act on the fixed-seed cascade that validation, predict and the reverse
        loops read. Under ``enable_dynamic_mask`` each training sample draws with a
        fresh seed and without enforcement, so neither reaches the training masks.
        """
        dynamic = bool(self.enable_dynamic_mask)
        return {
            "training_masks": "per_sample_draw" if dynamic else "fixed_cascade",
            "enforce_nested": {
                "fixed_cascade": bool(self.enforce_nested),
                "training": False if dynamic else bool(self.enforce_nested),
            },
            "mask_seed": {
                "fixed_cascade": self.seed,
                "training": "per_sample_global_rng" if dynamic else self.seed,
            },
        }

    def check_line_axis(self, kspace: Tensor, split: str) -> dict[str, Any]:
        """Gate the mask's line axis against the first ``split`` batch, once per split.

        ``kspace`` is the clean batch with the plane last (``[..., H, W]``). The
        mask is read at the sparsest rung, where its line structure is plainest.
        The verdict is kept in :attr:`line_axis_gate` for the snapshot
        provenance; a later call for the same split returns it without
        touching the data, so the per-step cost is one dict lookup.

        Raises:
            LineAxisMismatchError: the data's never-acquired lines and the
                mask's lines lie on different axes (#2160).
        """
        recorded = self.line_axis_gate.get(split)
        if recorded is not None:
            return recorded
        from spectramr.infrastructure.physics.line_axis_gate import check_line_axis

        shape = (int(kspace.shape[-2]), int(kspace.shape[-1]))
        mask = self.mask_at(self.num_timesteps - 1, shape)
        record = check_line_axis(kspace, mask.to(kspace.device), split=split)
        record["mask_direction"] = self.mask_direction
        record["declared_phase_encode_axis"] = self.phase_encode_axis
        self.line_axis_gate[split] = record
        logger.info("[LineAxisGate] %s", record)
        return record

    def _cascade_masks(self, image_shape: tuple[int, int], *, raw: bool = False):
        """Yield ``(t, kept)`` boolean masks along the fixed-seed cascade.

        Same path as :meth:`describe_ladder` — the fixed-seed generator, never
        ``q_sample`` — so the audits below certify the cascade that validation
        and the ladder gate actually see (``enable_dynamic_mask`` randomisation
        applies to training draws only).

        With ``raw=True`` nesting enforcement is suspended for the duration, so the
        caller sees the FAMILY's intrinsic cascade rather than the enforced one.
        Without it every ``enforce_nested`` arm certifies leak-free by construction
        and the audit cannot tell a family that nests on its own from one that only
        nests because the cumulative intersection deleted the offending bins — a
        distinction that matters, because the intersection pays for nesting in
        sampling budget. Restored in ``finally`` the same way
        ``_generate_batch_masks_dynamic`` restores it.
        """
        h, w = image_shape
        accelerator = self.mask_generator._get_accelerator(self.mask_type)
        original_enforce = getattr(accelerator, "enforce_nested", False)
        try:
            if raw:
                accelerator.enforce_nested = False
            for t in range(self.num_timesteps):
                yield t, self.mask_at(t, (h, w))
        finally:
            accelerator.enforce_nested = original_enforce

    def nesting_leak_report(self, image_shape: tuple[int, int], *, raw: bool = False) -> list[dict]:
        """C1 leak audit: k-space bins re-introduced after being removed.

        Cold diffusion's reverse theory assumes the cocycle
        ``D_t = D_{s->t} o D_s``, which for masking degradation holds **iff**
        the kept-sets are nested: ``K_T ⊆ ... ⊆ K_0``. A *re-introduced* bin —
        absent at some earlier (less degraded) level ``s`` but present again at
        ``t > s`` — is a leak: the level-``t`` state carries content the chain
        already declared unmeasured, per-level data consistency no longer
        composes, and the schedule forces a fabrication floor of its own on top
        of the measurement's.

        Returns one dict per leaking level ``t``:

        * ``t`` — the level at which previously-removed bins reappear,
        * ``reintroduced_bins`` — how many bins at ``t`` were removed at ANY
          earlier level (running-union check, so a leak against a non-adjacent
          level is still caught),
        * ``consecutive_bins`` — the subset violating adjacency
          (``K_t ⊄ K_{t-1}``) directly,
        * ``leak_fraction`` — ``reintroduced_bins`` over the bins kept at ``t``.

        Empty list ⇔ the cascade is leak-free (chain-nested). With
        ``enforce_nested=True`` this is guaranteed by construction; the audit
        is the *witness* that the guarantee actually held at this matrix size.

        Args:
            image_shape: ``(H, W)`` to build the cascade at.
            raw: Suspend nesting enforcement and report the FAMILY's own cascade.
                The default (``False``) answers "is what validation sees nested?";
                ``raw=True`` answers "would it still be nested without the
                intersection?". For the 41 ``enforce_nested`` arms in
                ``kspace_filling`` both answers are currently "yes" and the enforced
                masks are bit-identical to the raw ones — but that is a measured
                property of those families, not something the default report can
                distinguish, which is why the raw leg exists.
        """
        report: list[dict] = []
        removed_so_far: Tensor | None = None
        prev: Tensor | None = None
        for t, kept in self._cascade_masks(image_shape, raw=raw):
            if removed_so_far is None:
                removed_so_far = ~kept
            else:
                reintroduced = kept & removed_so_far
                n = int(reintroduced.sum())
                if n:
                    assert prev is not None
                    report.append(
                        {
                            "t": t,
                            "reintroduced_bins": n,
                            "consecutive_bins": int((kept & ~prev).sum()),
                            "leak_fraction": n / max(int(kept.sum()), 1),
                        }
                    )
                removed_so_far = removed_so_far | ~kept
            prev = kept
        return report

    def inert_step_report(self, image_shape: tuple[int, int]) -> list[int]:
        """Levels whose forward step changes nothing: ``K_t == K_{t-1}``.

        An inert level removes no line (and re-introduces none), so a
        timestep-conditioned network spends capacity distinguishing physically
        identical inputs, and the reverse chain schedules a step with nothing
        to reveal (issue #535's forward-side counterpart; the reverse-side
        predicate is ``reveal_plan.RevealPlan.active``).

        Continuous schedules over many timesteps necessarily contain inert
        runs by pigeonhole (T=1000 into 256 line counts), so treat this as a
        WARNING-grade diagnostic there and as a real defect only for discrete
        ladders — same scoping logic as :meth:`declared_ladder_defects`.
        """
        inert: list[int] = []
        prev: Tensor | None = None
        for t, kept in self._cascade_masks(image_shape):
            if prev is not None and bool((kept == prev).all()):
                inert.append(t)
            prev = kept
        return inert

    def removed_line_energy_stats(
        self,
        image_shape: tuple[int, int],
        batch: Tensor,
        *,
        domain: str = "image",
    ) -> list[dict]:
        """C4 allocation audit: per-level energy of the removed k-space bins.

        The papers' per-level ambiguity bound requires each level's removed
        content (its step budget ``delta_t``) to stay small and to be spread
        across levels — dumping the low-frequency band into one level makes
        that level's fibre radius exceed its budget and the level untrainable.
        The computable proxy certified here: the share of clean-data k-space
        energy carried by each level's removed bins.

        Args:
            image_shape: ``(H, W)`` matrix size to audit at.
            batch: Clean data batch. ``domain="image"`` (default) treats it as
                a spatial-domain batch ``[B, C, H, W]`` and transforms via the
                ``fft_ops`` SSOT; ``domain="kspace"`` uses it directly.
                Anything else raises — no silent reinterpretation.

        Returns one dict per level ``t >= 1``: ``t``, ``bins_removed``
        (``K_{t-1} \\ K_t``), ``energy_fraction`` (of total clean energy) and
        ``share`` (of all removed energy). Shares sum to 1 whenever any energy
        is removed.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c

        if domain == "image":
            img = batch if torch.is_complex(batch) else batch.to(torch.complex64)
            k = fft2c(img)
        elif domain == "kspace":
            k = batch if torch.is_complex(batch) else batch.to(torch.complex64)
        else:
            raise ValueError(f"domain must be 'image' or 'kspace', got {domain!r}")
        if k.shape[-2:] != tuple(image_shape):
            raise ValueError(
                f"batch spatial shape {tuple(k.shape[-2:])} does not match "
                f"image_shape {tuple(image_shape)}; the audit would measure a "
                "different k-space geometry than the cascade generates."
            )
        # Mean energy per bin over batch and channels -> [H, W].
        energy = (k.real**2 + k.imag**2).mean(dim=tuple(range(k.dim() - 2)))
        total = float(energy.sum().clamp_min(1e-12))

        per_level: list[dict] = []
        prev: Tensor | None = None
        for t, kept in self._cascade_masks(image_shape):
            if prev is not None:
                removed = prev & ~kept
                frac = float(energy[removed].sum()) / total
                per_level.append(
                    {
                        "t": t,
                        "bins_removed": int(removed.sum()),
                        "energy_fraction": frac,
                    }
                )
            prev = kept
        removed_total = sum(s["energy_fraction"] for s in per_level)
        for s in per_level:
            s["share"] = s["energy_fraction"] / removed_total if removed_total > 0 else 0.0
        return per_level

    def schedule_certification_report(
        self,
        image_shape: tuple[int, int],
        batch: Tensor | None = None,
        *,
        max_level_share: float = 0.5,
        domain: str = "image",
    ) -> dict:
        """Combined pre-training schedule certification (C1 + inert + C4).

        Bundles :meth:`nesting_leak_report`, :meth:`inert_step_report` and —
        when a clean ``batch`` is supplied — :meth:`removed_line_energy_stats`
        into one report with explicit verdicts. Deliberately separate from
        :meth:`declared_ladder_defects` (whose exact output strings the CI
        ratchet ``scripts/ci/check_acceleration_ladder_realisable.py``
        baselines): this report is additive and may grow fields freely.

        ``allocation.ok`` checks that no single level carries more than
        ``max_level_share`` of all removed energy — the computable C4 proxy
        ("spread the low-frequency lines across levels").
        """
        leaks = self.nesting_leak_report(image_shape)
        inert = self.inert_step_report(image_shape)
        out: dict = {
            "leak_free": not leaks,
            "leaks": leaks,
            "inert_steps": inert,
            "allocation": None,
        }
        if batch is not None:
            per_level = self.removed_line_energy_stats(image_shape, batch, domain=domain)
            top = max(per_level, key=lambda s: s["share"], default=None)
            out["allocation"] = {
                "per_level": per_level,
                "max_level_share": top["share"] if top else 0.0,
                "argmax_level": top["t"] if top else None,
                "threshold": max_level_share,
                "ok": (top["share"] <= max_level_share) if top else True,
            }
        return out

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward diffusion: degrade k-space via nested undersampling.

        Returns:
            x_t: Degraded k-space [B, C, H, W] or [B, C, H, W, D]
            masks: Undersampling masks [B, 1, H, W] or [B, 1, H, W, D]
        """
        device = x_start.device
        is_5d = x_start.dim() == 5
        if is_5d:
            # upstream permutes to [B, D, C, H, W]
            B_orig, D, C, H, W = x_start.shape
            x_start = x_start.contiguous().view(B_orig * D, C, H, W)
            t_expanded = t.repeat_interleave(D)
        else:
            B_orig = x_start.shape[0]
            t_expanded = t

        B, C, H, W = x_start.shape
        # One host read on the first batch of each mode; a dict lookup after.
        self.check_line_axis(x_start, "train" if self.training else "eval")
        acquired = self.acquired_lines(x_start)

        # Generate nested masks from SSOT generator
        # generate_batch_masks returns [B, 1, H, W]
        if self.enable_dynamic_mask and self.training:
            # Per-sample pattern randomisation (training only): each sample gets
            # a fresh accelerator seed → a different pattern at the same R, so the
            # model learns to invert undersampling in general rather than one
            # fixed mask. Validation/eval falls through to the fixed-seed path
            # below, keeping metrics reproducible.
            base_masks = self._generate_batch_masks_dynamic(
                B, t_expanded, (H, W), device, acquired=acquired
            )
        else:
            base_masks = self.cascade_batch_masks(t_expanded, (H, W), device, acquired=acquired)

        # Expand mask to match channels for degradation [B, C, H, W]
        channel_masks = self.mask_generator.expand_mask_to_channels(base_masks, C)

        # Apply degradation: x_t = x_0 ⊙ M_t
        x_t = x_start * channel_masks
        if self.forward_noise_sigma > 0:
            # Soft/hot forward process: noise only on the unmeasured bins, so the measured
            # bins are bit-identical. Off (sigma 0) there is no draw and no branch taken.
            level = forward_noise_levels(
                self.forward_noise_sigma, self.forward_noise_schedule, t_expanded, self.num_timesteps
            )
            eps = torch.randn_like(x_t) if noise is None else noise.reshape(x_t.shape)
            x_t = x_t + hot_noise_increment(x_t, channel_masks, level.to(x_t.device), eps)

        if is_5d:
            # Reshape back to 5D [B_orig, D, C, H, W]
            x_t = x_t.view(B_orig, D, C, H, W)
            base_masks = base_masks.view(B_orig, D, 1, H, W)

        return x_t, base_masks

    def cascade_batch_masks(
        self,
        timesteps: Tensor,
        image_shape: tuple[int, int],
        device: torch.device,
        acquired: Tensor | None = None,
    ) -> Tensor:
        """The fixed-seed cascade at ``timesteps``, ``[B, 1, H, W]``, whatever the module's mode.

        The reverse loops read their masks here and never through :meth:`q_sample`, whose
        train-mode branch draws the dynamic training masks under ``enable_dynamic_mask`` and
        would make a reconstruction depend on the module's mode (#2471).

        ``acquired`` (``[B, N]``, from :meth:`acquired_lines`) counts each rung's R over
        that sample's acquired lines only (#2160); None, or a mask without a line axis,
        keeps the full-support cascade. A full support reproduces it bit for bit.
        """
        if acquired is not None and self.line_dim is not None:
            from spectramr.models.diffusion.acquired_line_budget import (
                budget_lines,
                lines_to_plane,
            )

            key = (int(image_shape[0]), int(image_shape[1]))
            ranking = self._line_ranking(key)
            acquired = acquired.to(device)
            self._check_core_fits(ranking, acquired, key)
            full_support = self.mask_generator.generate_batch_masks(
                batch_size=int(timesteps.shape[0]),
                timesteps=timesteps,
                image_shape=image_shape,
                pattern=self.mask_type,
            ).to(device)
            budget = lines_to_plane(budget_lines(ranking, timesteps, acquired), key, self.line_dim)
            # A sample that acquired every line keeps the cascade itself, so a fully
            # sampled batch is bit-identical to the pre-#2160 masks for any family.
            whole = acquired.all(dim=-1).view(-1, 1, 1, 1)
            return torch.where(whole, full_support, budget.to(full_support.dtype))
        return self.mask_generator.generate_batch_masks(
            batch_size=int(timesteps.shape[0]),
            timesteps=timesteps,
            image_shape=image_shape,
            pattern=self.mask_type,
        ).to(device)

    def _generate_batch_masks_dynamic(
        self,
        batch_size: int,
        timesteps: Tensor,
        image_shape: tuple[int, int],
        device: torch.device,
        acquired: Tensor | None = None,
    ) -> Tensor:
        """Per-sample randomised undersampling masks for dynamic-mask training.

        With ``acquired`` each draw keeps its R over that sample's acquired lines,
        ranked by the same per-sample seed (``acquired_line_budget.dynamic_lines``).

        Draws a fresh accelerator seed for every sample so each one gets a
        *different* pattern at its acceleration level, while the acceleration
        FRACTION (set by ``R(t)``) is left untouched. The seed comes from the
        global RNG (seeded once by ``initialize_accelerator``), so the run stays
        reproducible while patterns vary per sample and per step. The owned
        accelerator's seed is restored afterwards so nothing else observes the
        mutation. Returns ``[batch_size, 1, H, W]`` to match
        ``KSpaceMaskGenerator.generate_batch_masks``.
        """
        accelerator = self.mask_generator._get_accelerator(self.mask_type)
        # ``ColdDiffusionAccelerator`` exposes a read-only ``seed`` property; the
        # *inner* accelerator it wraps (always present — set in its __init__)
        # carries the settable seed. Typed ``Any`` because the concrete inner
        # accelerator (e.g. ``UniformCartesianKSpaceAccelerator``) sets ``seed``
        # in __init__ but the ``KSpaceAccelerator`` base does not declare it.
        inner: Any = accelerator.accelerator
        original_seed = getattr(inner, "seed", None)
        # Nesting enforcement is off inside this loop. The accelerator caches its
        # first-drop map on (shape, device, seed), so a per-sample seed misses every
        # time, rebuilds the cascade and leaks a cache entry (256x256, batch 2:
        # 11 -> 334 ms/step). Enforcement belongs to the fixed-seed cascade; both
        # mutations are restored in the ``finally``.
        original_enforce = getattr(accelerator, "enforce_nested", False)
        # The accelerators take host ints, so the batch's timesteps and partial-support
        # flags cross to the host in ONE transfer here rather than one per sample (#1241).
        # The B seeds come from the CPU global RNG in one draw, the same stream as B
        # single draws.
        seeds = torch.randint(0, 2**31 - 1, (batch_size,)).tolist()
        row = timesteps.detach().reshape(-1)[:batch_size].long()
        if acquired is not None and self.line_dim is not None:
            partial = ~acquired.all(dim=-1).to(row.device)
            steps, flags = torch.stack([row, partial.long()]).cpu().tolist()
        else:
            steps, flags = row.cpu().tolist(), [0] * batch_size
        masks = []
        try:
            accelerator.enforce_nested = False
            for i in range(batch_size):
                inner.seed = seeds[i]
                masks.append(
                    self.mask_generator.generate_acceleration_mask(
                        steps[i], image_shape, pattern=self.mask_type
                    )
                )
                if flags[i]:
                    masks[-1] = self._budget_dynamic_draw(
                        masks[-1], acquired[i], image_shape, seeds[i]
                    )
        finally:
            inner.seed = original_seed
            accelerator.enforce_nested = original_enforce
        return torch.stack(masks, dim=0).to(device)

    def _budget_dynamic_draw(
        self, draw: Tensor, acquired: Tensor, image_shape: tuple[int, int], seed: int
    ) -> Tensor:
        """One per-sample draw ``[1, H, W]`` re-ranked over its acquired lines (#2160)."""
        from spectramr.models.diffusion.acquired_line_budget import dynamic_lines, lines_to_plane

        key = (int(image_shape[0]), int(image_shape[1]))
        ranking = self._line_ranking(key)
        self._check_core_fits(ranking, acquired[None], key)
        across = -2 if self.line_dim == -1 else -1
        drawn = (draw[0] > 0).any(dim=across).to(acquired.device)
        lines = dynamic_lines(drawn, acquired, ranking.core.to(acquired.device), seed)
        return lines_to_plane(lines[None], key, self.line_dim)[0].to(draw.device, draw.dtype)

    def prepare_validation_mask(
        self,
        batch_size: int,
        timesteps: Tensor,
        image_shape: tuple[int, int],
        num_channels: int = 2,
        batch_data: dict | None = None,
    ) -> Tensor:
        """Prepare mask for validation step.

        Args:
            batch_size: Number of samples in batch
            timesteps: Timestep tensor [B,]
            image_shape: (H, W) spatial dimensions
            num_channels: Number of channels to expand mask to
            batch_data: Optional batch data containing pre-computed mask

        Returns:
            Mask tensor [B, C, H, W]
        """
        # Check for pre-computed mask in batch_data
        mask = None
        if batch_data is not None:
            try:
                if hasattr(batch_data, "__getitem__"):
                    if "acceleration_mask" in batch_data:
                        mask = batch_data["acceleration_mask"]
                    elif "mask" in batch_data:
                        mask = batch_data["mask"]
            except (KeyError, TypeError) as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        # Generate masks if not provided
        if mask is None:
            device = timesteps.device
            B, H, W = batch_size, image_shape[0], image_shape[1]
            # Ensure timesteps is an appropriately sized tensor for the batch
            if not isinstance(timesteps, torch.Tensor):
                timesteps = torch.full((B,), timesteps, device=device, dtype=torch.long)
            elif timesteps.dim() == 0:
                timesteps = timesteps.expand(B)

            # Use unified generator
            mask = self.mask_generator.generate_batch_masks(
                batch_size=B,
                timesteps=timesteps,
                image_shape=(H, W),
                pattern=self.mask_type,
            ).to(device)

        # Expand to match channels if needed
        if mask.shape[1] != num_channels:
            mask = self.mask_generator.expand_mask_to_channels(mask, num_channels)

        return mask

    def prepare_generation_kwargs(
        self,
        lr_batch: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Prepare keyword arguments for cold diffusion generation.

        Args:
            lr_batch: Low-resolution/measured k-space [B, C, H, W]
            mask: Undersampling mask [B, C, H, W]

        Returns:
            Dict with 'mask' and 'kspace_measured' keys
        """
        return {
            "mask": mask,
            "kspace_measured": lr_batch,
        }

    def apply_data_consistency(
        self,
        prediction: Tensor,
        measurement: Tensor,
        mask: Tensor,
        method: str = "hard",
        lambda_weight: float = 1.0,
    ) -> Tensor:
        """Apply data consistency to prediction.

        Args:
            prediction: Model prediction in k-space [B, C, H, W]
            measurement: Measured k-space data [B, C, H, W]
            mask: Sampling mask [B, C, H, W]
            method: "hard" or "soft" data consistency
            lambda_weight: Soft-DC proximal weight λ (ignored for "hard").
                λ→∞ recovers hard replacement; λ=0 is a pure passthrough.

        Returns:
            Data-consistent prediction [B, C, H, W]

        Raises:
            ValueError: On an unknown ``method`` (silent fallback to a default
                is forbidden — pitfall #9).
        """
        if method == "hard":
            # Replace measured frequencies exactly.
            return prediction * (1 - mask) + measurement * mask
        if method == "soft":
            # Closed-form soft-DC proximal blend, SSOT-consistent with
            # infrastructure/physics/data_consistency.SoftDataConsistency:
            #   k_out = (k_pred + λ·m·y) / (1 + λ·m)
            # The legacy body here ignored ``lambda_weight`` and was
            # byte-identical to the "hard" branch — a silent no-op knob
            # (pitfall #15). This is the genuine weighted blend.
            return (prediction + lambda_weight * mask * measurement) / (1.0 + lambda_weight * mask)
        raise ValueError(f"Unknown data-consistency method {method!r}; expected 'hard' or 'soft'.")

    def get_acceleration_schedule(self) -> Tensor:
        """Return the acceleration factor schedule R(t) for all timesteps.

        Returns:
            schedule: [T] tensor of acceleration factors
        """
        t = torch.arange(self.num_timesteps, dtype=torch.float32)
        t_norm = t / max(self.num_timesteps - 1, 1)
        return 1.0 + (self.max_accel - 1.0) * t_norm

    def get_sampling_fraction_schedule(self) -> Tensor:
        """Return the fraction of k-space sampled at each timestep.

        Returns:
            schedule: [T] tensor of sampling fractions (1/R)
        """
        R = self.get_acceleration_schedule()
        return 1.0 / R


class PhysicsInformedColdDiffusion(nn.Module):
    """Cold Diffusion with physics-informed undersampling degradation.

    Combines KSpaceUndersamplingProcess with data consistency enforcement
    for MRI reconstruction.
    """

    #: Lines each sample acquired, set per :meth:`sample` call (#2160).
    measured_lines: Tensor | None = None

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int,
        max_acceleration: float = 8.0,
        center_fraction: float = 0.0325,
        dc_method: str = "hard",
        dc_weight: float = 1.0,
        sampling_steps: int | None = None,
        reverse_mode: str = "additive",
        reverse_clip_ratio: float = 4.0,
        clip_reference: str = "global_max",
        sampler_sigma: float = 0.0,
        sampler_seed: int | None = None,
        selection_rule: str = "fixed",
        refine_alpha: float | None = None,
        *,
        kspace_log_scaled: bool,
        integrity_check: str = "off",
        decay_envelope: str | None = None,
        projection: str | None = None,
        dc_trust: str | None = None,
        support_radius: int | None = None,
        reverse_dc: str | None = None,
    ) -> None:
        """__init__.

        Args:
            model (nn.Module): Description.
            kspace_log_scaled: whether the k-space this sampler is handed is
                ``log1p``-compressed (``data.processing.enable_log_scaling``).
                Required and keyword-only ON PURPOSE: ``reverse_clip_ratio`` is
                a PHYSICAL multiplier, so the ceiling must be built with
                :func:`apply_ceiling_ratio`, and a default here would silently
                restore the exponential bound that made a declared 1.3 realise
                29.8x on real M4Raw data (issue #1281). It cannot be inferred
                from the tensor -- compressed and physical k-space are the same
                dtype and shape -- so it is declared, not guessed.
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            center_fraction (float): Description.
            dc_method (str): Description.
            dc_weight (float): Description.
            sampling_steps (int | None): Description.
            reverse_mode: reverse-process variant. ``"additive"`` is the legacy
                Bansal-Alg-2 accumulate loop (kept for baselines/repro).
                ``"replace_freeze"`` is the corrected monotone infill: write each
                revealed coefficient once from the (hard-DC'd, magnitude-bounded)
                x0 estimate then freeze it, revealing from the NEXT SCHEDULED
                level (not ``t-1``). Fixes the inference-only magnitude blow-up
                that ``additive`` exhibits under strided schedules with an
                unbounded head (see docs/experiment_11_kspace_cold_diffusion.md).
            reverse_clip_ratio: in ``replace_freeze``, per-sample ceiling on a
                written coefficient's magnitude = ``ratio × max|observed|``
                (fixed over the trajectory ⇒ provably bounds the output).
            sampler_sigma: C6 reverse-step noise scale. 0.0 (default) keeps the
                sampler fully deterministic, bit-for-bit identical to the
                pre-knob behavior (no generator is even constructed). σ>0 adds
                seeded Gaussian noise to each x0 estimate AFTER data
                consistency, masked off the observed support, BEFORE the
                magnitude clamp — so data consistency and boundedness both
                survive σ>0.
            sampler_seed: seed for the dedicated noise generator. The generator
                is RESEEDED at each ``sample()`` entry, making sampling a
                deterministic function of (measurement, mask, sampler_seed) in either
                module mode, since the reveals read ``cascade_batch_masks`` (#2471):
                same seed ⇒ identical reconstruction, every validation image
                sees the same noise realization (the C7 exchangeability story
                depends on this). ``None`` ⇒ nondeterministic σ>0 draws.
            selection_rule: which lines each reverse step reveals. Only
                ``"fixed"`` (the process' fixed-seed mask cascade) exists;
                validated so a stochastic rule can't slip in unvalidated.
            refine_alpha: weight of each newer estimate of an already-revealed
                coefficient under ``reveal_refine_dc``; required by that mode and
                rejected by every other (``reverse_refine.validate_refine_alpha``).
            integrity_check: ``off`` or ``raise``; with ``raise`` every write of
                the reverse loop is checked against ``reverse_integrity``'s
                invariants and the first violation raises.
            decay_envelope, projection, dc_trust, support_radius, reverse_dc: the
                physics options of the reverse loops, validated by
                ``reverse_physics.resolve_reverse_physics``.
        """
        super().__init__()
        # Fail loud at BUILD, not mid reverse-diffusion. A ``dc_method`` outside
        # the physics SSOT (or one the sampler can neither compute nor delegate)
        # would otherwise only surface on the first validation sample — the
        # 2026-07-05 ``dc_method='adaptive'`` crash (pitfall #9/#15).
        if dc_method not in VALID_DC_METHODS:
            raise ValueError(
                f"Unknown dc_method {dc_method!r} for PhysicsInformedColdDiffusion. "
                f"Valid choices: {sorted(VALID_DC_METHODS)}."
            )
        if reverse_mode not in VALID_REVERSE_MODES:
            raise ValueError(
                f"Unknown reverse_mode {reverse_mode!r} for "
                f"PhysicsInformedColdDiffusion. Valid: {sorted(VALID_REVERSE_MODES)}."
            )
        if reverse_mode == "additive":
            # ``additive`` is the DEFAULT here and at two other construction sites,
            # so an arm reaches it by saying nothing -- yet its correctness rests on
            # a precondition nothing checks. ``p_sample``'s update only behaves as a
            # replacement while the cascade is nested; on a leaking family a bin that
            # exits and re-enters is written twice and the contributions sum. It also
            # discards the data-consistency work done on the observed support. It is
            # retained for Bansal Algorithm-2 reproduction, so the honest move is to say
            # so once at construction rather than to change it or leave the default silent.
            logger.warning(
                "reverse_sampling_mode='additive' is the legacy Bansal Alg-2 loop and "
                "is correct ONLY on a nested mask cascade: it double-writes any bin "
                "that leaves and re-enters the support, discards data consistency on "
                "the observed support. Use "
                "'replace_freeze' (or 'replace_freeze_dc') unless you are explicitly "
                "reproducing the Bansal baseline."
            )
        if not (reverse_clip_ratio > 0):
            raise ValueError(f"reverse_clip_ratio must be > 0, got {reverse_clip_ratio!r}.")
        if clip_reference not in VALID_CLIP_REFERENCES:
            raise ValueError(
                f"Unknown clip_reference {clip_reference!r}. Valid: "
                f"{sorted(VALID_CLIP_REFERENCES)}."
            )
        validate_sampler_determinism(sampler_sigma, selection_rule)
        self.refine_alpha = validate_refine_alpha(reverse_mode, refine_alpha)
        self.integrity_check = validate_integrity_check(integrity_check)
        self.physics = resolve_reverse_physics(
            reverse_mode,
            dc_method,
            decay_envelope=decay_envelope,
            projection=projection,
            dc_trust=dc_trust,
            support_radius=support_radius,
            reverse_dc=reverse_dc,
        )
        self.model = model
        self.num_timesteps = num_timesteps
        self.dc_method = dc_method
        self.dc_weight = dc_weight
        self.sampling_steps = resolve_sampling_steps(sampling_steps, num_timesteps)
        self.reverse_mode = reverse_mode
        self.reverse_clip_ratio = float(reverse_clip_ratio)
        self.clip_reference = clip_reference
        self.kspace_log_scaled = bool(kspace_log_scaled)
        self.sampler_sigma = float(sampler_sigma)
        self.sampler_seed = None if sampler_seed is None else int(sampler_seed)
        self.selection_rule = selection_rule
        self._sampler_generator: torch.Generator | None = None
        #: Which member of a multi-sample ensemble the next stream belongs to;
        #: recorded by ``_reseed_sampler_generator`` and added to ``sampler_seed``.
        self._sampler_seed_offset: int = 0

        # SSOT degradation schedule. The model (the cold-diffusion generator)
        # already builds a ``kspace_process`` from the YAML acceleration block —
        # the *same* operator it trained against (step/power-law schedule,
        # base_acceleration, equispaced vs variable-density masks, the explicit
        # ``acceleration_range``). Reuse it so the reverse trajectory degrades
        # with exactly that schedule. Constructing a fresh process here would
        # silently fall back to library defaults (``schedule_type="linear"``,
        # ``base_acceleration=1.0``, ``mask_type="variable_density"``), making
        # the iterative sampler restore the model from masks it never saw at
        # each timestep — the experiment_11 "true cold diffusion" desync. The
        # fresh-construct path remains only for models that expose no
        # ``kspace_process`` (e.g. test stubs / non-generator denoisers). It is
        # deliberately left device-less: this class receives no config and no
        # device, and reading one off ``model.parameters()`` would make it a
        # second device resolver (non-negotiable 17) on a path no real generator
        # reaches. Real models arrive with the config-inherited device already on
        # ``model.kspace_process`` (#1508).
        model_process = getattr(model, "kspace_process", None)
        if isinstance(model_process, KSpaceUndersamplingProcess):
            self.process = model_process
        else:
            self.process = KSpaceUndersamplingProcess(
                num_timesteps=num_timesteps,
                max_acceleration=max_acceleration,
                center_fraction=center_fraction,
            )
        # The forward process owns the training noise level; the reverse denoiser input
        # is re-noised from the same declaration, so one knob drives both sides.
        validate_forward_noise_exclusive(self.process.forward_noise_sigma, sampler_sigma)

        # The t=0-terminal keying only delivers its terminal call when the
        # schedule actually reaches 0. On a ladder whose floor is 1 the mode
        # would run, report success, and call the model at t=1 exactly like the
        # convention it exists to replace -- an inert knob, not an error
        # (pitfall 15). The floor is 0 when base_acceleration > 1 or when
        # `undersampling.train_identity_rung` opens the identity rung.
        if self.reverse_mode in TERMINAL_OVERWRITE_REVERSE_MODES:
            _floor = self.process.min_meaningful_timestep()
            if _floor != 0:
                raise ValueError(
                    f"reverse_sampling_mode={self.reverse_mode!r} keys its reveal off the "
                    f"rung it is called at so that the terminal step runs at t=0, but this "
                    f"process floors the schedule at t={_floor}, so the terminal call would "
                    "land there and the mode would behave exactly like 'replace_freeze_dc'. "
                    "Set undersampling.train_identity_rung: true (base_acceleration == 1), "
                    "or use base_acceleration > 1."
                )

        # What the last reverse call actually ran, declared here so a caller can
        # read the attributes without first having sampled. ``last_terminal_step``
        # is the LOWEST timestep the model was called at, which is not the tail of
        # the schedule: under ``dc_method='hard'`` a step whose reveal is empty is
        # skipped, and on a ladder whose ``mask(0)`` is all-ones that is always the
        # t=0 step, so the terminal rung is scheduled and never evaluated (#2067).
        # Reported rather than inferred -- the previous two were computed and
        # discarded, which is how the skip stayed invisible.
        self.last_effective_steps: int = 0
        self.last_skipped_steps: int = 0
        self.last_terminal_step: int | None = None
        self.last_schedule: list[int] = []
        self.last_integrity_steps_verified: int = 0
        self.last_band_divisor_max: Tensor | None = None
        self.last_decay_slope: Tensor | None = None
        self.last_decay_unfitted: Tensor | None = None
        self.last_projection_rejected: Tensor | None = None

    @property
    def reverse_stats(self) -> dict[str, Any]:
        """What the last :meth:`sample` call ran, for the validation record.

        ``reverse_mode`` rides along because a consumer reasoning about WHICH
        step wrote a coefficient needs to know whether steps write once and
        freeze. Reading it off the sampler object would work and is what a
        caller reached for first; carrying it here keeps one transport for "what
        the last sample call did" instead of two.
        """
        return {
            "effective_steps": self.last_effective_steps,
            "skipped_steps": self.last_skipped_steps,
            "terminal_timestep_called": self.last_terminal_step,
            "schedule": list(self.last_schedule),
            "reverse_mode": str(self.reverse_mode),
            "integrity_check": self.integrity_check,
            "integrity_steps_verified": self.last_integrity_steps_verified,
            "band_divisor_max": (
                None if self.last_band_divisor_max is None else float(self.last_band_divisor_max)
            ),
            "decay_slope": (
                None if self.last_decay_slope is None else float(self.last_decay_slope)
            ),
            "decay_unfitted": (
                None if self.last_decay_unfitted is None else int(self.last_decay_unfitted)
            ),
            "projection_rejected_steps": (
                None
                if self.last_projection_rejected is None
                else int(self.last_projection_rejected)
            ),
            "reverse_physics": dataclasses.asdict(self.physics),
        }

    def _reseed_sampler_generator(self, seed_offset: int = 0) -> None:
        """Drop the noise generator so the next draw starts a fresh stream.

        Called at every ``sample()`` entry: per-call reseeding makes the sampler
        a deterministic function of (measurement, mask, sampler_seed) — exactly
        the C6 contract — and gives every validation image the same noise
        realization, which the C7 conformal calibration relies on.

        ``seed_offset`` names one member of a multi-sample ensemble: the next
        stream is seeded with ``sampler_seed + seed_offset``, so member ``i`` is
        a deterministic function of (measurement, mask, sampler_seed, i) and two
        members never replay the same noise. Offset 0 is the legacy stream, so a
        caller that never passes it is bit-for-bit unchanged. With
        ``sampler_seed=None`` every stream is drawn from entropy and the offset
        has nothing to shift.
        """
        if isinstance(seed_offset, bool) or not isinstance(seed_offset, int) or seed_offset < 0:
            raise ValueError(f"seed_offset must be a non-negative int, got {seed_offset!r}.")
        self._sampler_seed_offset = seed_offset
        self._sampler_generator = None

    def _active_sampler_generator(self) -> torch.Generator:
        """The (lazily created, stored) CPU generator for reverse-step noise.

        Create-once-and-store per trajectory — NOT per step. Reseeding inside
        each ``p_sample`` call would replay the identical noise tensor at every
        timestep (correlated across steps, silently wrong); here the stream
        advances step to step and only ``_reseed_sampler_generator`` resets it.
        """
        if self._sampler_generator is None:
            generator = torch.Generator(device="cpu")
            if self.sampler_seed is not None:
                generator.manual_seed(self.sampler_seed + self._sampler_seed_offset)
            else:
                # A fresh torch.Generator has a FIXED default state, which would
                # make seed=None silently reproducible. None means "genuinely
                # nondeterministic", so seed from system entropy.
                generator.seed()
            self._sampler_generator = generator
        return self._sampler_generator

    def denoise(self, x: Tensor, t: Tensor, support: Tensor) -> Tensor:
        """One model call, conditioned on the support ``x`` carries at this step.

        Training feeds the network the rung's mask ``M_t`` beside ``x_t``, so every
        reverse call passes the support its state actually has (observed plus
        committed bins), never the observation mask alone, which is the support of
        the trajectory's head only. ``acquired`` is the scanner-written plane, or
        None when no ``measured_lines`` were declared.

        Args:
            x: Current k-space state ``[B, C, H, W]``.
            t: Timestep per sample ``[B]``.
            support: Known bins of ``x``, ``[B, 1 or C, H, W]``.

        Returns:
            The model's x0 estimate (the first element when it returns a tuple).
        """
        if self.process.forward_noise_sigma > 0:
            x = self._renoise_unmeasured(x, t, support)
        plane = (support > 0).any(dim=-3, keepdim=True).to(dtype=x.real.dtype)
        acquired = self.acquirable_plane(tuple(x.shape[-2:]))
        if acquired is not None:
            acquired = acquired.to(device=x.device, dtype=x.real.dtype)
        x0 = self.model(x, t, mask=plane, acquired=acquired)
        return x0[0] if isinstance(x0, tuple) else x0

    def _renoise_unmeasured(self, x: Tensor, t: Tensor, support: Tensor) -> Tensor:
        """The denoiser's input with the uncommitted unmeasured bins re-noised at ``sigma_t``.

        Measured and committed bins (``support``) are left as the state holds them. The draw
        comes from the sampler's seeded stream, so a sampler seed reproduces it exactly. The
        state itself is not touched: only the model input carries the noise.
        """
        known = (support > 0).to(dtype=x.real.dtype)
        level = forward_noise_levels(
            self.process.forward_noise_sigma,
            self.process.forward_noise_schedule,
            t,
            self.process.num_timesteps,
        )
        eps = torch.randn(
            x.shape, generator=self._active_sampler_generator(), dtype=x.dtype, device="cpu"
        ).to(x.device)
        return x + hot_noise_increment(x, known, level.to(x.device), eps)

    def _recovered_mask(
        self, t: Tensor, t_prev: Tensor, shape: tuple[int, int], device: torch.device
    ) -> Tensor:
        """Bins the cascade reveals between level ``t`` and the less-degraded ``t_prev``."""
        mask_t = self.process.cascade_batch_masks(t, shape, device, acquired=self.measured_lines)
        mask_t_prev = self.process.cascade_batch_masks(
            t_prev, shape, device, acquired=self.measured_lines
        )
        # Nested cascade: mask_t is a subset of mask_t_prev.
        return torch.clamp(mask_t_prev.float() - mask_t.float(), min=0.0)

    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        measurement: Tensor,
        mask: Tensor,
        t_next: Tensor | None = None,
        support: Tensor | None = None,
        is_final: bool | None = None,
    ) -> Tensor:
        """Single reverse diffusion step with data consistency.

        Args:
            x_t: Current k-space estimate [B, C, H, W]
            t: Current timestep [B]
            measurement: Acquired k-space [B, C, H, W]
            mask: Sampling mask [B, 1, H, W]
            t_next: The level this step restores to, the next scheduled timestep;
                ``None`` means ``t - 1``. Bansal's update is
                ``x_{t'} = x_t - D(x0, t) + D(x0, t')`` for the NEXT level ``t'``.
            support: The bins ``x_t`` already carries, passed to the model;
                ``None`` means ``mask`` (a state holding the measurement only).
            is_final: Whether this is the ``t == 0`` step, when the caller already knows it
                as a Python int; ``None`` reads it from ``t`` (a host sync).

        Returns:
            x_t_minus_1: Less-degraded k-space estimate
        """
        # 1. Predict clean k-space x_0
        x_0_pred = self.denoise(x_t, t, mask if support is None else support)

        # 2. Apply Data Consistency (CRITICAL for physics)
        if self.dc_method == "hard":
            # Replace predicted values with measurements where we have data
            x_0_pred = x_0_pred * (1.0 - mask) + measurement * mask
        else:
            # soft / noise_adjusted / adaptive / kan_adaptive / target_aware_fsdc
            # / noise_adaptive: reuse the model's trained ``dc_layer`` — the SAME
            # operator it trained with — exactly as the schedule reuses
            # ``model.kspace_process``. x_0_pred is k-space here, so
            # ``is_kspace_domain=True`` skips the internal FFT.
            #
            # ``SoftDataConsistency`` (the layer 'soft' builds)
            # is a proximal blend ``(k_pred + lam*y)/(1+lam)`` where ``dc_weight``
            # feeds ``lam`` — a trust temperature, not a blend fraction
            # (``infrastructure/physics/dc_settings.py`` is the SSOT for that
            # reading). Delegating to the layer is what makes ``lam`` a live,
            # optimizer-moved parameter here rather than a fixed coefficient.
            dc_layer = getattr(self.model, "dc_layer", None)
            if dc_layer is None:
                raise ValueError(
                    f"dc_method={self.dc_method!r} needs the model's learned "
                    "dc_layer to delegate to, but the model exposes no "
                    "'dc_layer'. Either build the generator with this dc_method "
                    "(so it constructs the layer) or use 'hard'."
                )
            x_0_pred = dc_layer(x_0_pred, measurement, mask, is_kspace_domain=True)

        # C6: optional stochastic reverse step, AFTER data consistency and
        # masked off the observed support, so measured lines stay exactly
        # data-consistent at any σ. σ=0 (default): no generator, no draw —
        # bit-for-bit the deterministic sampler.
        if self.sampler_sigma > 0:
            x_0_pred = inject_reverse_step_noise(
                x_0_pred,
                self.sampler_sigma,
                self._active_sampler_generator(),
                exclude_support=mask,
            )

        # 3. Get target state x_{t-1}
        if (t.min() <= 0) if is_final is None else is_final:
            # At t=0, return final prediction
            return x_0_pred

        # Restore to the next scheduled level: under a strided schedule ``t - 1`` reveals one
        # level per call and never writes the bands between scheduled levels.
        t_prev = torch.clamp(t - 1, min=0) if t_next is None else t_next

        # Cold Diffusion restoration x_{t-1} = x_t - D(x_0, t) + D(x_0, t-1): the
        # lines clean at t-1 but corrupt at t.
        shape = tuple(x_0_pred.shape[-2:])
        mask_recovered = self._recovered_mask(t, t_prev, shape, x_0_pred.device)

        # 3. Recovered lines from prediction
        recovered_lines = x_0_pred * mask_recovered

        # 4. Update state: Add recovered lines to the current estimate.
        #
        # ``mask_recovered`` is clamped at 0, so it is exactly zero wherever
        # ``mask_t`` is set: a bin already in the cascade support is never written
        # again and the addition IS a replacement -- but only while the cascade is
        # NESTED. If a bin leaves the support and re-enters (which is what a
        # non-nesting family does, e.g. radial leaks at 24 of 28 levels), it is
        # written a SECOND time and the two contributions sum: measured 2.0 on a
        # hand-built 4-bin cascade whose x_0_pred was 1.0 everywhere. Nesting is the
        # precondition for this line, not a nicety.
        #
        # A further limitation of this legacy branch, fixed in ``_sample_replace_freeze``
        # and deliberately NOT changed here because ``additive`` reproduces Bansal
        # Algorithm 2:
        #   * everything computed on the observed support -- hard/soft/learned data
        #     consistency, clipping, sampler noise -- is discarded, since only
        #     ``x_0_pred * mask_recovered`` is ever written. A stale ``x_t`` value on
        #     an observed bin survives even when DC would correct it.
        # Prefer a freeze mode for new arms; all 60 corpus arms that enable the
        # multi-step sampler declare ``replace_freeze_dc`` (measured 2026-09-13 --
        # re-measure rather than quoting this).
        x_t_minus_1 = x_t + recovered_lines

        return x_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        measurement: Tensor,
        mask: Tensor,
        return_trajectory: bool = False,
        start_timestep: int | None = None,
        seed_offset: int = 0,
        measured_lines: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Full reverse diffusion sampling.

        ``measured_lines`` (``[B, N]``) are the lines each sample acquired; every
        rung mask of the reverse loop then counts R over them, as the forward
        process that built ``measurement`` did (#2160). None keeps full support.

        Args:
            measurement: Acquired k-space [B, C, H, W]
            mask: Sampling mask [B, 1, H, W]
            return_trajectory: If True, return all intermediate states
            start_timestep: Timestep the measurement is actually degraded AT, i.e.
                the head of the reverse trajectory. ``None`` keeps the legacy
                ``num_timesteps - 1`` head, which is only correct for a
                measurement at MAX acceleration. Cascading validation evaluates
                several rungs against the SAME schedule, so every rung below the
                top used to replay the fully-degraded trajectory: at R=2 the
                measurement sits at t=1 and the 27 steps above it cannot write
                anything (their ``mask_next`` is inside the observed support), so
                they were pure overhead. Pass the rung's own timestep and the
                schedule starts where the data actually is.

            seed_offset: Ensemble member index (``>= 0``). Seeds this call's
                noise stream with ``sampler_seed + seed_offset`` so the N members
                of a validation ensemble draw distinct, individually reproducible
                noise (``validation.sampling.ensemble_samples``). 0 (default) is
                the stream every single-sample call has always used.

        Returns:
            x_0: Reconstructed k-space [B, C, H, W]
            trajectory: Optional list of intermediate states

        Raises:
            ValueError: If ``start_timestep`` falls outside
                ``[min_meaningful_timestep(), num_timesteps - 1]``. Below the
                floor is rejected rather than clamped because ``torch.linspace``
                ASCENDS for an inverted range, and the ``sorted(..., reverse=True)``
                normalisation below would then reinstate timesteps ABOVE the
                requested head — silently restoring the very trajectory the
                caller asked to skip (non-negotiable 3: no silent fallbacks).
        """
        # C6: fresh noise stream per call — same (measurement, mask, seed) in,
        # same reconstruction out, regardless of how many samples ran before.
        self._reseed_sampler_generator(seed_offset)
        self.last_integrity_steps_verified = 0
        self.last_band_divisor_max = None
        self.last_decay_slope = None
        self.last_decay_unfitted = None
        self.last_projection_rejected = None
        self.measured_lines = measured_lines

        # [STABILIZATION FIX] Strided Reverse Diffusion
        # Calculate stride to sample timesteps uniformly.
        #
        # The schedule STOPS at the process' minimum meaningful timestep rather
        # than always at 0 (issue #535). ``sample_timesteps`` never draws t below
        # that floor during training, so a schedule that ends at 0 when the floor
        # is 1 evaluates its FINAL step — the one whose full reveal writes 54-99%
        # of the reconstruction — at a timestep the time embedding never saw. The
        # final step reveals every remaining coefficient regardless of its t, so
        # stopping at the floor loses nothing.
        steps = self.sampling_steps
        t_floor = self.process.min_meaningful_timestep()
        t_head = self.num_timesteps - 1 if start_timestep is None else int(start_timestep)
        if not t_floor <= t_head <= self.num_timesteps - 1:
            raise ValueError(
                f"start_timestep={t_head} is outside the meaningful reverse range "
                f"[{t_floor}, {self.num_timesteps - 1}] for this process "
                f"(base_acceleration gives min_meaningful_timestep()={t_floor}). "
                "Pass the timestep the measurement is degraded at, or None for the "
                "fully-degraded head."
            )
        timestep_schedule = torch.linspace(t_head, t_floor, steps + 1).long().tolist()

        # Ensure unique and strictly decreasing
        timestep_schedule = sorted(set(timestep_schedule), reverse=True)

        # Published for the validation record: the reveal partition is defined
        # by this schedule, and re-deriving it downstream would be a second
        # owner of it (non-negotiable 17).
        self.last_schedule = list(timestep_schedule)

        loop = _REVERSE_LOOPS.get(self.reverse_mode)
        if loop is None:
            raise ValueError(
                f"reverse_mode={self.reverse_mode!r} has no reverse loop. "
                f"Registered: {sorted(_REVERSE_LOOPS)}."
            )
        result = loop(self, measurement, mask, timestep_schedule, return_trajectory)
        # Never-acquired lines are zero in the target whatever the loop wrote there;
        # one projection here owns that for every loop.
        acquirable = self.acquirable_plane(tuple(measurement.shape[-2:]))
        if acquirable is None:
            return result
        acquirable = acquirable.to(device=measurement.device, dtype=measurement.real.dtype)
        if return_trajectory:
            out, trajectory = result
            return out * acquirable, trajectory
        return result * acquirable

    @torch.no_grad()
    def _sample_additive(
        self,
        measurement: Tensor,
        mask: Tensor | None,
        timestep_schedule: list[int],
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Bansal Algorithm-2 loop (``reverse_mode='additive'``), one ``p_sample`` per step."""
        device = measurement.device
        B = measurement.shape[0]
        x = measurement.clone()
        trajectory = [x] if return_trajectory else None
        # The same support the freeze loops fall back to when no mask is passed.
        mask = self._observed_support(measurement, mask, 1e-8)
        if self.integrity_check != "off":
            ceil = self._magnitude_ceiling(measurement, mask, 1e-8)
        support = mask.amax(dim=-3, keepdim=True)
        for step, t_idx in enumerate(timestep_schedule):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            t_next = (
                torch.full((B,), timestep_schedule[step + 1], device=device, dtype=torch.long)
                if step + 1 < len(timestep_schedule)
                else None
            )
            x = self.p_sample(
                x, t, measurement, mask, t_next=t_next, support=support, is_final=t_idx <= 0
            )
            if t_idx > 0:
                t_prev = torch.clamp(t - 1, min=0) if t_next is None else t_next
                grown = self._recovered_mask(t, t_prev, tuple(x.shape[-2:]), device)
                support = torch.clamp(support + grown.amax(dim=-3, keepdim=True), max=1.0)
            if self.integrity_check != "off":
                check_step(self, x, measurement, mask, step=step, t=t_idx, ceil=ceil)

            if return_trajectory:
                trajectory.append(x.clone())

        # The additive loop skips nothing, so every scheduled step is a model
        # call and the tail of the schedule is the terminal one.
        self.last_effective_steps = len(timestep_schedule)
        self.last_skipped_steps = 0
        self.last_terminal_step = timestep_schedule[-1] if timestep_schedule else None

        if return_trajectory:
            return x, trajectory
        return x

    def _observed_support(self, measurement: Tensor, mask: Tensor | None, eps: float) -> Tensor:
        """Observed support: the sampling mask, else the nonzero measurement coefficients.

        Observation is a property of the COEFFICIENT, not of Re and Im
        separately: an observed coefficient whose imaginary part happens to be 0
        must not be treated as half-unobserved (that would leave the model's Im in
        place under hard DC). Pair, test, expand back.
        """
        if mask is not None:
            return (mask > 0).float()
        _obs_pair = (paired_magnitude(measurement) > eps).float()
        return _obs_pair if torch.is_complex(measurement) else _obs_pair.repeat_interleave(2, dim=-3)

    @torch.no_grad()
    def _sample_replace_freeze(
        self,
        measurement: Tensor,
        mask: Tensor | None,
        timestep_schedule: list[int],
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Corrected monotone-infill reverse loop (``reverse_mode='replace_freeze'``).

        Each revealed coefficient is written exactly ONCE from the x0 estimate
        (hard-DC'd on the observed support, then magnitude-bounded) and then
        FROZEN; reveals come from the NEXT SCHEDULED level so the revealed bands
        tile the full support (fixes the additive loop's strided ``t-1``
        skip/re-hit). The observed support is hard-pinned to ``measurement``
        every step, and every written magnitude is capped at
        ``reverse_clip_ratio × max|observed|`` — a per-sample constant fixed over
        the whole trajectory, which provably bounds the output and removes the
        inference-only blow-up.
        """
        device = measurement.device
        B = measurement.shape[0]
        eps = 1e-8
        n = len(timestep_schedule)

        x = measurement.clone()

        # Observed support (exact, frozen).
        obs = self._observed_support(measurement, mask, eps)

        # Per-sample magnitude ceiling from the measured coefficients, FIXED over
        # the trajectory: max|written| ≤ ratio · max|observed| ⇒ ‖x‖∞ bounded.
        ceil = self._magnitude_ceiling(measurement, obs, eps)
        write_ceil = measured_write_ceiling(
            ceil, measurement, obs.amax(dim=-3, keepdim=True) > 0, "hard"
        )

        committed = obs.clone()
        trajectory = [x.clone()] if return_trajectory else None
        # ``committed`` is seeded with ``obs`` here, so the plan needs no obs term.
        plan = plan_reveals(
            self, timestep_schedule, committed, torch.zeros_like(obs), x, B, device
        )

        skipped = 0
        terminal: int | None = None
        for i, t_idx in enumerate(timestep_schedule):
            # Skip provably-inert steps (issue #535). This variant hard-pins the
            # observed support unconditionally, so a step that reveals nothing
            # leaves ``x`` bit-for-bit unchanged.
            if not plan.active[i]:
                skipped += 1
                if return_trajectory:
                    trajectory.append(x.clone())
                continue

            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            terminal = t_idx
            # ``committed`` was seeded with ``obs``, so it is the whole support of x.
            x0 = self.denoise(x, t, committed)

            # Hard, idempotent data consistency on the observed support.
            x0 = x0 * (1.0 - obs) + measurement * obs

            # C6: optional reverse-step noise — after DC, off the observed
            # support, BEFORE the clamp so the boundedness proof still holds.
            # (Skipped inert steps above make no model call and hence no draw,
            # so their bit-for-bit no-op justification survives σ>0.)
            if self.sampler_sigma > 0:
                x0 = inject_reverse_step_noise(
                    x0, self.sampler_sigma, self._active_sampler_generator(), exclude_support=obs
                )

            # Radial, phase-preserving (#1281); the hard-pinned support is left alone (#2310).
            x0 = clamp_to_magnitude_ceiling(x0, write_ceil, eps)

            # Reveal newly-uncovered lines, write once, then freeze. On the FINAL
            # step reveal EVERY remaining line: the reconstruction target is fully
            # sampled, but ``q_sample(x0, t=0)`` returns the base-acceleration mask
            # (support 1/base_acceleration), so relying on it leaves
            # 1 - 1/base_acceleration of k-space permanently zero
            # (base_acceleration=2.0 => 50% dropped => a hard low-pass "DC blob";
            # see docs/experiment_11_kspace_cold_diffusion.rst). The revealed lines
            # carry the model's own (full-target-supervised) prediction, never the
            # measurement, so this does not leak ground truth.
            # The skip gate and the write read the same plan, so they cannot disagree.
            reveal = plan.reveal(i, committed.dtype)
            x = x * (1.0 - reveal) + x0 * reveal
            check_step(
                self, x, measurement, obs, step=i, t=t_idx, ceil=ceil,
                reveal=reveal, committed=committed,
            )
            committed = torch.clamp(committed + reveal, max=1.0)

            if return_trajectory:
                trajectory.append(x.clone())

        check_coverage(self, committed, obs)
        if skipped:
            logger.info(
                "cold reverse loop: %d of %d scheduled steps were inert (no reveal) "
                "and were skipped; %d model call(s) actually ran",
                skipped,
                n,
                n - skipped,
            )
        self.last_effective_steps = n - skipped
        self.last_skipped_steps = skipped
        self.last_terminal_step = terminal

        if return_trajectory:
            return x, trajectory
        return x

    def _magnitude_ceiling(self, measurement: Tensor, obs: Tensor, eps: float) -> Tensor:
        """Per-coefficient magnitude ceiling for written predictions.

        ``global_max`` is the legacy per-sample constant
        ``reverse_clip_ratio * max|observed|``. Because ``max|observed|`` is the
        k-space DC peak (~37x the RMS coefficient on this data), a ratio near 1
        permits every unobserved coefficient to be written at ~48 sigma — a bound
        that bounds nothing that matters. ``band_local`` references each
        coefficient to its own radial frequency band instead (issue #536).
        """
        if self.clip_reference == "band_local":
            return band_local_magnitude_ceiling(
                measurement * obs,
                self.reverse_clip_ratio,
                eps=eps,
                log_scaled=self.kspace_log_scaled,
            )
        # TRUE complex modulus over the observed support. ``obs`` is per-channel
        # and the modulus is per-COEFFICIENT, so the mask is paired down the same
        # way; a coefficient counts as observed when its (Re, Im) pair is.
        mag = paired_magnitude(measurement)
        # ``obs`` reaches here in one of three shapes and only ONE of them needs
        # pairing:
        #   * 1 channel  -- the sampling mask, already channel-broadcast;
        #   * C channels, complex measurement -- matches ``mag`` as-is;
        #   * C channels, real-stacked measurement -- pair down to C//2.
        # Pairing a 1-channel mask would raise on the odd channel count, which is
        # exactly what the replace_freeze suite caught.
        if obs.shape[-3] == 1 or torch.is_complex(measurement):
            obs_paired = obs > 0
        else:
            obs_paired = paired_magnitude(obs) > 0
        reduce_dims = tuple(range(1, mag.dim()))
        ref = (mag * obs_paired).amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
        return apply_ceiling_ratio(ref, self.reverse_clip_ratio, self.kspace_log_scaled)

    def _reveal_support(
        self,
        i: int,
        timestep_schedule: list[int],
        committed: Tensor,
        obs: Tensor,
        x: Tensor,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        """Which coefficients reverse step ``i`` writes.

        The single owner of the reveal arithmetic. ``reveal_plan.plan_reveals``
        walks it over the schedule before the loop, and the loop both skips and
        writes from that plan, so the skip gate and the write cannot disagree
        about a step (non-negotiable 17).

        The terminal step takes every remaining unobserved coefficient the scanner
        could have acquired: the reconstruction target is fully sampled there. Lines
        outside ``measured_lines`` were never acquired and are zero in the target, so
        they keep the measurement's zeros instead of the model's estimate.
        """
        n = len(timestep_schedule)
        if i + 1 >= n:
            remaining = (1.0 - committed) * (1.0 - obs)
            acquirable = self.acquirable_plane(tuple(x.shape[-2:]))
            return remaining if acquirable is None else remaining * acquirable
        key_t = timestep_schedule[i + 1]
        key_mask = self.process.cascade_batch_masks(
            torch.full((batch,), key_t, device=device, dtype=torch.long),
            tuple(x.shape[-2:]),
            device,
            acquired=self.measured_lines,
        )
        return torch.clamp(key_mask.float() * (1.0 - committed) * (1.0 - obs), 0.0, 1.0)

    def acquirable_plane(self, shape: tuple[int, int]) -> Tensor | None:
        """``[B, 1, H, W]`` float plane of ``measured_lines``, or ``None`` when undeclared."""
        if self.measured_lines is None:
            return None
        from spectramr.models.diffusion.acquired_line_budget import lines_to_plane

        return lines_to_plane(self.measured_lines, shape, self.process.line_dim)

    def _apply_observed_dc(self, x0: Tensor, measurement: Tensor, obs: Tensor) -> Tensor:
        """Data consistency on the OBSERVED support, per ``self.dc_method``.

        Mirrors the ``p_sample`` DC branching but restricted to the observed
        support ``obs`` (unobserved lines are owned by the reveal path, left as
        the raw prediction). ``hard`` reproduces the measurement pin; every other
        method (``soft`` / ``noise_adjusted`` / ``adaptive`` / ``kan_adaptive`` /
        ``target_aware_fsdc`` / ``noise_adaptive``) delegates to the model's
        trained ``dc_layer`` so the SAME operator used at training — including a
        learned ``lambda_param`` for soft DC — is applied at sampling.
        ``dc_weight`` feeds that layer's ``lam`` (a trust temperature, not a
        blend fraction; ``dc_settings.py`` is the SSOT for that reading), so
        delegating rather than recomputing the blend here is what keeps a
        live, optimizer-moved ``lam`` in effect at sampling too.
        """
        if self.physics.reverse_dc == "unconstrained":
            return x0
        if self.dc_method == "hard":
            return x0 * (1.0 - obs) + measurement * obs
        dc_layer = getattr(self.model, "dc_layer", None)
        if dc_layer is None:
            raise ValueError(
                f"dc_method={self.dc_method!r} needs the model's learned dc_layer "
                "to delegate to in replace_freeze_dc mode, but the model exposes "
                "no 'dc_layer'. Build the generator with this dc_method or use "
                "'hard'."
            )
        dc_out = dc_layer(x0, measurement, obs, is_kspace_domain=True)
        # Full output, as training applies it: Wiener gain writes alpha*k off the support.
        return dc_out

    @torch.no_grad()
    def _sample_replace_freeze_dc(
        self,
        measurement: Tensor,
        mask: Tensor | None,
        timestep_schedule: list[int],
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Monotone-infill reverse loop that HONORS ``dc_method`` on observed lines.

        Same reveal-once-then-freeze machinery as ``_sample_replace_freeze`` for the
        UNobserved support, but the observed support is updated EVERY step with the
        configured data consistency (``dc_method``) instead of being hard-frozen at
        the raw measurement. A soft / adaptive / noise-adaptive ``dc_method`` can
        thus DENOISE the measured k-space lines (e.g. low-field super-resolution)
        while the fixed magnitude clamp still bounds the output. ``dc_method='hard'``
        reproduces ``replace_freeze`` (observed lines == measurement).

        Boundedness/stability: the observed update is a data-consistent blend toward
        the fixed ``measurement`` anchor (contraction), and the phase-preserving
        magnitude clamp to ``ceil = reverse_clip_ratio × max|observed|`` (fixed over
        the trajectory) is applied AFTER it, so the output stays bounded by ``ceil``
        for every ``dc_method``.
        """
        device = measurement.device
        B = measurement.shape[0]
        eps = 1e-8
        n = len(timestep_schedule)

        x = measurement.clone()

        obs = self._observed_support(measurement, mask, eps)
        ctx = prepare_reverse(self, measurement, obs, eps=eps)
        ceil = ctx.ceil

        # Observed lines are OWNED by the per-step DC update (not the reveal path),
        # so seed ``committed`` with zeros and gate reveals away from ``obs``.
        committed = torch.zeros_like(obs)
        trajectory = [x.clone()] if return_trajectory else None
        plan = plan_reveals(self, timestep_schedule, committed, obs, x, B, device)

        skipped = 0
        terminal: int | None = None
        for i, t_idx in enumerate(timestep_schedule):
            # Skip provably-inert steps (issue #535). When a step reveals nothing
            # AND dc_method is hard, its whole effect is
            # ``x <- x*(1-obs) + measurement*obs`` on a support already equal to
            # ``measurement`` — bit-for-bit identical to x. At R=2 the observed
            # support covers every mask_next, so 19 of 21 steps were exact no-ops:
            # the model was called 21 times and 19 outputs were discarded. Their
            # cost is real (~90% of validation compute) and their presence made a
            # single-shot reconstruction look like a 20-step trajectory.
            pinned = self.dc_method == "hard" and self.physics.reverse_dc == "per_step"
            # The terminal t=0 call of the t0 mode runs even when it reveals nothing:
            # it rewrites the unobserved bins, so it is the decisive estimate.
            overwrite = self.reverse_mode in TERMINAL_OVERWRITE_REVERSE_MODES and i + 1 == n
            if pinned and not plan.active[i] and not overwrite:
                skipped += 1
                if return_trajectory:
                    trajectory.append(x.clone())
                continue

            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            terminal = t_idx
            x0 = self.denoise(x, t, torch.clamp(obs + committed, max=1.0))

            # DC, projection and reverse-step noise, then the phase-preserving
            # clamp last so the bound holds for every dc_method and sigma (#1281).
            where = f"{self.reverse_mode} step {i} (t={t_idx})"
            x0 = refine_estimate(self, x0, measurement, obs, ctx, where=where)
            x0 = bound_estimate(self, x0, obs, ctx, eps)

            # Reveal UNobserved lines, write once, freeze (observed lines are
            # owned by the DC update above). On the FINAL step reveal EVERY
            # remaining unobserved line: the reconstruction target is fully
            # sampled, but ``q_sample(x0, t=0)`` returns only the
            # base-acceleration mask (support 1/base_acceleration), so relying on
            # it leaves 1 - 1/base_acceleration of k-space permanently zero
            # (base_acceleration=2.0 => 50% => a hard low-pass "DC blob"; see
            # docs/experiment_11_kspace_cold_diffusion.rst). Filled with the
            # model's own (full-target-supervised) prediction, never the
            # measurement, so no ground-truth leak.
            reveal = plan.reveal(i, committed.dtype)
            # Observed support (re-blended each step) + freshly revealed lines.
            if overwrite:
                unobserved = 1.0 - obs
                acquirable = self.acquirable_plane(tuple(x.shape[-2:]))
                if acquirable is not None:
                    unobserved = unobserved * acquirable.to(unobserved.dtype)
                own = torch.clamp(unobserved + obs, 0.0, 1.0)
            else:
                own = torch.clamp(reveal + obs, 0.0, 1.0)
            x = x * (1.0 - own) + x0 * own
            check_step(
                self, x, measurement, obs, step=i, t=t_idx, ceil=ceil,
                reveal=reveal, committed=committed, trust_radius=ctx.trust_radius,
            )
            committed = torch.clamp(committed + reveal, max=1.0)

            if return_trajectory:
                trajectory.append(x.clone())

        check_coverage(self, committed, obs)
        if skipped:
            logger.info(
                "cold reverse loop: %d of %d scheduled steps were inert (no reveal "
                "under hard DC) and were skipped; %d model call(s) actually ran",
                skipped,
                n,
                n - skipped,
            )
        self.last_effective_steps = n - skipped
        self.last_skipped_steps = skipped
        self.last_terminal_step = terminal

        if return_trajectory:
            return x, trajectory
        return x


#: ``reverse_mode`` -> reverse loop. Every key of ``VALID_REVERSE_MODES`` must be
#: here; ``test_kspace_process`` pins the two sets equal.
_REVERSE_LOOPS = {
    "additive": PhysicsInformedColdDiffusion._sample_additive,
    "additive_normalized": sample_additive_normalized,
    "replace_freeze": PhysicsInformedColdDiffusion._sample_replace_freeze,
    "replace_freeze_dc": PhysicsInformedColdDiffusion._sample_replace_freeze_dc,
    "replace_freeze_dc_t0": PhysicsInformedColdDiffusion._sample_replace_freeze_dc,
    "reveal_refine_dc": sample_reveal_refine,
    "single_call_dc": sample_single_call,
    "support_ordered_dc": sample_support_ordered,
}


__all__ = [
    "DEFAULT_SAMPLING_STEPS",
    "VALID_SELECTION_RULES",
    "KSpaceUndersamplingProcess",
    "FORWARD_NOISE_SCHEDULES",
    "PhysicsInformedColdDiffusion",
    "forward_noise_levels",
    "hot_noise_increment",
    "inject_reverse_step_noise",
    "measured_write_ceiling",
    "paired_complex",
    "resolve_sampling_steps",
    "resolve_undersampling_kwargs",
    "validate_forward_noise",
    "validate_forward_noise_exclusive",
    "validate_sampler_determinism",
]
