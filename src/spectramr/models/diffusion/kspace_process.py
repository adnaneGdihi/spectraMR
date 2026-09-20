"""K-Space Undersampling Diffusion Process.

Physics-informed degradation operator that models MRI acquisition
as deterministic undersampling rather than Gaussian noise.

This implements the forward process q(x_t | x_0) = x_0 * M_t where M_t
is a sampling mask with acceleration factor R(t) that increases with t.

Reference: Bansal et al., "Cold Diffusion: Inverting Arbitrary Image Transforms"
           Applied to MRI: Degradation is undersampling, not noise.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.data_consistency import VALID_DC_METHODS

# Reverse-process variants for the cold-diffusion k-space sampler.
#   additive       — legacy Bansal-Alg-2 accumulate loop (x += x0*(M_{t-1}-M_t));
#                    kept as the reproduction baseline.
#   replace_freeze — corrected monotone infill: write each revealed coefficient
#                    once (hard-DC'd on the observed support, magnitude-bounded),
#                    reveal from the NEXT SCHEDULED level, then freeze. Fixes the
#                    inference-only magnitude blow-up (see docs).
#   replace_freeze_dc — same reveal-once-freeze for the UNobserved support, but the
#                    observed support HONORS ``dc_method`` every step (soft / adaptive
#                    / noise_adaptive) instead of hard-freezing to the raw measurement.
#                    Lets a learned/soft DC DENOISE the measured lines (low-field
#                    super-resolution) while the magnitude clamp still bounds output.
#                    ``replace_freeze`` (above) is left byte-identical for existing arms.
VALID_REVERSE_MODES: frozenset[str] = frozenset(
    {"additive", "replace_freeze", "replace_freeze_dc", "replace_freeze_dc_t0"}
)

#: Reverse modes whose reveal at step ``i`` keys off ``mask(schedule[i])`` -- the
#: rung the model is CALLED at -- rather than ``mask(schedule[i+1])``.
#:
#: The default keying makes the step at ``t`` write what the NEXT rung would
#: reveal, so on a ladder whose ``mask(0)`` is all-ones the step labelled ``t=1``
#: writes every remaining coefficient and the scheduled ``t=0`` step is inert and
#: skipped -- the decisive write is made by a call whose time embedding says
#: ``t=1`` (#2067). Under this keying the rung and the write agree, and the
#: terminal call happens at ``t=0``.
#:
#: A separate mode value rather than a change to the existing ones: 60 corpus
#: arms declare ``replace_freeze_dc`` and their published reconstructions were
#: produced under the old convention.
CURRENT_STEP_KEYED_REVERSE_MODES: frozenset[str] = frozenset({"replace_freeze_dc_t0"})

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
    return torch.sqrt(re * re + im * im)


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
            shape is required by this change.
        eps: modulus floor, guarding 0/0 on unwritten coefficients.

    Returns:
        Same shape and dtype as ``x``.
    """
    scale = (ceiling / paired_magnitude(x).clamp_min(eps)).clamp(max=1.0)
    if torch.is_complex(x):
        return x * scale
    # One scale per (Re, Im) pair -> expand back over the interleaved axis so
    # both components of a coefficient move together (this is what preserves
    # the phase).
    return x * scale.repeat_interleave(2, dim=-3)


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
    ``**kwargs``) > default, preserved verbatim from the generator including its
    falsy-means-unset ``or`` chaining: an explicit ``center_fraction: 0.0``
    falls through to the default rather than being honoured.

    Args:
        acceleration_config: ``AccelerationConfigSchema``, its ``model_dump()``,
            or a raw mapping. ``None`` is treated as empty.
        overrides: Secondary source, typically the generator's ``model_kwargs``.

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

    # Without this the step schedule falls back to [1.0, max_acceleration], so
    # R(0)=1 is fully sampled and t=0 degenerates to the identity.
    accel_range = accel.get("acceleration_range")
    if accel_range is not None and "acceleration_range" not in schedule_kwargs:
        schedule_kwargs["acceleration_range"] = list(accel_range)

    resolved: dict[str, Any] = {
        "max_acceleration": accel.get("max_acceleration") or over.get("max_acceleration", 32.0),
        "base_acceleration": accel.get("base_acceleration") or over.get("base_acceleration", 1.0),
        "center_fraction": accel.get("center_fraction") or over.get("center_fraction", 0.0325),
        "min_center_fraction": accel.get("min_center_fraction") or over.get("min_center_fraction"),
        "mask_type": _plain(accel.get("acceleration_type"))
        or over.get("acceleration_type", "variable_density"),
        "seed": accel.get("mask_seed") or over.get("seed", 42),
        "schedule_type": _plain(accel.get("schedule_type"))
        or over.get("schedule_type", "power_law"),
        "schedule_kwargs": schedule_kwargs,
        "enable_dynamic_mask": bool(
            accel.get("enable_dynamic_mask", over.get("enable_dynamic_mask", False))
        ),
        # Read with an explicit default rather than the falsy-``or`` idiom the
        # lines above preserve for backwards compatibility: ``or`` cannot
        # distinguish "declared False" from "absent", which for a bool is the
        # whole value.
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
            accel.get("nested_tolerance") or over.get("nested_tolerance", 0.5)
        ),
    }
    # Only forward the axis when one was actually declared. Passing None would be
    # mapped onto ``line_axis=None`` downstream, which the line-based
    # accelerators reject; absent means "keep the accelerator's own default".
    direction = _plain(accel.get("mask_direction")) or over.get("mask_direction")
    if direction is not None:
        resolved["mask_direction"] = direction
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
        max_acceleration: Acceleration at ``t=T``.
        base_acceleration: Acceleration at ``t=0``.
        center_fraction: ACS fraction at low acceleration.
        min_center_fraction: ACS fraction at max acceleration. ``None`` means
            "static" and collapses onto ``center_fraction`` (issue #534: an
            unforwarded value made every rung above R≈12 realise the same mask).
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

    Returns:
        Keyword arguments accepted by ``create_kspace_accelerator``.
    """
    return {
        "max_acceleration": max_acceleration,
        "base_acceleration": base_acceleration,
        "center_fraction": center_fraction,
        "min_center_fraction": (
            min_center_fraction if min_center_fraction is not None else center_fraction
        ),
        "seed": seed,
        "enforce_nested": enforce_nested,
        "nested_tolerance": nested_tolerance,
        **({} if mask_direction is None else {"mask_direction": mask_direction}),
        "acceleration_schedule": schedule_type,
        **(schedule_kwargs or {}),
    }


def accelerator_kwargs_from_config(
    acceleration_config: Any = None,
    overrides: dict[str, Any] | None = None,
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

    Returns:
        ``(default_pattern, accelerator_kwargs)``.
    """
    resolved = resolve_undersampling_kwargs(acceleration_config, overrides)
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
    )


class KSpaceUndersamplingProcess(nn.Module):
    """Deterministic undersampling process for k-space cold diffusion.

    Delegates to KSpaceMaskGenerator to ensure SSOT for sampling patterns.
    Guarantees Nested Property: M_{t+1} subset of M_t.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        max_acceleration: float = 8.0,
        base_acceleration: float = 1.0,
        center_fraction: float = 0.0325,
        min_center_fraction: float | None = None,
        mask_type: str = "variable_density",
        seed: int | None = 42,
        schedule_type: str = "linear",
        schedule_kwargs: dict | None = None,
        prior_channel_range: tuple[int, int] | None = None,
        enable_dynamic_mask: bool = False,
        train_identity_rung: bool = False,
        enforce_nested: bool = False,
        nested_tolerance: float = 0.5,
        mask_direction: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize k-space undersampling process.

        Args:
            num_timesteps: Number of diffusion timesteps.
            max_acceleration: Maximum acceleration factor at t=T.
            base_acceleration: Minimum acceleration factor at t=0.
            center_fraction: ACS fraction at low acceleration (R≈1).
            min_center_fraction: ACS fraction at max acceleration.
                If None, uses center_fraction (static behavior).
            mask_type: Sampling pattern type.
            seed: Random seed for reproducibility.
            schedule_type: Acceleration schedule type.
            schedule_kwargs: Additional schedule parameters.
            prior_channel_range: Optional ``(start, end)`` half-open channel
                slice that is kept *fully sampled* (mask = 1) by ``q_sample``.
                Use for cross-contrast translation where one contrast (e.g.
                T1) is the prior and only the *other* contrast's channels
                should be undersampled. ``end`` is exclusive, mirroring
                Python slicing. ``None`` = legacy uniform behaviour.
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
                accelerator's ``line_axis``.
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

        if prior_channel_range is not None:
            s, e = prior_channel_range
            if not (isinstance(s, int) and isinstance(e, int)) or s < 0 or e <= s:
                raise ValueError(
                    f"prior_channel_range must be a (start, end) pair with "
                    f"0 <= start < end (got {prior_channel_range!r})."
                )
        self.prior_channel_range = prior_channel_range

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
                min_center_fraction=self.min_center_fraction,
                seed=seed,
                schedule_type=schedule_type,
                schedule_kwargs=schedule_kwargs,
                enforce_nested=enforce_nested,
                nested_tolerance=nested_tolerance,
                mask_direction=mask_direction,
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
        predicate is ``PhysicsInformedColdDiffusion._step_reveals_anything``).

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

        # Generate nested masks from SSOT generator
        # generate_batch_masks returns [B, 1, H, W]
        if self.enable_dynamic_mask and self.training:
            # Per-sample pattern randomisation (training only): each sample gets
            # a fresh accelerator seed → a different pattern at the same R, so the
            # model learns to invert undersampling in general rather than one
            # fixed mask. Validation/eval falls through to the fixed-seed path
            # below, keeping metrics reproducible.
            base_masks = self._generate_batch_masks_dynamic(B, t_expanded, (H, W), device)
        else:
            base_masks = self.mask_generator.generate_batch_masks(
                batch_size=B,
                timesteps=t_expanded,
                image_shape=(H, W),
                pattern=self.mask_type,
            ).to(device)

        # Expand mask to match channels for degradation [B, C, H, W]
        channel_masks = self.mask_generator.expand_mask_to_channels(base_masks, C)

        # Apply degradation: x_t = x_0 ⊙ M_t
        x_t = x_start * channel_masks

        # Cross-contrast prior: keep the declared channel range fully
        # sampled. ``base_masks`` stays a single-channel ``[B, 1, H, W]``
        # tensor (the visualisation/loss path uses it as a probe), but the
        # actual degraded tensor ``x_t`` restores the prior channels. This
        # is the surgical fix for the experiment_cross_contrast intent
        # ("T1 fully sampled prior to fill T2/FLAIR k-space"): the
        # broadcast mask multiplied every channel uniformly before this
        # block, so T1 was as undersampled as T2.
        if self.prior_channel_range is not None:
            s, e = self.prior_channel_range
            if e > C:
                raise ValueError(
                    f"prior_channel_range={self.prior_channel_range!r} is out "
                    f"of bounds for k-space tensor with C={C} channels."
                )
            x_t[:, s:e] = x_start[:, s:e]

        if is_5d:
            # Reshape back to 5D [B_orig, D, C, H, W]
            x_t = x_t.view(B_orig, D, C, H, W)
            base_masks = base_masks.view(B_orig, D, 1, H, W)

        return x_t, base_masks

    def _generate_batch_masks_dynamic(
        self,
        batch_size: int,
        timesteps: Tensor,
        image_shape: tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Per-sample randomised undersampling masks for dynamic-mask training.

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
        # Nesting enforcement MUST be off for this loop, and it is a performance
        # cliff rather than a correctness question. ``ColdDiffusionAccelerator``
        # caches its first-drop map on ``(shape, device, seed)``; mutating the
        # seed per sample below makes every sample a cache MISS, so each one
        # rebuilds the whole cascade at ``num_timesteps`` mask evaluations —
        # 28x the work per sample — and leaves a permanent cache entry behind,
        # so the dict grows without bound for the life of the run. Measured at
        # 256x256, batch 2: 11 ms/step -> 334 ms/step, a 30x regression.
        #
        # ``get_acceleration_mask`` already documents that enforcement is meant
        # to apply to the fixed-seed cascade ONLY ("rebuilding per sample would
        # cost T mask evaluations per item"), but nothing implemented that; it
        # simply recomputed. Restored the same way the seed is, so the mutation
        # is invisible outside this loop.
        original_enforce = getattr(accelerator, "enforce_nested", False)
        masks = []
        try:
            accelerator.enforce_nested = False
            for i in range(batch_size):
                inner.seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                masks.append(
                    self.mask_generator.generate_acceleration_mask(
                        int(timesteps[i].item()),
                        image_shape,
                        pattern=self.mask_type,
                    )
                )
        finally:
            inner.seed = original_seed
            accelerator.enforce_nested = original_enforce
        return torch.stack(masks, dim=0).to(device)

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

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
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
        *,
        kspace_log_scaled: bool,
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
                deterministic function of (measurement, mask, sampler_seed):
                same seed ⇒ identical reconstruction, every validation image
                sees the same noise realization (the C7 exchangeability story
                depends on this). ``None`` ⇒ nondeterministic σ>0 draws.
            selection_rule: which lines each reverse step reveals. Only
                ``"fixed"`` (the process' fixed-seed mask cascade) exists;
                validated so a stochastic rule can't slip in unvalidated.
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
            # discards the data-consistency work done on the observed support and
            # steps ``t - 1`` under a strided schedule. It is retained verbatim for
            # Bansal Algorithm-2 reproduction, so the honest move is to say so once
            # at construction rather than to change it or to leave the default silent.
            logger.warning(
                "reverse_sampling_mode='additive' is the legacy Bansal Alg-2 loop and "
                "is correct ONLY on a nested mask cascade: it double-writes any bin "
                "that leaves and re-enters the support, discards data consistency on "
                "the observed support, and steps t-1 under a strided schedule. Use "
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
        self.model = model
        self.num_timesteps = num_timesteps
        self.dc_method = dc_method
        self.dc_weight = dc_weight
        self.sampling_steps = sampling_steps or num_timesteps
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

        # The t=0-terminal keying only delivers its terminal call when the
        # schedule actually reaches 0. On a ladder whose floor is 1 the mode
        # would run, report success, and call the model at t=1 exactly like the
        # convention it exists to replace -- an inert knob, not an error
        # (pitfall 15). The floor is 0 when base_acceleration > 1 or when
        # `undersampling.train_identity_rung` opens the identity rung.
        if self.reverse_mode in CURRENT_STEP_KEYED_REVERSE_MODES:
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

    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        measurement: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Single reverse diffusion step with data consistency.

        Args:
            x_t: Current k-space estimate [B, C, H, W]
            t: Current timestep [B]
            measurement: Acquired k-space [B, C, H, W]
            mask: Sampling mask [B, 1, H, W]

        Returns:
            x_t_minus_1: Less-degraded k-space estimate
        """
        # 1. Predict clean k-space x_0
        x_0_pred = self.model(x_t, t)

        # Handle tuple output (normalized_pred, scale)
        if isinstance(x_0_pred, tuple):
            x_0_pred = x_0_pred[0]

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
            # ``SoftDataConsistency`` (the layer 'soft'/'noise_adjusted' build)
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
        if t.min() <= 0:
            # At t=0, return final prediction
            return x_0_pred

        # Degrade prediction to t-1 level
        t_prev = torch.clamp(t - 1, min=0)

        # [STABILIZATION FIX] Cold Diffusion Restoration Formula
        # Formula: x_{t-1} = x_t - D(x_0, t) + D(x_0, t-1)
        # 1. Generate masks for current and previous state
        _, mask_t = self.process.q_sample(x_0_pred, t)
        _, mask_t_prev = self.process.q_sample(x_0_pred, t_prev)

        # 2. Compute degradation delta: D(x_0, t-1) - D(x_0, t)
        # This represents the k-space lines that are "clean" at t-1 but "corrupt" at t.
        # mask_t_prev has MORE 1s than mask_t (Nested Property: mask_t subset of mask_t_prev)
        mask_recovered = torch.clamp(mask_t_prev.float() - mask_t.float(), min=0.0)

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
        # Two further limitations of this legacy branch, both fixed in
        # ``_sample_replace_freeze`` and both deliberately NOT changed here because
        # ``additive`` exists to reproduce Bansal Algorithm 2 verbatim:
        #   * everything computed on the observed support -- hard/soft/learned data
        #     consistency, clipping, sampler noise -- is discarded, since only
        #     ``x_0_pred * mask_recovered`` is ever written. A stale ``x_t`` value on
        #     an observed bin survives even when DC would correct it.
        #   * ``t_prev = t - 1`` while ``sample`` walks a STRIDED schedule, so each
        #     step reveals one level's worth and whole bands between scheduled levels
        #     are never revealed.
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
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Full reverse diffusion sampling.

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
        device = measurement.device
        B = measurement.shape[0]

        # C6: fresh noise stream per call — same (measurement, mask, seed) in,
        # same reconstruction out, regardless of how many samples ran before.
        self._reseed_sampler_generator(seed_offset)

        # Start from measurement (most degraded state)
        x = measurement.clone()

        trajectory = [x] if return_trajectory else None

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

        if self.reverse_mode == "replace_freeze":
            return self._sample_replace_freeze(
                measurement, mask, timestep_schedule, return_trajectory
            )

        if self.reverse_mode in ("replace_freeze_dc", "replace_freeze_dc_t0"):
            return self._sample_replace_freeze_dc(
                measurement, mask, timestep_schedule, return_trajectory
            )

        # Legacy additive Bansal-Alg-2 reverse loop (reverse_mode="additive").
        for t_idx in timestep_schedule:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            x = self.p_sample(x, t, measurement, mask)

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

        # Observed support (exact, frozen). Prefer the sampling mask; else the
        # nonzero measurement entries.
        if mask is not None:
            obs = (mask > 0).float()
        else:
            # Observation is a property of the COEFFICIENT, not of Re and Im
            # separately: an observed coefficient whose imaginary part happens
            # to be 0 must not be treated as half-unobserved (that would leave
            # the model's Im in place under hard DC). Pair, test, expand back.
            _obs_pair = (paired_magnitude(measurement) > eps).float()
            obs = (
                _obs_pair
                if torch.is_complex(measurement)
                else _obs_pair.repeat_interleave(2, dim=-3)
            )

        # Per-sample magnitude ceiling from the measured coefficients, FIXED over
        # the trajectory: max|written| ≤ ratio · max|observed| ⇒ ‖x‖∞ bounded.
        ceil = self._magnitude_ceiling(measurement, obs, eps)

        committed = obs.clone()
        trajectory = [x.clone()] if return_trajectory else None

        skipped = 0
        terminal: int | None = None
        for i, t_idx in enumerate(timestep_schedule):
            # Skip provably-inert steps (issue #535). This variant hard-pins the
            # observed support unconditionally, so a step that reveals nothing
            # leaves ``x`` bit-for-bit unchanged. ``committed`` is seeded with
            # ``obs`` here, so the reveal test needs no separate obs term.
            if not self._step_reveals_anything(
                i, timestep_schedule, committed, torch.zeros_like(obs), x, B, device
            ):
                skipped += 1
                if return_trajectory:
                    trajectory.append(x.clone())
                continue

            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            terminal = t_idx
            x0 = self.model(x, t)
            if isinstance(x0, tuple):
                x0 = x0[0]

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

            # Phase-preserving magnitude clamp to the fixed ceiling. Radial
            # (one shared scale per complex coefficient), so this comment is now
            # literally true -- the elementwise predecessor bounded a SQUARE and
            # rotated phase up to 45 deg (issue #1281).
            x0 = clamp_to_magnitude_ceiling(x0, ceil, eps)

            # Reveal newly-uncovered lines, write once, then freeze. On the FINAL
            # step reveal EVERY remaining line: the reconstruction target is fully
            # sampled, but ``q_sample(x0, t=0)`` returns the base-acceleration mask
            # (support 1/base_acceleration), so relying on it leaves
            # 1 - 1/base_acceleration of k-space permanently zero
            # (base_acceleration=2.0 => 50% dropped => a hard low-pass "DC blob";
            # see docs/experiment_11_kspace_cold_diffusion.rst). The revealed lines
            # carry the model's own (full-target-supervised) prediction, never the
            # measurement, so this does not leak ground truth.
            # Observed lines are already inside ``committed`` here, so the
            # owner is called with an empty observed support -- the same
            # arguments ``_step_reveals_anything`` is given above, which is what
            # keeps the skip gate and the write from disagreeing.
            reveal = self._reveal_support(
                i, timestep_schedule, committed, torch.zeros_like(committed), x0, B, device
            )
            x = x * (1.0 - reveal) + x0 * reveal
            committed = torch.clamp(committed + reveal, max=1.0)

            if return_trajectory:
                trajectory.append(x.clone())

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

    def _step_reveals_anything(
        self,
        i: int,
        timestep_schedule: list[int],
        committed: Tensor,
        obs: Tensor,
        x: Tensor,
        batch: int,
        device: torch.device,
    ) -> bool:
        """Whether reverse step ``i`` would write any coefficient.

        The final scheduled step always reveals everything still uncommitted, so it
        is never inert. An intermediate step reveals only where its ``mask_next``
        falls outside ``committed | obs``; at low acceleration the observed support
        already covers every ``mask_next``, so the answer is False for most of the
        trajectory (issue #535).
        """
        return bool(
            self._reveal_support(i, timestep_schedule, committed, obs, x, batch, device).any()
        )

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

        The single owner of the reveal arithmetic: the loop writes what this
        returns and ``_step_reveals_anything`` skips when it is empty, so the
        skip gate and the write cannot disagree about a step (non-negotiable
        17). They were two copies of the expression, which is the shape #2067
        is about at the level above.

        The terminal step always takes every remaining unobserved coefficient:
        the reconstruction target is fully sampled, and keying it off a mask
        would leave whatever that mask excludes permanently zero.
        """
        n = len(timestep_schedule)
        if i + 1 >= n:
            return (1.0 - committed) * (1.0 - obs)
        key_t = (
            timestep_schedule[i]
            if self.reverse_mode in CURRENT_STEP_KEYED_REVERSE_MODES
            else timestep_schedule[i + 1]
        )
        _, key_mask = self.process.q_sample(
            x, torch.full((batch,), key_t, device=device, dtype=torch.long)
        )
        return torch.clamp(key_mask.float() * (1.0 - committed) * (1.0 - obs), 0.0, 1.0)

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
        # Confine the DC layer's changes to the observed support; the reveal path
        # owns the unobserved lines.
        return x0 * (1.0 - obs) + dc_out * obs

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

        if mask is not None:
            obs = (mask > 0).float()
        else:
            # Observation is a property of the COEFFICIENT, not of Re and Im
            # separately: an observed coefficient whose imaginary part happens
            # to be 0 must not be treated as half-unobserved (that would leave
            # the model's Im in place under hard DC). Pair, test, expand back.
            _obs_pair = (paired_magnitude(measurement) > eps).float()
            obs = (
                _obs_pair
                if torch.is_complex(measurement)
                else _obs_pair.repeat_interleave(2, dim=-3)
            )

        ceil = self._magnitude_ceiling(measurement, obs, eps)

        # Observed lines are OWNED by the per-step DC update (not the reveal path),
        # so seed ``committed`` with zeros and gate reveals away from ``obs``.
        committed = torch.zeros_like(obs)
        trajectory = [x.clone()] if return_trajectory else None

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
            if self.dc_method == "hard" and not self._step_reveals_anything(
                i, timestep_schedule, committed, obs, x, B, device
            ):
                skipped += 1
                if return_trajectory:
                    trajectory.append(x.clone())
                continue

            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            terminal = t_idx
            x0 = self.model(x, t)
            if isinstance(x0, tuple):
                x0 = x0[0]

            # dc_method-aware data consistency on the observed support (vs the
            # hard freeze in replace_freeze): a soft/learned method denoises here.
            x0 = self._apply_observed_dc(x0, measurement, obs)

            # C6: optional reverse-step noise — after DC, off the observed
            # support, BEFORE the clamp so boundedness holds for every σ.
            # (Skipped inert steps above make no model call and hence no draw.)
            if self.sampler_sigma > 0:
                x0 = inject_reverse_step_noise(
                    x0, self.sampler_sigma, self._active_sampler_generator(), exclude_support=obs
                )

            # Phase-preserving magnitude clamp to the fixed ceiling, AFTER the DC
            # blend, so the boundedness guarantee holds for every dc_method.
            # Radial, per complex coefficient (issue #1281).
            x0 = clamp_to_magnitude_ceiling(x0, ceil, eps)

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
            reveal = self._reveal_support(
                i, timestep_schedule, committed, obs, x0, B, device
            )
            # Observed support (re-blended each step) + freshly revealed lines.
            own = torch.clamp(reveal + obs, 0.0, 1.0)
            x = x * (1.0 - own) + x0 * own
            committed = torch.clamp(committed + reveal, max=1.0)

            if return_trajectory:
                trajectory.append(x.clone())

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


__all__ = [
    "VALID_SELECTION_RULES",
    "KSpaceUndersamplingProcess",
    "PhysicsInformedColdDiffusion",
    "inject_reverse_step_noise",
    "paired_complex",
    "resolve_undersampling_kwargs",
    "validate_sampler_determinism",
]
