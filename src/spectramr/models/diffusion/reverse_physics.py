"""Physics options of the bounded reverse loops, and their validation.

Five ``model.model_kwargs`` keys select the physics-grounded pieces of the
reverse process (``docs/explanation/kspace_reverse_processes.md``, Algorithms
D-H). Each is validated here, once, and the generator, the sampler and the
``ConfigHealthChecker`` all call :func:`resolve_reverse_physics`, so the build
and the audit cannot disagree about a value.

``reverse_decay_envelope`` (F)
    ``off`` | ``per_sample`` | ``batch_pooled`` | ``per_sample_radial`` |
    ``batch_pooled_radial``. Fits the power-law decay of the measured bins and
    bounds every unmeasured bin by it (``reverse_decay``). A bin is one of the
    mask's lines (a row or a column, whichever axis the mask removes lines
    along), or with ``_radial`` a shell of radial ``|k|``, which a 2-D
    point mask (gaussian, variable density, radial, spiral) samples evenly where
    it samples no row whole (``reverse_decay_axis``). ``batch_pooled`` fits one
    slope across the batch with a per-sample amplitude, for rungs whose few
    measured bins leave a per-sample slope outside its valid range.
``reverse_projection`` (E, H)
    ``none`` | ``spirit`` | ``joint_tv``. A projection applied to each estimate
    after data consistency and before the energy bound: coil self-consistency
    (``reverse_spirit``) or joint total variation along the phase-encode
    direction (``reverse_joint_tv``).
``reverse_dc_trust`` (G)
    ``measured`` | ``fitted_decay``. With ``dc_method: noise_adaptive``,
    ``fitted_decay`` takes the Wiener signal power from the decay envelope
    instead of ``|y|^2`` and bounds each measured bin's departure from the
    measurement to ``3 sigma`` (``reverse_trust``).
``reverse_support_radius`` (D)
    Required by ``reverse_sampling_mode: support_ordered_dc`` and rejected by
    every other mode: the reach, in phase-encode lines, within which an
    unmeasured line counts as supported by a measured one.
``reverse_dc``
    ``per_step`` | ``unconstrained``. Under ``replace_freeze_dc`` (and ``_t0``)
    the measured lines are rewritten every step by ``dc_method`` applied to the
    x0 estimate; ``unconstrained`` writes the raw estimate there instead, so the
    measurement enters only as the initial state and the model's next input
    drifts from it: the control for a claim that per-step data consistency
    helps. Every other loop refuses it: ``additive`` discards the measured-line
    update until its last step, and the remaining loops pin or blend by
    construction.

None of the four changes the energy bound of the loop it runs in; each only
removes energy or moves an estimate before the bound is applied. Plain
``additive`` and hard-freeze ``replace_freeze`` do not call the hooks, so a key
set under them would be unread and raises. ``additive`` also bounds nothing, so
``reverse_clip_ratio`` raises there too unless ``reverse_integrity_check`` reads it
as the ceiling it reports against (#2428).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bansal's additive loop, the one reverse mode that applies no magnitude bound.
UNBOUNDED_REVERSE_MODE = "additive"

#: The freeze loop that hard-pins the measured lines whatever ``dc_method`` says.
HARD_PINNED_REVERSE_MODE = "replace_freeze"

#: The support-ordered commitment loop (Algorithm D).
SUPPORT_ORDERED_REVERSE_MODE = "support_ordered_dc"

#: The loops whose measured lines are rewritten each step, so ``reverse_dc`` changes the trajectory.
UNCONSTRAINABLE_REVERSE_MODES: frozenset[str] = frozenset({"replace_freeze_dc", "replace_freeze_dc_t0"})

#: Reverse modes whose loops call the projection and decay-bound hooks.
HOOKED_REVERSE_MODES: frozenset[str] = frozenset(
    {
        "additive_normalized",
        "replace_freeze_dc",
        "replace_freeze_dc_t0",
        "reveal_refine_dc",
        "single_call_dc",
        SUPPORT_ORDERED_REVERSE_MODE,
    }
)

#: ``reverse_decay_envelope`` -> (one slope pooled across the batch, the axis the fit bins over).
#: ``line`` is the mask's own line axis, resolved per sampler from its process
#: (``reverse_decay_axis.LINE_DECAY_AXES``); it was a fixed ``row`` until #2160.
LINE_ENVELOPE_AXIS = "line"
DECAY_ENVELOPE_SHAPES: dict[str, tuple[bool, str]] = {
    "per_sample": (False, LINE_ENVELOPE_AXIS),
    "batch_pooled": (True, LINE_ENVELOPE_AXIS),
    "per_sample_radial": (False, "radial"),
    "batch_pooled_radial": (True, "radial"),
}

VALID_DECAY_ENVELOPES: frozenset[str] = frozenset({"off", *DECAY_ENVELOPE_SHAPES})
VALID_PROJECTIONS: frozenset[str] = frozenset({"none", "spirit", "joint_tv"})
VALID_DC_TRUST: frozenset[str] = frozenset({"measured", "fitted_decay"})
VALID_REVERSE_DC: frozenset[str] = frozenset({"per_step", "unconstrained"})

#: ``model_kwargs`` key of each field, for error messages and the audit's yaml_keys.
REVERSE_PHYSICS_KEYS: dict[str, str] = {
    "decay_envelope": "reverse_decay_envelope",
    "projection": "reverse_projection",
    "dc_trust": "reverse_dc_trust",
    "support_radius": "reverse_support_radius",
    "reverse_dc": "reverse_dc",
}


@dataclass(frozen=True)
class ReversePhysics:
    """Resolved physics options of one sampler."""

    decay_envelope: str = "off"
    projection: str = "none"
    dc_trust: str = "measured"
    support_radius: int | None = None
    reverse_dc: str = "per_step"


def _choice(key: str, value: object, valid: frozenset[str], default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{key}={value!r} is not one of {sorted(valid)}.")
    return value


def _support_radius(reverse_mode: str, value: object) -> int | None:
    key = REVERSE_PHYSICS_KEYS["support_radius"]
    if reverse_mode != SUPPORT_ORDERED_REVERSE_MODE:
        if value is not None:
            raise ValueError(
                f"{key} is read only by reverse_sampling_mode="
                f"{SUPPORT_ORDERED_REVERSE_MODE!r}; it was set under {reverse_mode!r}."
            )
        return None
    if value is None:
        raise ValueError(
            f"reverse_sampling_mode={SUPPORT_ORDERED_REVERSE_MODE!r} requires {key}, "
            "the reach in phase-encode lines within which a line counts as supported."
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be an integer >= 1, got {value!r}.")
    return value


def resolve_reverse_physics(
    reverse_mode: str,
    dc_method: str,
    *,
    decay_envelope: object = None,
    projection: object = None,
    dc_trust: object = None,
    support_radius: object = None,
    reverse_dc: object = None,
) -> ReversePhysics:
    """Validate the physics keys against the mode and ``dc_method``; raise on any mismatch."""
    resolved = ReversePhysics(
        decay_envelope=_choice(
            REVERSE_PHYSICS_KEYS["decay_envelope"], decay_envelope, VALID_DECAY_ENVELOPES, "off"
        ),
        projection=_choice(
            REVERSE_PHYSICS_KEYS["projection"], projection, VALID_PROJECTIONS, "none"
        ),
        dc_trust=_choice(REVERSE_PHYSICS_KEYS["dc_trust"], dc_trust, VALID_DC_TRUST, "measured"),
        support_radius=_support_radius(reverse_mode, support_radius),
        reverse_dc=_choice(
            REVERSE_PHYSICS_KEYS["reverse_dc"], reverse_dc, VALID_REVERSE_DC, "per_step"
        ),
    )
    if resolved.reverse_dc == "unconstrained":
        if reverse_mode not in UNCONSTRAINABLE_REVERSE_MODES:
            raise ValueError(
                f"reverse_dc='unconstrained' is read only by reverse_sampling_mode in "
                f"{sorted(UNCONSTRAINABLE_REVERSE_MODES)}; under {reverse_mode!r} the "
                "control would still be data-consistent or would differ only at its last step."
            )
        if resolved.dc_trust != "measured":
            raise ValueError(
                f"reverse_dc='unconstrained' writes no data consistency, so "
                f"reverse_dc_trust={resolved.dc_trust!r} would be unread."
            )
    engaged = [
        REVERSE_PHYSICS_KEYS[name]
        for name, off in (
            ("decay_envelope", "off"),
            ("projection", "none"),
            ("dc_trust", "measured"),
        )
        if getattr(resolved, name) != off
    ]
    if engaged and reverse_mode not in HOOKED_REVERSE_MODES:
        raise ValueError(
            f"{', '.join(engaged)} {'is' if len(engaged) == 1 else 'are'} not read by "
            f"reverse_sampling_mode={reverse_mode!r}. Modes that read them: "
            f"{sorted(HOOKED_REVERSE_MODES)}."
        )
    if reverse_mode == HARD_PINNED_REVERSE_MODE and dc_method != "hard":
        raise ValueError(
            f"reverse_sampling_mode={HARD_PINNED_REVERSE_MODE!r} pins the measured lines to "
            f"the measurement and never reads dc_method={dc_method!r}; use "
            "'replace_freeze_dc' to run that data consistency in the same loop."
        )
    if resolved.dc_trust == "fitted_decay":
        if dc_method != "noise_adaptive":
            raise ValueError(
                "reverse_dc_trust='fitted_decay' sets the Wiener signal power of "
                f"dc_method='noise_adaptive'; this arm declares dc_method={dc_method!r}."
            )
        if resolved.decay_envelope == "off":
            raise ValueError(
                "reverse_dc_trust='fitted_decay' takes its signal power from the decay "
                "envelope; set reverse_decay_envelope to 'per_sample' or 'batch_pooled'."
            )
        if DECAY_ENVELOPE_SHAPES[resolved.decay_envelope][1] != LINE_ENVELOPE_AXIS:
            raise ValueError(
                "reverse_dc_trust='fitted_decay' builds a separable S(|k_line|) q(k_across) "
                "signal power from a line envelope; "
                f"reverse_decay_envelope={resolved.decay_envelope!r} "
                "fits shells of radial |k|. Use 'per_sample' or 'batch_pooled'."
            )
    return resolved


def resolve_reverse_physics_kwargs(
    reverse_mode: str, dc_method: str, model_kwargs: dict[str, object]
) -> ReversePhysics:
    """:func:`resolve_reverse_physics` over the ``model_kwargs`` spelling of the keys."""
    if (
        reverse_mode == UNBOUNDED_REVERSE_MODE
        and "reverse_clip_ratio" in model_kwargs
        and (model_kwargs.get("reverse_integrity_check") or "off") == "off"
    ):
        raise ValueError(
            f"reverse_clip_ratio is not read by reverse_sampling_mode={UNBOUNDED_REVERSE_MODE!r}, "
            "whose loop applies no magnitude bound. Remove it, or use 'additive_normalized' "
            "for the bounded additive loop."
        )
    if model_kwargs.get("reverse_dc") == "unconstrained" and model_kwargs.get(
        "dc_passthrough_center_size"
    ):
        raise ValueError(
            "reverse_dc='unconstrained' with dc_passthrough_center_size pins the centre "
            "patch to the measurement after the loop; drop one of them."
        )
    return resolve_reverse_physics(
        reverse_mode,
        dc_method,
        **{field: model_kwargs.get(key) for field, key in REVERSE_PHYSICS_KEYS.items()},
    )


__all__ = [
    "DECAY_ENVELOPE_SHAPES",
    "LINE_ENVELOPE_AXIS",
    "HARD_PINNED_REVERSE_MODE",
    "HOOKED_REVERSE_MODES",
    "REVERSE_PHYSICS_KEYS",
    "SUPPORT_ORDERED_REVERSE_MODE",
    "UNBOUNDED_REVERSE_MODE",
    "UNCONSTRAINABLE_REVERSE_MODES",
    "VALID_DC_TRUST",
    "VALID_DECAY_ENVELOPES",
    "VALID_PROJECTIONS",
    "VALID_REVERSE_DC",
    "ReversePhysics",
    "resolve_reverse_physics",
    "resolve_reverse_physics_kwargs",
]
