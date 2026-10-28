"""Themed-strategy → required-component SSOT and guardrail decision logic.

Some strategy keys route to a *generic* base strategy
(`ReconstructionTrainingStrategy`, `DiffusionTrainingStrategy`,
`CycleBlochStrategy`) yet advertise paradigm-specific mathematics by their
name (e.g. ``tropical_mrf``, ``koopman_fmri``, ``sheaf_kspace``). The
generic base does not branch on the key, so unless the YAML wires a
*registered component* that actually encodes the advertised mathematics —
a loss or a model — the run silently degrades to a vanilla baseline. That
is CLAUDE.md pitfall #9 ("silent fallbacks are forbidden") institutionalized
across many keys (see the 2026-05-28 PR-0 façade triage,
``TODO/pr0_facade_strategy_triage_2026_05_28.md``).

This module is the single source of truth for *which registered loss or
model* each themed key must wire, plus a pure decision function the audit
ladder calls. Adding a new themed-generic alias to
``STRATEGY_CLASS_PATHS`` without an entry here is fine; adding an entry
here forces every YAML using that key to wire one of the listed
components, or fail the Tier-1 audit.

Design note: kept as a standalone module (not another method on the
6700-line ``config_health_checker.py``) so the SSOT map and its pure
logic are independently testable and do not bloat the god-object.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThemedComponentStatus:
    """Result of the themed-component guardrail decision.

    Attributes:
        applicable: True if ``strategy_key`` is a themed generic-base alias
            governed by this guardrail (False ⇒ the key is exempt — an
            honest generic key like ``reconstruction`` or an unrelated key).
        satisfied: True if the config wires at least one required component.
        required: The ``{"losses": {...}, "models": {...}}`` requirement
            (empty when not applicable).
        matched: The component name(s) that satisfied the requirement.
    """

    applicable: bool
    satisfied: bool
    required: dict[str, set[str]] = field(default_factory=dict)
    matched: tuple[str, ...] = ()


# SSOT: themed key → registered components that encode its advertised math.
# A config using the key must wire at least ONE listed loss OR the listed
# model. Group-C keys (already legitimately wired before PR-0) are included
# so the guardrail documents the real wiring; Group-A keys are the ones
# completed by the 2026-05-28 façade-implementation pass.
THEMED_GENERIC_COMPONENTS: dict[str, dict[str, set[str]]] = {
    # ---- Group C: pre-existing, legitimately wired (math in loss/model) ----
    "spectral_triple": {
        "losses": {"spectral_triple_intertwining", "connes_intertwining", "morita_morphism"},
        "models": set(),
    },
    "flow_matching": {
        "losses": {
            "rectified_flow",
            "biophysical_flow",
            "conditional_flow_matching",
            "ot_conditional_flow_matching",
        },
        "models": {"divergence_free_flow"},
    },
    "rectified_flow": {"losses": {"rectified_flow"}, "models": {"latent_flow"}},
    "tissue_diffusion_pretrain": {
        "losses": {
            "tissue_parameter_physics_constraint",
            "tissue_parameter_prior_smoothness",
        },
        "models": {"tissue_parameter_diffusion"},
    },
    "heisenberg_phase": {"losses": set(), "models": {"heisenberg_recon_unet"}},
    "hyperbolic_cardiac": {"losses": set(), "models": {"hyperbolic_cine_unet"}},
    "symplectic_bloch": {"losses": set(), "models": {"bloch_cycle_network"}},
    # ---- Group A: completed 2026-05-28 (loss wraps an existing primitive) ----
    "tropical_mrf": {"losses": {"tropical_semiring_consistency"}, "models": set()},
    "tropical_quantitative_maps": {
        "losses": {"tropical_semiring_consistency"},
        "models": set(),
    },
    "sheaf_kspace": {"losses": {"sheaf_consistency"}, "models": set()},
    "sheaf_multi_contrast": {"losses": {"sheaf_consistency"}, "models": set()},
    "koopman_fmri": {"losses": {"koopman_linearity"}, "models": set()},
    "hodge_motion": {"losses": {"hodge_decomposition"}, "models": set()},
    "hjb_trajectory": {"losses": {"hjb_residual"}, "models": set()},
    "hjb_waveform": {"losses": {"hjb_residual"}, "models": set()},
    "frontdoor_federated": {"losses": {"frontdoor_criterion"}, "models": set()},
    "frontdoor_scanner": {"losses": {"frontdoor_criterion"}, "models": set()},
    "stein_federated": {"losses": {"stein_discrepancy"}, "models": set()},
    "fisher_rao_flow": {"losses": {"fisher_rao_geodesic"}, "models": set()},
}


def _lower(names: set[str]) -> set[str]:
    return {n.lower() for n in names}


def themed_component_status(
    strategy_key: str | None,
    declared_losses: set[str],
    model_type: str | None,
) -> ThemedComponentStatus:
    """Decide whether a themed generic-base key wires its required component.

    Args:
        strategy_key: The resolved strategy short-name (``training_mode``).
        declared_losses: Registered loss names declared anywhere in the
            config's loss blocks.
        model_type: The config's ``model.model_type``.

    Returns:
        A :class:`ThemedComponentStatus`. ``applicable=False`` means the key
        is exempt (not a themed generic-base alias).
    """
    req = THEMED_GENERIC_COMPONENTS.get((strategy_key or "").strip())
    if req is None:
        return ThemedComponentStatus(applicable=False, satisfied=True)

    losses_req = _lower(req.get("losses", set()))
    models_req = _lower(req.get("models", set()))
    matched = sorted({loss for loss in declared_losses if loss.lower() in losses_req})
    model_ok = model_type is not None and model_type.lower() in models_req
    if model_ok:
        matched.append(str(model_type))
    return ThemedComponentStatus(
        applicable=True,
        satisfied=bool(matched),
        required=req,
        matched=tuple(matched),
    )


__all__ = [
    "THEMED_GENERIC_COMPONENTS",
    "ThemedComponentStatus",
    "themed_component_status",
]
