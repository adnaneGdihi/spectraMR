"""Frozen reference implementations of the PRE-SSOT weight semantics.

**This module is scaffolding. Delete it once the migration is complete.** Nothing in a
training path may import it.

It exists for exactly one purpose: to prove that switching to
:mod:`spectramr.models.losses.weights` is a numeric **no-op**. The migration codemod uses
it to compute the weight each arm was *actually* training with, writes that value into
the YAML, and then the corpus test asserts
``new_table.weight(n) == legacy_effective_weight(cfg, n, family=f)`` for every arm.

There was never ONE legacy weight — there were three, and which one an arm got depended
on which computer its strategy happened to construct:

``computer``
    ``BaseLossComputer._get_loss_weight``. Scanned sections in a fixed order via
    ``model_dump()``, so **schema defaults counted as declarations** (this is why
    ``lambda_l1``'s 10.0 default always won over a declared ``image_losses`` weight of
    1.0). Fell back to 1.0.

``recon``
    ``resolve_static_loss_weight`` (the reconstruction/diffusion computer override).
    Eight lookup paths -- half of them ``objectives.*``, a block v6.0 **removed** -- then
    its own 18-entry default table.

``inline``
    ``BaseTrainingStrategy._get_loss_weight``. Only ever looked at
    ``losses.reconstruction.lambda_*``, then a *third*, 9-entry default table.

``folding``
    ``loss_folding.declared_loss_weights``. The inline (type-B) strategies compute their
    objective directly and fold the declarative lists back in, reading the weight straight
    off the ``*_losses[]`` entry. For these arms the list weight **is** the lambda, not a
    mere enable-gate.

The tables disagree with each other (``log_spectral`` is 1.0 in one and 0.1 in
another), which is the whole reason this refactor exists.

Picking the family is not optional colour: ``hfen`` on a folding arm trained at its
declared 0.2, and 0.0 under ``computer`` (no ``lambda_hfen`` field, so the schema default
answers). Ask the wrong one and you conclude a live loss was never computed. Use
:func:`legacy_family_for` — never re-derive the family at a call site.
"""

from __future__ import annotations

from typing import Any, Literal

LegacyFamily = Literal["computer", "recon", "inline", "folding"]

#: Strategies whose computer is ``unified_diffusion_reconstruction`` and therefore
#: overrides ``_get_loss_weight`` with ``resolve_static_loss_weight`` (the ``recon``
#: family); every other computer inherits ``BaseLossComputer._get_loss_weight``
#: (the ``computer`` family).
RECON_FAMILY_MODES: frozenset[str] = frozenset(
    {
        "diffusion",
        "cold_diffusion",
        "kspace_cold_diffusion",
        "graph_cold_diffusion",
        "reconstruction",
    }
)

#: Strategy keys that override ``_compute_losses_impl`` and fold the declarative lists in
#: through ``loss_folding.fold_builder_image_losses`` (directly, or by inheriting a caller).
#: For these the ``*_losses[].weight`` entry is the effective lambda.
#:
#: ``reconstruction`` is deliberately ABSENT: it *defines* ``_apply_builder_image_losses``
#: for the inline strategies to inherit but never calls it — its own objective goes through
#: ``UnifiedReconstructionLossComputer`` (the ``recon`` family). A source scan that does not
#: exclude the defining method counts it by mistake.
#:
#: Re-derived from the strategy registry by
#: ``tests/experiments/test_loss_weight_no_op_migration.py::test_folding_strategy_set_matches_the_source``,
#: which fails if a strategy adopts the seam without landing here.
#:
#: Five entries were added 2026-08-09 after that re-derivation went red: they had
#: adopted the seam without landing here, which is precisely the drift it exists to
#: catch. Each was confirmed to CALL ``_apply_builder_image_losses``, not merely
#: inherit the definition, so none is the ``reconstruction`` false positive above.
#:
#: Two live arms were classified ``computer`` as a result, and the damage was NOT
#: the 0.0 the incidents above produced -- ``computer`` answers with whatever
#: ``lambda_<n>`` defaults to, which is a different number per term:
#:
#: * ``lcah_recon_multifield`` declares ``l1`` at 1.0; ``lambda_l1`` defaults to
#:   **10.0**, so its only objective term was weighted 10x what the arm asked for.
#: * ``dlbae_brain_5field`` declares ``multifield_data_consistency`` at 1.0, and
#:   that lambda also defaults to 1.0 -- so it was unaffected **by coincidence**,
#:   not by design. Do not read one clean arm as evidence the set was adequate.
FOLDING_STRATEGIES: frozenset[str] = frozenset(
    {
        "acq_hypernetwork",
        "bloch_field",
        "bloch_synth",
        "confluence",
        "cross_field_translation",
        "dispersion_bloch_ae",
        "field_cocycle",
        "field_conditioned_inr",
        "field_fno",
        "field_wiener",
        "koopman_field",
        "monotone_field",
        "mrs_quantification",
        "perfusion_kinetic",
        "phase_contrast_flow",
        "scattering_besov",
        "ulf_redegrad_tta",
    }
)


def legacy_family_for(*, strategy_class: str | None, training_mode: str | None) -> LegacyFamily:
    """Which legacy resolver an arm's loss path ACTUALLY reached. The only selector.

    Both the migration codemod and the no-op proof call this. When they each rolled their
    own, they rolled the same bug — keying off ``training_mode`` alone, never selecting
    ``folding`` — so the proof agreed with the migration it was supposed to audit and 52
    arms were disabled down to an empty objective while the suite stayed green.

    ``strategy_class`` is optional in the schema; the factory falls back to
    ``training_mode`` as the strategy key, so this does too.
    """
    if (strategy_class or training_mode) in FOLDING_STRATEGIES:
        return "folding"
    return "recon" if training_mode in RECON_FAMILY_MODES else "computer"


#: ``BaseLossComputer._get_loss_weight`` — the section scan order (first hit wins).
_COMPUTER_SECTIONS: tuple[str, ...] = (
    "reconstruction",
    "latent",
    "adversarial",
    "diffusion",
    "gan",
    "physics",
    "registration",
    "ssl",
    "spatial",
    "evidential",
)

#: ``unified_diffusion_reconstruction._DEFAULT_LOSS_WEIGHTS``.
_RECON_DEFAULTS: dict[str, float] = {
    "l1": 1.0,
    "l2": 1.0,
    "mse": 1.0,
    "kspace": 1.0,
    "log_spectral": 1.0,
    "perceptual": 0.1,
    "adversarial": 0.01,
    "kl_divergence": 0.0001,
    "vq_loss": 1.0,
    "pinn": 0.1,
    "complex_l1": 0.1,
    "background_suppression": 1.0,
    "frequency_weighted_l1_kspace": 1.0,
    "energy_conservation": 1.0,
    "hfen": 1.0,
    "frequency_domain": 0.1,
    "diffusion": 1.0,
}

#: ``strategies/base.py::_get_loss_weight`` — a THIRD table, disagreeing with the above.
_INLINE_DEFAULTS: dict[str, float] = {
    "l1": 1.0,
    "l2": 1.0,
    "mse": 1.0,
    "kspace": 1.0,
    "log_spectral": 0.1,  # NB: 1.0 in _RECON_DEFAULTS
    "perceptual": 0.1,
    "adversarial": 0.01,
    "kl_divergence": 0.0001,
    "vq_loss": 1.0,
}

#: ``resolve_static_loss_weight`` path list, in order. ``objectives.*`` is dead config
#: (v6.0 removed the block) but the lookup still consulted it first.
_RECON_PATHS: tuple[str, ...] = (
    "objectives.reconstruction.lambda_{n}",
    "objectives.{n}.weight",
    "losses.reconstruction.lambda_{n}",
    "losses.spatial.lambda_{n}",
    "losses.evidential.lambda_{n}",
    "losses.{n}.weight",
    "objectives.reconstruction.{n}_weight",
    "lambda_{n}",
)


def _dig(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if node is None:
            return None
        node = node.get(part) if isinstance(node, dict) else getattr(node, part, None)
    return node


def _legacy_computer(loss_config: Any, name: str) -> float:
    """``BaseLossComputer._get_loss_weight``. model_dump => SCHEMA DEFAULTS COUNT."""
    attr = f"lambda_{name}"
    dumped = loss_config.model_dump() if hasattr(loss_config, "model_dump") else {}
    for section in _COMPUTER_SECTIONS:
        sec = dumped.get(section)
        if not isinstance(sec, dict):
            continue
        if sec.get(attr) is not None:
            return float(sec[attr])
        if section == name and sec.get("weight") is not None:
            return float(sec["weight"])
    return 1.0


def _legacy_recon(config_root: Any, name: str) -> float:
    """``resolve_static_loss_weight``. Takes the FULL settings (it reads ``objectives.*``)."""
    for template in _RECON_PATHS:
        value = _dig(config_root, template.format(n=name))
        if value is not None:
            return float(value)
    return float(_RECON_DEFAULTS.get(name, 1.0))


def _legacy_inline(config_root: Any, name: str) -> float:
    """``BaseTrainingStrategy._get_loss_weight`` (warm-up gate excluded)."""
    for template in (
        "losses.reconstruction.lambda_{n}",
        "objectives.reconstruction.lambda_{n}",
    ):
        value = _dig(config_root, template.format(n=name))
        if value is not None:
            return float(value)
    return float(_INLINE_DEFAULTS.get(name, 1.0))


def _legacy_folding(loss_config: Any, name: str) -> float:
    """``loss_folding.declared_loss_weights`` — the list weight, defaulting to 1.0."""
    # FROZEN at three lists, deliberately NOT LOSS_LIST_DOMAINS (#1924). This module
    # is the oracle for what the PRE-SSOT code did, and the pre-SSOT code did not
    # look at ``latent_losses``. Deriving this from the live SSOT would let the
    # oracle drift with the implementation it exists to check, which is the whole
    # point of freezing it. It dies with this module (see the module docstring).
    for list_name in ("image_losses", "kspace_losses", "complex_losses"):
        for entry in getattr(loss_config, list_name, None) or []:
            raw = getattr(entry, "name", None)
            if raw is None and isinstance(entry, dict):
                raw = entry.get("name")
            if raw != name:
                continue
            weight = getattr(entry, "weight", None)
            if weight is None and isinstance(entry, dict):
                weight = entry.get("weight")
            return 1.0 if weight is None else float(weight)
    return 1.0


def legacy_effective_weight(config_root: Any, name: str, *, family: LegacyFamily) -> float:
    """The weight ``name`` was ACTUALLY training at, before the SSOT.

    Args:
        config_root: the full ``TrainingSettings`` (``recon``/``inline`` read paths
            outside the ``losses`` block).
        name: the loss name AS SPELLED IN THE CONFIG (legacy resolvers never
            canonicalised aliases — that was one of the bugs).
        family: which legacy resolver this arm's strategy actually reached.
    """
    loss_config = _dig(config_root, "losses")
    if family == "computer":
        return _legacy_computer(loss_config, name)
    if family == "recon":
        return _legacy_recon(config_root, name)
    if family == "inline":
        return _legacy_inline(config_root, name)
    if family == "folding":
        return _legacy_folding(loss_config, name)
    raise ValueError(f"unknown legacy family: {family!r}")
