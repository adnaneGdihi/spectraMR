"""Context resolver -- the single config->IR seam (cached-cascade WS-X).

``resolve(config)`` turns the frozen ``TrainingSettings`` into a
:class:`~spectramr.domain.entities.experiment_context.ResolvedExperimentContext`
by looking up each chosen component's *contract* from its registry. This is the
"functional core, imperative shell" boundary: resolve once -> (Phase A:
reconcile) -> then check.

Scope: the public signature
``resolve(config: TrainingSettings) -> ResolvedExperimentContext`` is frozen and
the model/strategy/loss profiles are populated from their registries.

The top-level<->nested model-domain reconciliation (``_propagate_top_level_model_domain``
/ ``_warn_on_deprecated_top_level_blocks`` in ``config/settings.py``) **stays in
settings.py** and is NOT relocated here. It is load-time config *normalization*
that the runtime path depends on directly -- ``strategies/base.py`` reads
``config.model.target_domain``, ``strategies/mixins/kspace.py`` reads
``config.model.model_domain``, and ``training/utils/domain_inference.py``'s P1
branch reads ``target_domain`` -- none of which read this (read-only, off-runtime)
IR. Removing the propagation would silently fall those consumers back to
``"image"`` (the pitfall-#15 / DC-blob class of bug). The resolver *mirrors* the
reconciled value into the IR (``_resolve_data_profile`` reads the already-propagated
domain) so the compatibility rules see it; it does not own it. See
``TODO/ws1_config_findings.md`` item WS1-core-06 ("do not move now").

The resolver is deliberately fail-soft: missing/unknown facts resolve to ``None``
("unannotated, skip the check") rather than crashing, so it is safe to call on any
config.

Layering: this module lives in ``infrastructure/validation/`` and may import the
registries (``infrastructure/training`` for strategies, ``models`` for model/loss
capabilities) and the domain IR -- all rightward/same-tier, layering-clean.
"""

from __future__ import annotations

import logging
from typing import Any

from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
from spectramr.domain.entities.experiment_context import (
    DataProfile,
    PhysicsProfile,
    ResolvedExperimentContext,
)
from spectramr.infrastructure.validation.spec_card import derive_loader_output_domain
from spectramr.models.capabilities import (
    AdapterCapabilities,
    LossCapabilities,
    ModelCapabilities,
    StrategyCapabilities,
)

logger = logging.getLogger(__name__)

__all__ = ["resolve"]


def _get(obj: Any, path: str, default: Any = None) -> Any:
    """Safe nested ``getattr`` over a dotted path (never raises)."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return cur if cur is not None else default


def _resolve_model_profile(model_type: str) -> ModelCapabilities:
    try:
        from spectramr.models.registry import get_model_capabilities

        caps = get_model_capabilities(model_type)
        if isinstance(caps, ModelCapabilities):
            return caps
    except Exception as exc:  # registry miss / import issue -- stay fail-soft
        logger.debug("model capabilities unresolved for %r: %s", model_type, exc)
    return ModelCapabilities()


def _resolve_strategy(config: Any) -> tuple[str, StrategyCapabilities]:
    try:
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        factory = TrainingStrategyFactory()
        strategy_class = factory.get_strategy_class(config)
        caps = factory.get_strategy_capabilities(strategy_class)
        return strategy_class.__name__, caps
    except Exception as exc:
        logger.debug("strategy unresolved: %s", exc)
        return (
            str(_get(config, "training.training_mode", "<unresolved>")),
            StrategyCapabilities(),
        )


def _iter_loss_names(config: Any) -> list[str]:
    """Best-effort enumeration of active loss names from the losses block.

    TODO(WS-X Phase A): make this exhaustive over the declarative v6.1 loss
    schema (image/kspace/complex/physics blocks + weights) and fold in the
    ``config_health_checker`` loss-domain checks. For now it collects entry
    names defensively so the IR carries *something* without crashing.
    """
    names: list[str] = []
    losses = _get(config, "losses")
    if losses is None:
        return names
    for block in LOSS_LIST_DOMAINS:
        entries = getattr(losses, block, None) or []
        try:
            for entry in entries:
                name = getattr(entry, "name", None) or getattr(entry, "type", None)
                if name:
                    names.append(str(name))
        except TypeError:
            continue
    return names


def _resolve_loss_profile(loss_names: list[str]) -> tuple[LossCapabilities, ...]:
    try:
        from spectramr.models.losses import get_loss_capabilities
    except Exception:
        return ()
    profile: list[LossCapabilities] = []
    for name in loss_names:
        caps = get_loss_capabilities(name)
        if caps is not None:
            profile.append(caps)
    return tuple(profile)


def _rank_from_patch_size(ps: Any) -> tuple[int, ...] | None:
    """Spatial rank = number of ``patch_size`` dims with extent > 1.

    Counting extent>1 (rather than assuming an ``[H, W, D]`` layout and reading
    dim 2) is robust across:
      * 1-D MRF/sequence data — ``[1000, 1, 1]`` -> ``(1,)`` (NOT a 2-D image),
      * 2-D slices — ``[H, W]`` or ``[H, W, 1]`` -> ``(2,)`` (incl. the 2-D slab
        mode of a 3-D-capable model: the *data* is 2-D, matching a ``(2,)`` cap),
      * 3-D volumes — ``[D, H, W]`` -> ``(3,)``.
    Sourced from data geometry, NOT ``model.spatial_dims`` (which would falsely
    reject slab-mode arms). Fail-soft -> None (also when no dim exceeds 1).
    """
    if not ps:
        return None
    try:
        rank = sum(1 for d in ps if isinstance(d, int) and d > 1)
    except TypeError:
        return None
    return (rank,) if rank in (1, 2, 3) else None


def _resolve_data_profile(config: Any) -> DataProfile:
    # spatial_dims is resolved from data geometry (patch_size), NOT
    # model.spatial_dims (see _rank_from_patch_size). `paired` (only True is
    # safely derivable) and `channel_layout` (duplicates
    # config_health_checker.check_domain_alignment — two checkers would risk
    # disagreeing on one arm) are deliberately left None.
    return DataProfile(
        dataset_type=_get(config, "data.dataset_type") or _get(config, "data.dataset"),
        multi_contrast=_get(config, "data.multi_contrast.enabled"),
        # What the loader emits (dataset family, then the coil-processing mode),
        # not the model's declared target domain: that proxy was the model OUTPUT
        # and rejected five arms whose k-space loader emits images under
        # ``rss_image`` (2026-09-03). One derivation, shared with the spec card.
        domain=derive_loader_output_domain(config),
        # `data.sampling.patch_size`, not `data.patch_size`: phase 9e moved the
        # key and this read was never repointed, so `_get` fail-softed to None
        # and `rule_spatial_rank` -- the guard against a 3-D model fed 2-D
        # patches -- returned [] on every arm in the corpus. Same shape as the
        # `config_version` miss above; the legacy spelling stays accepted
        # because the rename is still `fold` posture.
        spatial_dims=_rank_from_patch_size(
            _get(config, "data.sampling.patch_size") or _get(config, "data.patch_size")
        ),
    )


def _resolve_physics_profile(config: Any) -> PhysicsProfile:
    # TODO(WS-X Phase A): resolve forward-model/coil/DC/sampling from the
    # physics schema. Phase-0 fills the cheap facts.
    return PhysicsProfile(
        forward_model=_get(config, "physics.forward_model"),
        data_consistency=_get(config, "physics.data_consistency.method")
        or _get(config, "physics.method"),
        coil_estimation=_get(config, "physics.coil_processing.estimation"),
        sampling=_get(config, "physics.sampling.mask_type"),
    )


def _resolve_adapter_chain(config: Any) -> tuple[AdapterCapabilities, ...]:
    """Populate the declared adapter chain from ``config.adapters``.

    Without this, ``ctx.adapter_chain`` is empty and the compatibility rules
    cannot see config-declared domain/shape bridges (e.g. an
    ``ifft_kspace_to_image`` pre-loss adapter), so a correctly-bridged arm
    (k-space-output model + image-domain loss) would be FALSELY rejected by the
    domain-chain rule. We look each declared adapter's name up in the adapter
    registry to recover its ``AdapterCapabilities`` (bridges_from/to). Fail-soft.
    """
    adapters = _get(config, "adapters")
    if adapters is None:
        return ()
    try:
        from spectramr.data.adapters.registry import get_adapter_capabilities
    except Exception as exc:
        logger.debug("adapter registry unavailable: %s", exc)
        return ()
    chain: list[AdapterCapabilities] = []
    for point in (
        "pre_model",
        "post_model",
        "pre_loss_pred",
        "pre_loss_target",
        "pre_metric",
    ):
        steps = getattr(adapters, point, None) or []
        try:
            for step in steps:
                name = getattr(step, "name", None) or (
                    step.get("name") if isinstance(step, dict) else None
                )
                if not name:
                    continue
                caps = get_adapter_capabilities(name)
                if isinstance(caps, AdapterCapabilities):
                    chain.append(caps)
        except TypeError:
            continue
    return tuple(chain)


def resolve(config: Any) -> ResolvedExperimentContext:
    """Resolve a frozen ``TrainingSettings`` into the experiment IR.

    Args:
        config: a loaded, frozen ``TrainingSettings`` (the config SSOT). Typed
            as ``Any`` to avoid an infrastructure->config import-time hard
            dependency on the schema class; the attribute contract is the v6.x
            ``TrainingSettings`` shape.

    Returns:
        A frozen :class:`ResolvedExperimentContext`.
    """
    model_type = str(_get(config, "model.model_type", "<unknown>"))
    strategy_name, strategy_profile = _resolve_strategy(config)
    loss_names = _iter_loss_names(config)
    # Phase-A reconciliation: record which strategy-declared required config
    # fields are unset, so the pure ``required_config_fields`` rule can flag
    # them without itself digging into the config (preserves rule purity).
    unresolved_required = tuple(
        f for f in strategy_profile.required_config_fields if _get(config, f) is None
    )
    return ResolvedExperimentContext(
        # `run.config_version`, not `config_version`: the key moved onto the
        # `run:` block in phase 5 and this read was never repointed, so every
        # resolved context since has stamped "?" and the audit header has
        # rendered "config v?". `_get` swallows the miss by design, and the path
        # is held as a STRING -- an attribute grep over the rename could not see
        # it. Same shape as the hasattr guards that survived an earlier rename.
        config_version=str(_get(config, "run.config_version", "?")),
        model_type=model_type,
        strategy_name=strategy_name,
        loss_names=tuple(loss_names),
        data_profile=_resolve_data_profile(config),
        model_profile=_resolve_model_profile(model_type),
        strategy_profile=strategy_profile,
        loss_profile=_resolve_loss_profile(loss_names),
        physics_profile=_resolve_physics_profile(config),
        adapter_chain=_resolve_adapter_chain(config),
        unresolved_required_fields=unresolved_required,
    )
