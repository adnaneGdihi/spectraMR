"""The multi-stage strategy's checkpoint state beyond its stage weights (#2962).

``MultiTrainingStrategy`` keeps its stages, stage optimizers and per-stage early
stopping in plain dicts, which ``BaseTrainingStrategy.strategy_state_dict`` does not
discover. The stage weights travel as ``models["stage:<name>"]`` through the generic
``checkpoint_models`` seam; everything else is the ``pipeline`` entry built here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

#: The key of the multi-stage entry inside the ``strategy`` part.
PIPELINE_KEY = "pipeline"
#: The ``models`` key prefix of a stage's weights; the reader's ``stage:<name>`` part.
STAGE_PREFIX = "stage:"


def stage_models(strategy: Any) -> dict[str, nn.Module]:
    """Each stage under its ``models`` key, sorted so every rank gathers in one order."""
    return {
        f"{STAGE_PREFIX}{name}": strategy.multi_stages[name]
        for name in sorted(strategy.multi_stages)
    }


def capture_pipeline_state(strategy: Any) -> dict[str, Any]:
    """Stage optimizers, early-stopping counters and the freeze the run has applied."""
    return {
        "stage_optimizers": {
            name: opt.state_dict() for name, opt in sorted(strategy._stage_optimizers.items())
        },
        # Created lazily at the first epoch end that resolves a monitor.
        "es_best": dict(getattr(strategy, "_es_best", {})),
        "es_wait": dict(getattr(strategy, "_es_wait", {})),
        # A list: the safetensors header encodes no set.
        "frozen_by_early_stopping": sorted(getattr(strategy, "_es_frozen", ())),
        "all_unfrozen": bool(strategy._all_unfrozen),
    }


def _set_trainable(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def check_pipeline_state(strategy: Any, entry: Mapping[str, Any] | None) -> None:
    """Refuse an entry this run cannot restore whole; resume calls it before any load.

    Raises:
        RuntimeError: No entry (a file with stages but no strategy part included), a
            stage optimizer set that differs from the live one, or an early-stopped
            stage this run does not have.
    """
    owner = type(strategy).__name__
    if entry is None:
        raise RuntimeError(
            f"{owner}: the checkpoint's strategy part has no {PIPELINE_KEY!r} entry, so "
            "the stage optimizers and stage early stopping cannot be restored (#2962)."
        )
    saved, live = set(entry["stage_optimizers"]), set(strategy._stage_optimizers)
    if saved != live:
        raise RuntimeError(
            f"{owner}: the checkpoint holds stage optimizers {sorted(saved)} but the run "
            f"built {sorted(live)}; a stage optimizer would resume at fresh state (#2962)."
        )
    unknown = sorted(set(entry["frozen_by_early_stopping"]) - set(strategy.multi_stages))
    if unknown:
        raise RuntimeError(
            f"{owner}: the checkpoint's early-stopped stages {unknown} are not stages of "
            f"this run ({sorted(strategy.multi_stages)})."
        )


def restore_pipeline_state(
    strategy: Any, entry: Mapping[str, Any], *, load_optimizer: bool
) -> None:
    """Re-apply the freeze, then load the stage optimizers and early-stopping counters.

    Runs before ``opt_g`` is loaded (resume restores strategy state first, #2796): the
    end-to-end unfreeze adds an ``opt_g`` param group, so it must exist again for the
    saved optimizer state to find the same groups. ``entry`` has passed
    :func:`check_pipeline_state`; ``load_optimizer`` is resume's
    ``checkpoint.load_optimizer_state`` gate (#2964).
    """
    # The order of on_epoch_start: unfreeze, recollect, then register, else
    # _register_global_params sees the stale list and adds no group.
    if entry["all_unfrozen"]:
        for module in strategy.multi_stages.values():
            _set_trainable(module, True)
        strategy._trainable_params = strategy._collect_trainable_params()
        strategy._register_global_params()
        strategy._all_unfrozen = True
    frozen = set(entry["frozen_by_early_stopping"])
    for name in frozen:
        _set_trainable(strategy.multi_stages[name], False)
    strategy._es_frozen = frozen
    strategy._trainable_params = strategy._collect_trainable_params()
    if load_optimizer:
        for name, optimizer in strategy._stage_optimizers.items():
            optimizer.load_state_dict(entry["stage_optimizers"][name])
    strategy._es_best = {k: float(v) for k, v in entry["es_best"].items()}
    strategy._es_wait = {k: int(v) for k, v in entry["es_wait"].items()}


__all__ = [
    "PIPELINE_KEY",
    "STAGE_PREFIX",
    "capture_pipeline_state",
    "check_pipeline_state",
    "restore_pipeline_state",
    "stage_models",
]
