"""Loading from a checkpoint. ``weights`` puts one part into one module, ``best_weights``
puts the best step's models and EMA shadow back; full-state resume lives in ``resume``.

Every weight reader (warm-start, priors, multi-stage stages, inference) goes through
``weights`` so the part names, prefix handling and zero-overlap refusal are decided once.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from spectramr.core.module_utils import strip_wrapper_prefixes
from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import RunState
from spectramr.infrastructure.checkpointing.ref import CheckpointRef, resolve

logger = logging.getLogger(__name__)

#: ``exact`` loads strictly; ``overlap`` transfers shared names and refuses zero overlap.
POLICIES: tuple[str, ...] = ("exact", "overlap")


class CheckpointPartMissingError(ValueError):
    """The checkpoint does not carry the requested part."""


class ZeroOverlapError(ValueError):
    """The part shares no parameter name with the module: an architecture mismatch."""

    def __init__(self, message: str, *, expected: int, unexpected: int) -> None:
        super().__init__(message)
        self.expected = expected
        self.unexpected = unexpected


@dataclass(frozen=True)
class WeightsReport:
    source: Path
    part: str
    matched: int
    total: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def part_state(state: RunState, part: str, *, source: Path) -> Mapping[str, Any]:
    """The state dict of ``part``: ``ema`` or a key of ``models`` (``stage:<name>`` included)."""
    if part == "ema":
        ema = state.get("ema") or {}
        if "generator" in ema:
            return ema["generator"]
    elif part in state["models"]:
        return state["models"][part]
    held = sorted(state["models"]) + (["ema"] if state.get("ema") else [])
    raise CheckpointPartMissingError(f"checkpoint {source} has no part {part!r}; it holds {held}")


def read_part(ref: CheckpointRef, part: str) -> dict[str, Any]:
    """The state dict of one part, wrapper prefixes stripped, for callers that load it
    themselves (a stage-1 VAE whose keys sit under a sub-module)."""
    source = resolve(ref)
    return strip_wrapper_prefixes(part_state(codecs.read(source), part, source=source))


def weights(ref: CheckpointRef, part: str, into: nn.Module, *, policy: str) -> WeightsReport:
    """Load ``part`` of the checkpoint ``ref`` names into ``into``.

    Raises:
        ValueError: Unknown policy (before any IO).
        ZeroOverlapError: Zero shared parameter names (a ``ValueError``).
        CheckpointPartMissingError: The file lacks ``part``.
        RuntimeError: ``exact`` and the state dict does not match ``into``.
    """
    if policy not in POLICIES:
        raise ValueError(f"weights policy {policy!r} is not one of {list(POLICIES)}")
    source = resolve(ref)
    loaded = read_part(source, part)
    model_keys = set(into.state_dict())
    matched = model_keys & set(loaded)
    if not matched:
        raise ZeroOverlapError(
            f"checkpoint {source} part {part!r} shares zero parameter names with "
            f"{type(into).__name__}; e.g. file has {sorted(loaded)[:3]}, model expects "
            f"{sorted(model_keys)[:3]}",
            expected=len(model_keys),
            unexpected=len(loaded),
        )
    result = into.load_state_dict(loaded, strict=policy == "exact")
    report = WeightsReport(
        source=source,
        part=part,
        matched=len(matched),
        total=len(model_keys),
        missing=tuple(sorted(result.missing_keys)),
        unexpected=tuple(sorted(result.unexpected_keys)),
    )
    logger.info(
        "[checkpoint] loaded %s of %s into %s (%s): %d/%d tensors, %d missing, %d unexpected",
        part,
        source,
        type(into).__name__,
        policy,
        report.matched,
        report.total,
        len(report.missing),
        len(report.unexpected),
    )
    return report


def best_weights(path: str | Path, pipeline: Any, *, parallel: Any, strategy: Any) -> None:
    """Put the best step's models, the strategy's models and the EMA shadow back.

    The final checkpoint is written before this runs, so it keeps the last state; the
    returned model is the best one. Engine-wrapped models are restored from the native
    ``best`` tag, module only: at ZeRO-3 their parameters are partitioned and a full
    state dict cannot be loaded into them.
    """
    # Lazy: capture -> checkpoint_adapters -> state_io must not load at weights() import
    # time for cli/ callers.
    from spectramr.infrastructure.checkpointing.capture import (
        check_strategy_models,
        collect_models,
        join_ema,
        strategy_models,
    )
    from spectramr.infrastructure.distributed.checkpoint_adapters import (
        resolve_checkpoint_adapter,
    )

    path = Path(path)
    state = codecs.read(path)
    adapter = resolve_checkpoint_adapter(parallel)
    live = collect_models(pipeline)
    ema = getattr(pipeline, "ema", None)
    # Every refusal comes before any module moves, so a refused restore leaves the
    # LAST state whole rather than best weights beside the last shadow (#2876).
    if ema is not None and not state.get("ema"):
        raise RuntimeError(
            f"{path}: EMA is enabled but the file carries no EMA shadow (a ZeRO-3 shadow "
            "is partitioned and is not saved), so the shadow that was scored at selection "
            "cannot be restored. Nothing was restored; the model holds the last weights."
        )
    shadow = join_ema(state["ema"]) if ema is not None else None
    native = bool(adapter.writes_native_artifact and state["meta"].get("native_tag"))
    if not native and "generator" not in state["models"]:
        raise CheckpointPartMissingError(
            f"{path} holds no 'generator' model; it has {sorted(state['models'])}"
        )
    owned = strategy_models(strategy)
    check_strategy_models(state["models"], owned, source=path)
    handled: frozenset[str] = frozenset()
    if native:
        _client, handled = adapter.load_native(
            models=live,
            checkpoint_dir=path.parent,
            tag=state["meta"]["native_tag"],
            module_only=True,
        )
    for key, module in live.items():
        if key not in handled and (key == "generator" or key in state["models"]):
            adapter.state_io.load_model_state(module, state["models"][key])
    for key, module in owned.items():
        adapter.state_io.load_model_state(module, state["models"][key])
    if shadow is None:
        return
    ema.load_shadow_state_dict(shadow)
    logger.info("Restored the best step's weights and EMA shadow from %s", path)


__all__ = [
    "POLICIES",
    "CheckpointPartMissingError",
    "WeightsReport",
    "ZeroOverlapError",
    "best_weights",
    "part_state",
    "read_part",
    "weights",
]
