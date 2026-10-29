"""Full-state resume: every part of a RunState back into the live run.

Collective under FSDP/DeepSpeed (the sharded loads), so every rank calls it. The order
is the director's: topology guard, native artifact, strategy state, models and
optimizers, EMA, schedulers, scaler, RNG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spectramr.core.parallel_groups import dp_rank
from spectramr.core.rng_state import restore_rng_state
from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.capture import (
    MODEL_KEYS,
    check_strategy_models,
    collect_models,
    join_ema,
    resolve_scaler,
    strategy_models,
)
from spectramr.infrastructure.checkpointing.key_drift import drift_hint
from spectramr.infrastructure.checkpointing.restore import CheckpointPartMissingError
from spectramr.infrastructure.checkpointing.topology import check_resume_topology
from spectramr.infrastructure.distributed.checkpoint_adapters import resolve_checkpoint_adapter
from spectramr.infrastructure.distributed.distributed_training import RankUtility
from spectramr.infrastructure.training.scheduler_system import load_scheduler_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeResult:
    """Where the resumed run continues, and the selection state the loop restores."""

    epoch: int
    step: int
    early_stopping: dict[str, Any] | None
    reason: str


def _gate(checkpoint_cfg: Any, name: str) -> bool:
    # Truthiness with a True default, as the director read load_scheduler_state.
    return bool(getattr(checkpoint_cfg, name, True))


def _restore_rng(rng: dict[str, Any] | None) -> None:
    # The file holds the data-parallel index-0 streams. Every rank with that index
    # (rank 0 and its model-parallel peers, such as spatial slabs of one volume) restores
    # them, so the peers that draw the same batch keep drawing the same noise. Other
    # replicas keep their own stream: replaying one stream on all of them would give
    # every replica one augmentation sequence.
    if rng and dp_rank() == 0:
        restore_rng_state(rng)
        logger.info("Restored RNG state (torch + cuda + numpy + python)")
    elif rng:
        logger.info(
            "[RNG] rank %d keeps its own stream; the checkpoint carries data-parallel "
            "replica 0's only.",
            RankUtility.get_rank(),
        )


def _to_device(value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def _refuse_partitioned_ema(pipeline: Any, state: dict[str, Any], adapter: Any, live: Any) -> None:
    # At ZeRO-3 a step file carries rank 0's partitioned shadow; loading it on every rank
    # would hand each one rank 0's partition, so it is refused before any load.
    ema_on = state.get("ema") and getattr(pipeline, "ema", None) is not None
    if ema_on and not adapter.ema_is_whole(live):
        raise RuntimeError(
            f"the [{adapter.name}] EMA shadow is not whole-parameter (ZeRO stage 3), so "
            "the file's shadow is rank 0's partition and cannot be restored on every rank."
        )


def _refuse_missing_native_state(
    state: dict[str, Any], adapter: Any, live: Any, *, checkpoint_dir: Path, load_optimizer: bool
) -> None:
    """Refuse, before any load, a file whose engine optimizer lives in an absent tag (#2965).

    ``capture`` leaves an engine-owned optimizer out of ``optim``: it is in the tag
    directory. Without that tag, or without a native engine to read it, the payload
    loop would otherwise fail on a bare ``KeyError: 'generator'``.
    """
    tag = state["meta"].get("native_tag")
    native = adapter.writes_native_artifact and tag
    if native:
        roots = adapter.native_tag_roots(models=live, checkpoint_dir=checkpoint_dir)
        missing = [str(root / tag) for root in roots if not (root / tag).is_dir()]
        if missing:
            raise CheckpointPartMissingError(
                f"the file names DeepSpeed tag {tag!r}, but {missing} is missing"
            )
    if not load_optimizer or "generator" in state.get("optim", {}):
        return
    if native and "generator" in adapter.engine_owned_keys(live):
        return
    raise CheckpointPartMissingError(
        f"the file holds no generator optimizer state; it lives in DeepSpeed tag {tag!r}. "
        "Resume under parallel.strategy deepspeed with that tag present, or set "
        "checkpoint.load_optimizer_state: false to resume the weights only."
    )


def resume(
    path: str | Path, pipeline: Any, *, strategy: Any, parallel: Any, config: Any
) -> ResumeResult:
    """Restore everything ``path`` holds into the run.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: No pipeline to load into.
        RuntimeError: Any load failure, naming the file and the exception type (#2902).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if not pipeline:
        raise ValueError("Pipeline must be set before loading")
    state: dict[str, Any] | None = None
    live: dict[str, Any] | None = None
    try:
        logger.info("Loading checkpoint: %s", path)
        state = codecs.read(path)
        meta = state["meta"]
        checkpoint_cfg = getattr(config, "checkpoint", None)
        # Before the collective sharded restore, so a backend swap fails fast.
        check_resume_topology(meta.get("topology"), checkpoint_cfg, parallel, source=path)
        load_optimizer = _gate(checkpoint_cfg, "load_optimizer_state")
        load_scheduler = _gate(checkpoint_cfg, "load_scheduler_state")

        adapter = resolve_checkpoint_adapter(parallel)
        live = collect_models(pipeline)
        _refuse_partitioned_ema(pipeline, state, adapter, live)
        _refuse_missing_native_state(
            state, adapter, live, checkpoint_dir=path.parent, load_optimizer=load_optimizer
        )
        owned_models = strategy_models(strategy)
        check_strategy_models(state["models"], owned_models, source=path)
        # Called even without a strategy part: a file with stages but no stage optimizers
        # must refuse here, before any load, not resume them fresh (#2962).
        if (check_owned := getattr(strategy, "check_strategy_state", None)) is not None:
            check_owned(state.get("strategy"))
        handled: frozenset[str] = frozenset()
        if adapter.writes_native_artifact and meta.get("native_tag"):
            # The payload carries gathered weights but not sharded ZeRO optimizer state, so
            # a native file must be read natively or the moments silently restart.
            native_state, handled = adapter.load_native(
                models=live,
                checkpoint_dir=path.parent,
                tag=meta["native_tag"],
                optimizer=load_optimizer,
                scheduler=load_scheduler,
            )
            if native_state is not None:
                logger.info("[%s] restored sharded state for %s", adapter.name, sorted(handled))

        # Strategy models (multi-stage stages) before strategy state: their freeze is
        # re-applied there, and both precede opt_g's load, whose groups hold them (#2962).
        for key, module in owned_models.items():
            adapter.state_io.load_model_state(module, state["models"][key])

        # Strategy state before any optimizer: a lazily sized module adds its param group
        # here, so the saved optimizer state finds the same groups (#2796).
        if (
            "strategy" in state
            and strategy is not None
            and hasattr(strategy, "load_strategy_state_dict")
        ):
            # The director read with map_location=pipeline.device; codecs read to CPU, and
            # a module the strategy builds from this state must land on the run's device.
            device = getattr(pipeline, "device", None)
            owned = state["strategy"] if device is None else _to_device(state["strategy"], device)
            strategy.load_strategy_state_dict(owned, load_optimizer=load_optimizer)
            logger.info("Restored strategy-owned state from checkpoint")

        # Handled keys are skipped per model, never per run: a plain discriminator beside
        # an engine still needs the payload (#1309), and a ZeRO optimizer cannot read the
        # generic dict (#2482).
        optim = state.get("optim", {})
        for key, opt_attr, _ in MODEL_KEYS:
            if key in handled:
                continue
            model = live.get(key)
            optimizer = getattr(pipeline, opt_attr, None)
            # The generator payload is mandatory; the discriminator's is not.
            if key == "generator" or (key in state["models"] and model is not None):
                adapter.state_io.load_model_state(model, state["models"][key])
            if load_optimizer and (key == "generator" or (key in optim and optimizer)):
                adapter.state_io.load_optimizer_state(model, optimizer, optim[key])

        ema = getattr(pipeline, "ema", None)
        if state.get("ema") and ema is not None:
            # Restores the warmup counter as well as the weights (#2172).
            ema.load_shadow_state_dict(join_ema(state["ema"]))
            logger.info("Restored EMA shadow weights")

        # Not skipped for handled keys: scheduler state is replicated, so a second load
        # is idempotent.
        if load_scheduler:
            for key, _, sched_attr in MODEL_KEYS:
                scheduler = getattr(pipeline, sched_attr, None)
                if key in state.get("sched", {}) and scheduler:
                    load_scheduler_state(scheduler, state["sched"][key], sched_attr)

        # Into the scaler that will actually be used (see capture.resolve_scaler).
        live_scaler = resolve_scaler(pipeline, strategy)
        if "scaler" in state and live_scaler is not None:
            live_scaler.load_state_dict(state["scaler"])
            logger.info("AMP GradScaler state restored from checkpoint")

        _restore_rng(state.get("rng"))
        logger.info("Checkpoint loaded: epoch=%s, step=%s", meta["epoch"], meta["step"])
        return ResumeResult(
            epoch=meta["epoch"],
            step=meta["step"],
            early_stopping=state.get("early_stopping"),
            reason=meta["reason"],
        )
    except Exception as exc:
        # The file and the type are part of the message: str(KeyError(0)) is a bare "0" (#2902).
        logger.error("Failed to load checkpoint %s: %s: %s", path, type(exc).__name__, exc)
        try:
            hint = drift_hint(state, live)
        except Exception:  # a diagnostic must never replace the error it explains
            logger.debug("checkpoint key-drift hint failed", exc_info=True)
            hint = None
        raise RuntimeError(
            f"Failed to load checkpoint {path}: {type(exc).__name__}: {exc}"
            + (f"\n{hint}" if hint else "")
        ) from exc


__all__ = ["ResumeResult", "resume"]
