"""The live training objects as one RunState, read the way the director read them.

Every rank calls ``capture``: FSDP and ZeRO-3 gathers are collectives, so a rank that
skipped it would hang the others. Only the writer rank's result is published (store).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from spectramr.core.rng_state import capture_rng_state
from spectramr.infrastructure.checkpointing.payload import BestMetric, RunState, build_meta
from spectramr.infrastructure.checkpointing.restore import CheckpointPartMissingError
from spectramr.infrastructure.checkpointing.topology import topology_record
from spectramr.infrastructure.distributed.checkpoint_adapters import resolve_checkpoint_adapter
from spectramr.infrastructure.optimization.ema import ModelEma

#: Each model key, the pipeline attribute of its optimizer, and of its scheduler.
MODEL_KEYS: tuple[tuple[str, str, str], ...] = (
    ("generator", "optimizer_g", "scheduler_g"),
    ("discriminator", "optimizer_d", "scheduler_d"),
)


def resolve_scaler(pipeline: Any, strategy: Any) -> Any | None:
    """The scaler that actually scaled this run's losses.

    The live one belongs to the strategy's AMP helper; ``pipeline.scaler`` was once a
    GradScaler no backward pass touched, so saving it stored a scale that never moved.
    """
    live = getattr(getattr(strategy, "amp_helper", None), "scaler", None)
    if live is not None:
        return live
    return getattr(pipeline, "scaler", None)


def collect_models(pipeline: Any) -> dict[str, Any]:
    """The wrapped modules by key, deliberately NOT unwrapped.

    DeepSpeed's save lives on the engine and FSDP's gather needs the FSDP root, so the
    bare module underneath would make either a silent no-op.
    """
    return {
        key: module
        for key, _opt, _sched in MODEL_KEYS
        if (module := getattr(pipeline, key, None)) is not None
    }


def strategy_models(strategy: Any) -> dict[str, Any]:
    """The strategy's own models (multi-stage stages), sorted by key.

    Sorted because the gathers are collectives; a key that would shadow a pipeline
    model raises rather than overwrite it in the file.
    """
    owned = getattr(strategy, "checkpoint_models", None)
    models = dict(owned()) if owned is not None else {}
    clash = sorted(set(models) & {key for key, _opt, _sched in MODEL_KEYS})
    if clash:
        raise ValueError(f"strategy checkpoint_models {clash} collide with pipeline models")
    return {key: models[key] for key in sorted(models)}


def check_strategy_models(saved: Mapping[str, Any], owned: Mapping[str, Any], *, source: Any) -> None:
    """Refuse a file whose strategy models differ from the run's, before any load.

    A stage left at its fresh initialisation while the resume reports success is the
    #2820 shape; a file with none at all predates the stages being saved (#2962).
    """
    pipeline_keys = {key for key, _opt, _sched in MODEL_KEYS}
    in_file = set(saved) - pipeline_keys
    if in_file == set(owned):
        return
    if owned and not in_file:
        raise CheckpointPartMissingError(
            f"{source} holds no stage models but the run has {sorted(owned)}; it was "
            "written before multi-stage runs saved their stages (#2962)."
        )
    raise CheckpointPartMissingError(
        f"{source} holds strategy models {sorted(in_file)} but the run has "
        f"{sorted(owned)}; missing {sorted(set(owned) - in_file)}, extra "
        f"{sorted(in_file - set(owned))} (#2962)."
    )


def split_ema(shadow: dict[str, Any]) -> dict[str, Any]:
    """``ModelEma.shadow_state_dict()`` as the ``ema`` part: bare weights plus the counter.

    The counter keeps the decay ramp across a resume (#2172); it lives beside the
    weights so ``weights(part="ema")`` loads them strictly into a bare model. A shadow
    without it raises ``KeyError`` rather than restarting the ramp at 0.
    """
    weights = dict(shadow)
    counter = weights.pop(ModelEma._NUM_UPDATES_KEY)
    return {"generator": weights, "num_updates": int(counter)}


def join_ema(ema: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`split_ema`, the form ``ModelEma.load_shadow_state_dict`` takes."""
    joined = dict(ema["generator"])
    joined[ModelEma._NUM_UPDATES_KEY] = torch.tensor(int(ema["num_updates"]))
    return joined


def capture(
    pipeline: Any,
    *,
    strategy: Any,
    parallel: Any,
    early_stopping: Any,
    role: str,
    reason: str,
    epoch: int,
    step: int,
    fmt: str,
    native_tag: str | None = None,
    best_metric: BestMetric | None = None,
    config_fingerprint: str | None = None,
) -> RunState:
    """Read every part of the run; collective, so every rank must call it, in this order."""
    adapter = resolve_checkpoint_adapter(parallel)
    io = adapter.state_io
    live = collect_models(pipeline)
    # Models then optimizers, generator first: the gathers are collectives and every
    # rank must enter them in the same order.
    state: RunState = {"models": {key: adapter.full_model_state(m) for key, m in live.items()}}
    # After the pipeline models, sorted, on every rank (#2962).
    for key, module in strategy_models(strategy).items():
        state["models"][key] = adapter.full_model_state(module)
    # An engine's ZeRO optimizer and scheduler are saved in its native tag directory; the
    # optimizer's state dict holds DeepSpeed's LossScaler object, which no codec stores.
    native = adapter.engine_owned_keys(live)
    optim = {
        key: io.optimizer_state(live[key], opt)
        for key, opt_attr, _ in MODEL_KEYS
        if key in live
        and key not in native
        and (opt := getattr(pipeline, opt_attr, None)) is not None
    }
    sched = {
        key: sch.state_dict()
        for key, _, sched_attr in MODEL_KEYS
        if key not in native and (sch := getattr(pipeline, sched_attr, None)) is not None
    }
    state["optim"] = optim
    if sched:
        state["sched"] = sched
    # A disabled scaler's state is {}, which GradScaler.load_state_dict refuses on resume.
    if (scaler := resolve_scaler(pipeline, strategy)) is not None and (
        scaler_state := scaler.state_dict()
    ):
        state["scaler"] = scaler_state

    # Not unwrap(ema).state_dict(): that peels ModelEma's own module and loses the warmup
    # counter (#2172). The best file carries the shadow only where it is whole-parameter.
    ema = getattr(pipeline, "ema", None)
    if ema is not None and (role == "step" or adapter.ema_is_whole(live)):
        state["ema"] = split_ema(ema.shadow_state_dict())

    if strategy is not None and hasattr(strategy, "strategy_state_dict"):
        owned = strategy.strategy_state_dict()
        if any(owned.values()):
            state["strategy"] = owned
    if early_stopping is not None:
        state["early_stopping"] = early_stopping.state_dict()

    # No collective: under plain DDP only rank 0 reaches a save, so it is rank 0's
    # stream that a resume replays.
    state["rng"] = capture_rng_state()
    state["meta"] = build_meta(
        role=role,
        reason=reason,
        epoch=epoch,
        step=step,
        fmt=fmt,
        topology=topology_record(parallel),
        config_fingerprint=config_fingerprint,
        best_metric=best_metric,
        native_tag=native_tag,
    )
    return state


__all__ = [
    "MODEL_KEYS",
    "capture",
    "check_strategy_models",
    "collect_models",
    "join_ema",
    "resolve_scaler",
    "split_ema",
    "strategy_models",
]
