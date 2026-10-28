"""The only checkpoint writer.

Under FSDP/DeepSpeed both the capture and the native save are collectives, so every
rank enters here (the loop's ``may_checkpoint``); only the writer rank touches the file.
With a ``CheckpointWriter`` the capture and native save stay on the calling thread and
the file write and prune run on the writer's thread (``checkpoint.async_write``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.async_writer import (
    CheckpointWriteError,
    CheckpointWriter,
)
from spectramr.infrastructure.checkpointing.capture import capture, collect_models
from spectramr.infrastructure.checkpointing.layout import (
    CheckpointLayout,
    native_tag,
    validate_format,
)
from spectramr.infrastructure.checkpointing.retention import prune, validate_keep_last_n
from spectramr.infrastructure.distributed.checkpoint_adapters import resolve_checkpoint_adapter
from spectramr.infrastructure.distributed.distributed_training import RankUtility

logger = logging.getLogger(__name__)

BEST_TAG = "best"


def is_writer_rank() -> bool:
    """Whether this rank writes; with ``rank0_only`` gathers the others hold empty dicts."""
    return RankUtility.is_main_rank()


def _write(
    pipeline: Any,
    *,
    path: Path,
    layout: CheckpointLayout,
    fmt: str,
    tag: str,
    strategy: Any,
    parallel: Any,
    early_stopping: Any,
    client_state: dict[str, Any],
    capture_kw: dict[str, Any],
    writer: CheckpointWriter | None = None,
    then: Callable[[], None] | None = None,
) -> Path:
    adapter = resolve_checkpoint_adapter(parallel)
    live = collect_models(pipeline)
    has_native = bool(adapter.native_tag_roots(models=live, checkpoint_dir=layout.directory))
    try:
        state = capture(
            pipeline,
            strategy=strategy,
            parallel=parallel,
            early_stopping=early_stopping,
            fmt=fmt,
            native_tag=tag if has_native else None,
            **capture_kw,
        )
        # COLLECTIVE: every rank calls; DeepSpeed's save ends in a barrier.
        adapter.save_native(
            models=live, checkpoint_dir=layout.directory, tag=tag, client_state=client_state
        )
        if is_writer_rank() and writer is not None:
            # Fatal exactly where the synchronous call site is: the best and the
            # loss-schedule saves are fail-soft in the loop (#713), every other step is not.
            fatal = capture_kw["role"] == "step" and capture_kw["reason"] != "loss_schedule"
            writer.submit(path, state, then, fatal=fatal)
        elif is_writer_rank():
            codecs.write(path, state)
            logger.info("Checkpoint saved: %s (%s)", path, capture_kw["reason"])
    except CheckpointWriteError:
        # An earlier queued write failed; it already names its own file.
        raise
    except Exception as exc:
        logger.error("Failed to save checkpoint %s: %s", path, exc)
        raise RuntimeError(f"Failed to save checkpoint {path}: {exc}") from exc
    return path


def save_step(
    pipeline: Any,
    *,
    layout: CheckpointLayout,
    config: Any,
    strategy: Any,
    parallel: Any,
    early_stopping: Any,
    reason: str,
    epoch: int,
    step: int,
    config_fingerprint: str | None = None,
    writer: CheckpointWriter | None = None,
) -> Path:
    """Write ``step_{step}.{fmt}`` and apply ``keep_last_n``; raises if the write fails.

    With ``writer`` the file and the prune land on the writer's thread; their failure
    is raised by the writer's next ``submit`` or ``drain``.
    """
    # Read before anything is written: a bad value after the file lands would be
    # reported as a failed save, and a failed save stops the run.
    keep_last_n = validate_keep_last_n(config.checkpoint.keep_last_n)
    fmt = validate_format(config.checkpoint.format)
    path = layout.step_path(step, fmt)
    adapter = resolve_checkpoint_adapter(parallel)
    # Read here, not in the prune: the writer thread must not walk the live pipeline.
    native_roots = (
        adapter.native_tag_roots(models=collect_models(pipeline), checkpoint_dir=layout.directory)
        if is_writer_rank()
        else ()
    )

    def apply_retention() -> None:
        # Outside the write: the checkpoint has landed, so retention logs its own errors.
        prune(layout, path, keep_last_n=keep_last_n, native_roots=native_roots)

    _write(
        pipeline,
        path=path,
        layout=layout,
        fmt=fmt,
        tag=native_tag(step),
        strategy=strategy,
        parallel=parallel,
        early_stopping=early_stopping,
        client_state={"epoch": epoch, "global_step": step},
        capture_kw={
            "role": "step",
            "reason": reason,
            "epoch": epoch,
            "step": step,
            "config_fingerprint": config_fingerprint,
        },
        writer=writer,
        then=apply_retention,
    )
    if is_writer_rank() and writer is None:
        apply_retention()
    return path


def save_best(
    pipeline: Any,
    *,
    layout: CheckpointLayout,
    config: Any,
    strategy: Any,
    parallel: Any,
    early_stopping: Any,
    epoch: int,
    step: int,
    metric_name: str,
    metric_value: float,
    config_fingerprint: str | None = None,
    writer: CheckpointWriter | None = None,
) -> Path:
    """Replace ``best.{fmt}``; there is always exactly one, and it is never pruned.

    With ``writer`` the file lands on the writer's thread; drain it before reading back.
    """
    fmt = validate_format(config.checkpoint.format)
    best = {"name": metric_name, "value": float(metric_value)}
    return _write(
        pipeline,
        path=layout.best_path(fmt),
        layout=layout,
        fmt=fmt,
        tag=BEST_TAG,
        strategy=strategy,
        parallel=parallel,
        early_stopping=early_stopping,
        client_state={
            "epoch": epoch,
            "global_step": step,
            **{f"best_metric_{k}": v for k, v in best.items()},
        },
        capture_kw={
            "role": "best",
            "reason": "best",
            "epoch": epoch,
            "step": step,
            "best_metric": best,
            "config_fingerprint": config_fingerprint,
        },
        writer=writer,
    )


__all__ = ["BEST_TAG", "is_writer_rank", "save_best", "save_step"]
