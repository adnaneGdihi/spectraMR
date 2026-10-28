"""Write a minimal schema-1 RunState for reader tests, through the real codec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import build_meta


def write_run_state(
    path: Path,
    models: dict[str, dict],
    *,
    ema: dict | None = None,
    role: str = "best",
    extra_parts: dict[str, Any] | None = None,
) -> Path:
    """``extra_parts`` adds any other RunState part (``optim``, ``sched``, ``rng``...)."""
    fmt = path.suffix.removeprefix(".")
    meta = build_meta(
        role=role,
        reason="best" if role == "best" else "periodic",
        epoch=1,
        step=10,
        fmt=fmt,
        topology={"world_size": 1, "parallel_strategy": "none"},
        best_metric={"name": "val_psnr", "value": 30.0} if role == "best" else None,
    )
    state = {"meta": meta, "models": models}
    if ema is not None:
        state["ema"] = {"generator": ema, "num_updates": 3}
    clash = sorted(set(extra_parts or {}) & set(state))
    if clash:
        raise ValueError(f"extra_parts may not replace {clash}; pass them as arguments")
    state.update(extra_parts or {})
    codecs.write(path, state)
    return path
