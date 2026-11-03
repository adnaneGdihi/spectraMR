"""Run a strategy step inside a snapshot trace and say whether a model call was recorded.

The snapshot hooks see a model only through ``env.models`` or
``_capture_model_output(module=...)``; a strategy that calls a module by any other
route leaves no ``model/`` stage, and its snapshot cannot show what was fed in.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from spectramr.core import snapshot_trace as st
from spectramr.core.snapshot_trace import View
from spectramr.infrastructure.training.snapshot.hooks import ModelHooks


def traced_step(strategy: Any, run: Callable[[], Any]) -> st.SnapshotTrace:
    """Call ``run()`` with the session's hooks on ``strategy.env.models``, under one trace.

    Exceptions from ``run`` propagate; the hooks are removed either way.
    """
    hooks = ModelHooks(getattr(strategy.env, "models", None) or {}, lambda kind: View())
    previous = getattr(strategy, "snapshot_session", None)
    strategy.snapshot_session = SimpleNamespace(hooks=hooks)
    trace = st.SnapshotTrace(phase="train", step=0)
    try:
        with st.active(trace):
            run()
    finally:
        hooks.remove()
        strategy.snapshot_session = previous
    return trace


def records_a_model_stage(trace: st.SnapshotTrace) -> bool:
    return any(s.group == "model" and s.name.startswith("input[") for s in trace.stages)


__all__ = ["records_a_model_stage", "traced_step"]
