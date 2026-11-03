"""Re-fetch one dataset sample in the main process so its loader stages are traced.

Training samples are built in DataLoader workers, where no trace is active. On a
snapshot step the session fetches ``dataset[index]`` once more here, seeded from the
step, and the traced Compose plus any dataset-level ``record`` calls land in the
step's trace. Every random state the fetch could advance -- torch, NumPy, ``random``,
and the generators the dataset and its transforms hold -- is saved first and
restored after, so the training stream is unchanged whether or not snapshots run.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from spectramr.core.snapshot_trace import SnapshotTrace

_MAX_DEPTH = 4
#: A file list of thousands of subjects holds no generator; do not walk all of it.
_MAX_CHILDREN = 64


def _generators(obj: Any, depth: int = 0, seen: set[int] | None = None) -> Iterator[Any]:
    """Every RNG object reachable from ``obj`` through attributes, lists and dicts."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > _MAX_DEPTH:
        return
    seen.add(id(obj))
    if isinstance(
        obj, (torch.Generator, np.random.Generator, np.random.RandomState, random.Random)
    ):
        yield obj
        return
    if isinstance(obj, (torch.Tensor, np.ndarray, str, bytes, int, float)):
        return
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    else:
        children = list(getattr(obj, "__dict__", {}).values())
    for child in children[:_MAX_CHILDREN]:
        yield from _generators(child, depth + 1, seen)


def _state(g: Any) -> Any:
    if isinstance(g, torch.Generator):
        return g.get_state()
    if isinstance(g, np.random.Generator):
        return g.bit_generator.state
    if isinstance(g, np.random.RandomState):
        return g.get_state()
    return g.getstate()


def _restore(g: Any, state: Any) -> None:
    if isinstance(g, torch.Generator):
        g.set_state(state)
    elif isinstance(g, np.random.Generator):
        g.bit_generator.state = state
    elif isinstance(g, np.random.RandomState):
        g.set_state(state)
    else:
        g.setstate(state)


def source_dataset(dataset: Any) -> Any:
    """The per-subject dataset behind a TorchIO patch ``Queue``; else ``dataset``."""
    return getattr(dataset, "subjects_dataset", dataset)


def trace_sample(trace: SnapshotTrace, dataset: Any, index: int, seed: int) -> None:
    """Fetch ``dataset[index]`` under ``trace`` without advancing any training RNG."""
    ds = source_dataset(dataset)
    held = [(g, _state(g)) for g in _generators(ds)]
    py_state, np_state = random.getstate(), np.random.get_state()
    batched = trace.batched
    trace.batched = False
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            ds[index % len(ds)]
    finally:
        trace.batched = batched
        random.setstate(py_state)
        np.random.set_state(np_state)
        for g, state in held:
            _restore(g, state)


__all__ = ["source_dataset", "trace_sample"]
