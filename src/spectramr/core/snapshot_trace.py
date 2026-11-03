"""The debug-snapshot trace: an ordered record of every stage one sample passed through.

Any layer calls :func:`record`. It costs one ``ContextVar`` read and returns when no
trace is active, so the call sites stay in the hot path on every step; only a step the
snapshot schedule selects opens a trace (``infrastructure/training/snapshot``), and only
then are tensors cloned to the host. The writer renders the stages in the order they
were recorded, which is the order the data flowed.

Stage groups, in pipeline order: ``loader`` (dataset and transforms), ``batch`` (the
collated batch), ``prepare`` (strategy-side preparation), ``model`` (forward hooks),
``strategy`` (named intermediates a strategy records), ``post`` (data consistency,
projection), and ``target``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

#: ``(label, num_channels, is_kspace[, log_compressed])`` -- one channel block of a
#: tensor whose channel axis superposes rendering treatments (cold diffusion's
#: ``cat([x_t, smaps_k])``). ``log_compressed=None`` inherits the tensor's answer.
ChannelSegment = tuple[str, int, bool] | tuple[str, int, bool, bool | None]


@dataclass(frozen=True)
class View:
    """How a recorded tensor is rendered; the stats are computed on it unchanged.

    ``authoritative`` means the k-space claim comes from the config, so the IFFT is
    applied without the spectrum veto; ``log_scaled`` means this tensor is on the
    compressed side of the arm's ``log1p``.
    """

    kspace: bool = False
    authoritative: bool = False
    log_scaled: bool = False
    segments: tuple[ChannelSegment, ...] | None = None


IMAGE = View()


@dataclass
class Stage:
    group: str
    name: str
    tensors: dict[str, torch.Tensor]
    views: dict[str, View]
    params: dict[str, Any]
    source: str


@dataclass
class SnapshotTrace:
    """The stages of one snapshot step, in recording order.

    ``batched`` traces keep sample 0 of every batched tensor; the loader trace is a
    single unbatched sample and keeps tensors whole. A stage name recorded more than
    ``max_repeats`` times (a model called inside a sampling loop) keeps its first
    calls and overwrites one ``#last`` slot, so a 1000-step reverse loop costs N + 1.
    """

    phase: str
    step: int
    batched: bool = True
    max_repeats: int = 2
    stages: list[Stage] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    #: Views a strategy declared for its model's input/output, consulted by the hooks.
    model_views: dict[str, View] = field(default_factory=dict)
    _seen: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(
        self,
        group: str,
        name: str,
        tensors: Mapping[str, Any],
        *,
        views: Mapping[str, View] | None,
        params: Mapping[str, Any] | None,
        source: str,
    ) -> None:
        key = (group, name)
        n = self._seen.get(key, 0)
        self._seen[key] = n + 1
        if n >= self.max_repeats:
            name = f"{name}#last"
            self.stages = [s for s in self.stages if (s.group, s.name) != (group, name)]
        elif n:
            name = f"{name}#{n}"
        kept: dict[str, torch.Tensor] = {}
        scalars: dict[str, Any] = dict(params or {})
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                kept[k] = self._capture(v)
            elif v is not None:
                scalars[k] = v
        if not kept and not scalars:
            return
        views = dict(views or {})
        self.stages.append(
            Stage(group, name, kept, {k: views.get(k, IMAGE) for k in kept}, scalars, source)
        )

    def _capture(self, t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if self.batched and t.dim() >= 3 and t.shape[0] > 1:
            t = t[:1]
        return t.to("cpu", copy=True)


_ACTIVE: ContextVar[SnapshotTrace | None] = ContextVar("spectramr_snapshot_trace", default=None)


def current() -> SnapshotTrace | None:
    """The active trace, or ``None`` on every step the schedule did not select."""
    return _ACTIVE.get()


def is_active() -> bool:
    return _ACTIVE.get() is not None


@contextmanager
def active(trace: SnapshotTrace) -> Iterator[SnapshotTrace]:
    token = _ACTIVE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE.reset(token)


@contextmanager
def suspended() -> Iterator[None]:
    """Record nothing inside, e.g. around a nested trace that owns its own stages."""
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def record(
    group: str,
    name: str,
    tensors: Mapping[str, Any],
    *,
    views: Mapping[str, View] | None = None,
    params: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any | Callable[[], Any]] | None = None,
) -> None:
    """Append one stage to the active trace; a no-op when none is active.

    ``extra`` values may be zero-arg callables; the writer resolves them, so a
    host sync they need is paid only on a snapshot step.
    """
    trace = _ACTIVE.get()
    if trace is None:
        return
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    source = (
        f"{caller.f_code.co_filename.rsplit('/src/', 1)[-1]}:{caller.f_code.co_name}"
        if caller
        else ""
    )
    trace.add(group, name, tensors, views=views, params=params, source=source)
    if extra:
        trace.extra.update({f"{group}/{name}.{k}": v for k, v in extra.items()})


def segments_tuple(segments: Sequence[ChannelSegment] | None) -> tuple[ChannelSegment, ...] | None:
    return tuple(segments) if segments else None


__all__ = [
    "IMAGE",
    "ChannelSegment",
    "SnapshotTrace",
    "Stage",
    "View",
    "active",
    "current",
    "is_active",
    "record",
    "segments_tuple",
    "suspended",
]
