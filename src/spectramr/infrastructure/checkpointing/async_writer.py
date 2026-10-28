"""Checkpoint file writes on a background thread, for ``checkpoint.async_write``.

The calling thread still captures the state and runs the native (collective) save;
only the serialisation, the atomic publish and the retention prune move here.

A write is submitted as fatal or not, mirroring the synchronous call sites. A fatal
failure (a periodic, yield or final step) is sticky: it is raised at every later
``submit``, ``drain`` and ``close``, so a run that cannot checkpoint stops (#1972). A
non-fatal one (best, loss-schedule) is logged and kept in ``soft_failures`` for the
loop to report, as the synchronous fail-soft saves do (#713).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spectramr.infrastructure.checkpointing import codecs

logger = logging.getLogger(__name__)


class CheckpointWriteError(RuntimeError):
    """A queued checkpoint write failed; ``path`` is the file that did not land."""

    def __init__(self, path: Path, cause: BaseException) -> None:
        self.path = path
        super().__init__(f"Failed to save checkpoint {path}: {cause}")


@dataclass(frozen=True)
class SoftFailure:
    """A non-fatal write that did not land, in the loop's ``best_checkpoint_error`` form."""

    role: str
    path: Path
    message: str
    step: int = -1


def _copy(obj: Any, devices: set[str]) -> Any:
    if isinstance(obj, torch.Tensor):
        devices.add(obj.device.type)
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if type(obj) in (dict, OrderedDict):
        out = type(obj)((k, _copy(v, devices)) for k, v in obj.items())
        metadata = getattr(obj, "_metadata", None)
        if metadata is not None:
            # torch's state_dict versioning; the synchronous file carries it.
            out._metadata = metadata
        return out
    if type(obj) in (list, tuple):
        return type(obj)(_copy(v, devices) for v in obj)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_copy(v, devices) for v in obj))
    if isinstance(obj, Mapping | list | tuple | set | frozenset) and _holds_live_state(obj):
        # An unrecognised container would be written while training mutates its tensors.
        raise TypeError(
            f"Cannot snapshot a {type(obj).__name__} that holds tensors or arrays; "
            "convert it to a dict, list or tuple before checkpointing"
        )
    return obj


def _holds_live_state(obj: Any) -> bool:
    items = obj.values() if isinstance(obj, Mapping) else obj
    for item in items:
        if isinstance(item, torch.Tensor | np.ndarray):
            return True
        if isinstance(item, Mapping | list | tuple | set | frozenset) and _holds_live_state(item):
            return True
    return False


def snapshot_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    """A CPU copy of every tensor and array in ``state``, with the containers rebuilt.

    Passthrough ``state_dict()`` returns the live parameter and optimizer tensors,
    which the next step overwrites in place while the write is still queued.
    """
    devices: set[str] = set()
    copy = _copy(state, devices)
    if "cuda" in devices:
        torch.cuda.synchronize()
    return copy


class CheckpointWriter:
    """One background thread, at most one write in flight.

    The thread starts on the first ``submit``, so a rank that never writes holds a
    writer that costs nothing.
    """

    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._pending: Future[None] | None = None
        self._fatal: tuple[Path, BaseException] | None = None
        self._closed = False
        self.soft_failures: list[SoftFailure] = []

    def submit(
        self, path: Path, state: dict, then: Callable[[], None] | None, *, fatal: bool = True
    ) -> None:
        """Waits for the previous write (re-raising its error), then queues this one."""
        self.drain()
        self._closed = False
        snapshot = snapshot_to_cpu(state)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-writer")
        self._pending = self._executor.submit(self._write, Path(path), snapshot, then, fatal)

    def drain(self) -> None:
        """Block until idle; re-raise a failed write as RuntimeError (#1972)."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None
        if self._fatal is not None:
            path, cause = self._fatal
            # A fresh exception each time: re-raising one instance grows its traceback.
            raise CheckpointWriteError(path, cause) from cause

    def close(self, *, raise_fatal: bool = True) -> None:
        """Wait for the write in flight, stop the thread, and raise a fatal failure.

        ``raise_fatal=False`` is for teardown while another exception propagates: a
        raising callback would replace the run's real error, so the failure is logged.
        A second close is a no-op, so an exception-path teardown registered beside the
        normal-path close neither re-raises nor re-logs what that close reported.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.drain()
        except CheckpointWriteError as exc:
            if raise_fatal:
                raise
            logger.error("Checkpoint writer closed after a failed write: %s", exc)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def _write(self, path: Path, state: dict, then: Callable[[], None] | None, fatal: bool) -> None:
        try:
            reason = state["meta"]["reason"]
            codecs.write(path, state)
            logger.info("Checkpoint saved: %s (%s)", path, reason)
            if then is not None:
                then()
        except Exception as exc:
            logger.error("Failed to save checkpoint %s: %s", path, exc)
            if fatal:
                self._fatal = (path, exc)
            else:
                info = state.get("meta") or {}
                message = f"iter {info.get('step')}: {path}: {type(exc).__name__}: {exc}"
                self.soft_failures.append(
                    SoftFailure(info.get("role", "step"), path, message, info.get("step", -1))
                )


__all__ = ["CheckpointWriteError", "CheckpointWriter", "SoftFailure", "snapshot_to_cpu"]
