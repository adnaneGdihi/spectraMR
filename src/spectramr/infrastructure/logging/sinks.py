"""The run's log sinks behind one registry, and the one rule for where they live.

Every output a run writes -- the text log, TensorBoard events, the metric and loss
CSVs -- is a sink owned by the ``LoggingService``. The writers stay where they are
(``MetricsTracker`` owns the CSVs, ``TensorBoardWriter`` the events); a sink is the
handle that names the destination, reports whether it is live, and closes it, so
``LoggingService.sink_records()`` can answer "where did this run write?" in one call.
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

__all__ = [
    "ArtifactSink",
    "FileSink",
    "ILogSink",
    "SinkRecord",
    "SinkRegistry",
    "TensorBoardSink",
    "resolve_run_dir",
    "resolve_sinks_dir",
]


class ILogSink(Protocol):
    """A destination a run writes to."""

    name: str

    @property
    def enabled(self) -> bool: ...

    @property
    def path(self) -> str | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SinkRecord:
    """What provenance stores about one sink."""

    name: str
    enabled: bool
    path: str | None


def resolve_run_dir(config: Any, is_sanity_check: bool = False) -> Path:
    """The run's output directory; sanity checks are redirected beneath it."""
    output_dir = "experiments/outputs"
    if config.training and config.training.output_dir:
        output_dir = config.training.output_dir
    if is_sanity_check:
        # Prevent completely overwriting the real experiment output dir.
        output_dir = str(Path(output_dir) / "sanity_check")
    return Path(output_dir)


def resolve_sinks_dir(logging_config: Any, run_dir: str | Path) -> tuple[str, str]:
    """``(directory, source)``: a declared ``logging.sinks.dir`` wins, else ``<run_dir>/logs``.

    ``model_fields_set`` separates "declared" from "defaulted" -- the schema default
    (``./logs``) would otherwise be indistinguishable from a deliberate choice and
    land the text log in the working directory, away from the run it describes.
    """
    sinks = logging_config.sinks
    if "dir" in sinks.model_fields_set:
        return str(sinks.dir), "declared"
    return str(Path(run_dir) / "logs"), "run_dir"


class FileSink:
    """The run's text log, one file per rank as the writing policy decides."""

    name = "file"

    def __init__(self, logger: logging.Logger, log_dir: str, filename: str) -> None:
        self._logger = logger
        self.resolved_path: str | None = None
        self.relocated_from: str | None = None
        self._handler: logging.FileHandler | None = None
        self._open(log_dir, filename)

    def _attach(self, path: str) -> None:
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
        self._logger.addHandler(handler)
        self._handler, self.resolved_path = handler, path

    def _open(self, log_dir: str, filename: str) -> None:
        try:
            os.makedirs(log_dir, exist_ok=True)
            self._attach(os.path.join(log_dir, filename))
        except OSError as exc:
            # A relocated log sits in a temp dir that is wiped at job teardown, so say so
            # on the log itself and as a warning: the first is the message most likely lost.
            self._attach(os.path.join(tempfile.mkdtemp(prefix="spectramr_logs_"), filename))
            self.relocated_from = str(log_dir)
            notice = (
                f"log directory {log_dir!r} is not writable ({type(exc).__name__}: {exc}); "
                f"this run's ENTIRE log has been relocated to {self.resolved_path!r}. On a "
                "compute node that temporary directory is wiped at job teardown, so copy the "
                "log out before the job ends or set logging.sinks.dir to a writable path. The "
                "relocation is recorded in provenance.json under `logging`."
            )
            self._logger.warning(notice)
            warnings.warn(notice, RuntimeWarning, stacklevel=4)

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    @property
    def path(self) -> str | None:
        return self.resolved_path

    def close(self) -> None:
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None


class TensorBoardSink:
    """Handle on the run's ``TensorBoardWriter`` (a disabled writer is a disabled sink)."""

    name = "tensorboard"

    def __init__(self, writer: Any) -> None:
        self.writer = writer

    @property
    def enabled(self) -> bool:
        return bool(self.writer)

    @property
    def path(self) -> str | None:
        return str(self.writer.event_dir) if self.enabled else None

    def close(self) -> None:
        close = getattr(self.writer, "close", None)
        if callable(close):
            close()


class ArtifactSink:
    """A destination another component writes (``metrics_csv``, ``loss_csv``).

    ``path`` is read lazily so the record names the file the writer actually uses.
    """

    def __init__(self, name: str, path: Callable[[], str | Path | None]) -> None:
        self.name = name
        self._path = path

    @property
    def enabled(self) -> bool:
        return self._path() is not None

    @property
    def path(self) -> str | None:
        value = self._path()
        return None if value is None else str(value)

    def close(self) -> None:
        """Nothing to close: the writer that owns the file closes it."""


class SinkRegistry:
    """Named sinks; a second sink under one name raises rather than shadowing the first."""

    def __init__(self) -> None:
        self._sinks: dict[str, ILogSink] = {}

    def add(self, sink: ILogSink, *, replace: bool = False) -> ILogSink:
        if sink.name in self._sinks:
            if not replace:
                raise ValueError(f"log sink {sink.name!r} is already registered")
            self._sinks[sink.name].close()
        self._sinks[sink.name] = sink
        return sink

    def get(self, name: str) -> ILogSink:
        try:
            return self._sinks[name]
        except KeyError:
            raise KeyError(f"no log sink {name!r}; registered: {sorted(self._sinks)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._sinks

    def records(self) -> list[SinkRecord]:
        return [SinkRecord(s.name, s.enabled, s.path) for s in self._sinks.values()]

    def close_all(self) -> None:
        for sink in self._sinks.values():
            sink.close()
        self._sinks.clear()
