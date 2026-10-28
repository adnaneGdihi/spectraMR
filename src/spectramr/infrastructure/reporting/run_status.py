r"""Whether a run finished, read from ``run_summary.json`` ``success``.

This module is the reporting layer's only reader of that flag. Three states:

* ``ok``: ``success`` is ``true``.
* ``failed``: ``success`` is ``false``. The training pipeline writes this when
  a run crashes, and the metric files left in the directory may belong to an
  earlier attempt. A comparison excludes such an arm from every table and
  figure and names it in the summary, the manifest and the table caption.
* ``unknown``: there is no ``run_summary.json``, the file is unreadable, it has
  no ``success`` key, or the key is ``null``. Runs downloaded before the flag
  existed look like this. They are kept, but every table marks them. An
  unknown run is never reported as a finished one.

Any other value of ``success`` raises, because guessing would decide the
question this module exists to answer.

It lives beside :mod:`.aggregator` rather than inside it so the aggregator
stays under the 300-line ceiling. It finds the file through
:func:`.aggregator.discover_artifacts`, which owns the root-versus-``logs/``
lookup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .aggregator import discover_artifacts

logger = logging.getLogger(__name__)

State = Literal["ok", "failed", "unknown"]


@dataclass(frozen=True)
class RunStatus:
    """The completion state of one run and the evidence it rests on."""

    state: State
    reason: str = ""
    error_type: str | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        """Short text for table footnotes and log lines, e.g. ``crashed: RuntimeError``."""
        if self.state == "ok":
            return "ok"
        if self.state == "failed":
            return f"crashed: {self.error_type or 'error type not recorded'}"
        return f"status unknown: {self.reason}"

    @property
    def error_line(self) -> str:
        """The last line of the recorded error, which names the exception.

        The full ``error`` field can hold a traceback, which must not be copied
        into logs or tables.
        """
        lines = [ln.strip() for ln in (self.error or "").splitlines() if ln.strip()]
        return lines[-1][:160] if lines else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.state,
            "reason": self.reason,
            "error_type": self.error_type,
            "error": self.error_line or None,
        }


def read_run_status(run_dir: str | Path) -> RunStatus:
    """Read the completion state of ``run_dir`` from its ``run_summary.json``."""
    path = discover_artifacts(run_dir).run_summary_json
    if path is None:
        return RunStatus("unknown", "no run_summary.json")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return RunStatus("unknown", f"run_summary.json unreadable ({type(exc).__name__})")
    if not isinstance(payload, dict):
        return RunStatus("unknown", "run_summary.json is not a JSON object")
    if "success" not in payload:
        return RunStatus("unknown", "run_summary.json has no success flag")
    success = payload["success"]
    if success is None:
        return RunStatus("unknown", "run_summary.json success is null")
    if success is True:
        return RunStatus("ok")
    if success is False:
        error_type = payload.get("error_type")
        error = payload.get("error")
        return RunStatus(
            "failed",
            "run_summary.json success is false",
            error_type=str(error_type) if error_type else None,
            error=str(error) if error else None,
        )
    raise ValueError(
        f"{path}: run_summary.json 'success' must be true, false or null, got {success!r}"
    )


def log_status(name: str, status: RunStatus, *, action: str) -> None:
    """Log a run that is not ``ok``: a warning when it failed, info when unknown."""
    if status.state == "ok":
        return
    detail = f" ({status.error_line})" if status.error_line else ""
    level = logging.WARNING if status.state == "failed" else logging.INFO
    logger.log(level, "run %s: %s%s; %s", name, status.label, detail, action)


def status_notes(statuses: dict[str, RunStatus]) -> dict[str, str]:
    """``{method: label}`` for every run that is not ``ok``, for table footnotes."""
    return {name: s.label for name, s in statuses.items() if s.state != "ok"}


__all__ = ["RunStatus", "log_status", "read_run_status", "status_notes"]
