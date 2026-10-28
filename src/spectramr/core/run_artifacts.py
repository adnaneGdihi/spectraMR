"""Names of run-directory artifacts that more than one layer reads or writes."""

from __future__ import annotations

import json
from pathlib import Path

#: ``<arm>/inference/``: the held-out test pass `campaign infer` re-runs from a saved checkpoint.
INFERENCE_SUBDIR = "inference"
#: ``<arm>/inference/parity.json``: whether that re-run reproduced the training-time test pass.
PARITY_JSON = "parity.json"


def write_parity(
    run_dir: str | Path, *, ok: bool, reason: str | None, rtol: float, atol: float
) -> Path:
    path = Path(run_dir) / INFERENCE_SUBDIR / PARITY_JSON
    path.write_text(
        json.dumps({"ok": bool(ok), "reason": reason, "rtol": rtol, "atol": atol}, indent=2)
    )
    return path


def read_parity(run_dir: str | Path) -> str:
    """``ok``, ``mismatch`` or ``unchecked`` (no verdict, or one that cannot be read)."""
    try:
        verdict = json.loads((Path(run_dir) / INFERENCE_SUBDIR / PARITY_JSON).read_text())
    except (OSError, ValueError):
        return "unchecked"
    return "ok" if verdict.get("ok") is True else "mismatch"


__all__ = ["INFERENCE_SUBDIR", "PARITY_JSON", "read_parity", "write_parity"]
