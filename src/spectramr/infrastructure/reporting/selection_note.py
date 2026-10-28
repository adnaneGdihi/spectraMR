"""Mark an arm whose selected checkpoint is its first validation pass.

An early-stopping run that picks its first pass reports its least-trained weights, because the
metric only got worse after it (an anchor-only alpha = 0 control does this). Read beside an
arm selected hundreds of steps later, the pair compares training lengths, not methods.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from spectramr.infrastructure.reporting.aggregator import discover_artifacts


def _first_validation_step(csv_path: Path | None) -> int | None:
    if csv_path is None:
        return None
    with csv_path.open(newline="") as handle:
        steps = [int(float(r["iteration"])) for r in csv.DictReader(handle) if r.get("iteration")]
    return min(steps) if steps else None


def first_pass_selection_notes(run_dirs: Mapping[str, Path]) -> dict[str, str]:
    """``{arm: note}`` for each run whose best checkpoint is its first validation pass."""
    notes: dict[str, str] = {}
    for arm, run_dir in run_dirs.items():
        artifacts = discover_artifacts(run_dir)
        if artifacts.final_metrics_json is None:
            continue
        best = json.loads(artifacts.final_metrics_json.read_text()).get(
            "early_stopping_best_iteration"
        )
        first = _first_validation_step(artifacts.validation_metrics_csv)
        if isinstance(best, int) and first is not None and best <= first:
            notes[arm] = (
                f"selected its first validation pass (step {best}); compare it with arms "
                "selected later at one common step"
            )
    return notes


__all__ = ["first_pass_selection_notes"]
