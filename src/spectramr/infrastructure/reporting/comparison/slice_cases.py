"""Per-case rows for the cross-run comparison, from a run's per-slice validation rows.

``spectramr report --compare`` draws its error bars and paired tests from per-case
rows. A run that writes ``per_call_metrics.csv`` supplies them through
:mod:`..per_case`; a run that writes only ``validation_batches.csv``
(``validation.per_batch_rows``, the n2n and self_supervised cohorts) supplies them
here (#2500). :mod:`..per_slice` stays the one reader of those rows, so the pass a
run is read at and the replay rule are the ones the per-subject statistics use.

Each slice is one case, and ``pair_key`` is its identity (subject, contrast,
slice, repetition group), never its batch index: the loaders shuffle, so batch
``i`` of one arm is a different slice from batch ``i`` of another.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..metric_names import parse_metric_name
from ..per_slice import find_rows_files, read_pass

_COLUMNS = [
    "run_id",
    "method",
    "case_id",
    "acceleration",
    "metric",
    "base_metric",
    "value",
    "pair_key",
    "step",
]


def slice_case_rows(
    run_dir: str | Path, *, step: int | None, run_id: str, method: str
) -> pd.DataFrame:
    """One row per (slice, metric) at the pass ``step`` selects; empty without rows."""
    empty = pd.DataFrame(columns=_COLUMNS)
    files = find_rows_files(run_dir)
    if not files:
        return empty
    try:
        rows = read_pass(files, step)
    except ValueError:
        # No pass at or before the checkpoint: nothing to report, never every pass.
        return empty
    frame = rows.frame.apply(pd.to_numeric, errors="coerce")
    frame.index = ["|".join(str(part) for part in key) for key in frame.index]
    long = (
        frame.rename_axis("case_id")
        .reset_index()
        .melt(id_vars="case_id", var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )
    parsed = long["metric"].map(parse_metric_name)
    keep = parsed.map(lambda n: n.aggregate is None)
    long, parsed = long[keep], parsed[keep]
    long["base_metric"] = parsed.map(lambda n: n.base)
    long["acceleration"] = parsed.map(lambda n: n.acceleration).astype(float)
    long["pair_key"] = "slice#" + long["case_id"]
    long["run_id"] = run_id
    long["method"] = method
    long["step"] = rows.iteration
    return long[_COLUMNS].reset_index(drop=True)


__all__ = ["slice_case_rows"]
