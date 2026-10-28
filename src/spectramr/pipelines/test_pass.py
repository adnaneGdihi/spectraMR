"""The held-out test pass: one scored sweep of ``data.source.test_index_path`` after training.

It runs on the weights the run ends with (the best ones under ``restore_best_weights``) and
writes ``final_eval.json`` -- ``{metric: {case_id: value}}``, the file the report reads as the
``test`` split -- plus ``test_metrics.json`` stating what was scored. Nothing here feeds
checkpoint selection or early stopping.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

from spectramr.infrastructure.builders.directors.test_split import declares_test_set
from spectramr.infrastructure.reporting.metric_names import SPLIT_PREFIXES
from spectramr.infrastructure.reporting.test_split.degradation_guard import DegradationGuard
from spectramr.infrastructure.reporting.test_split.report import write_test_report
from spectramr.infrastructure.reporting.test_split.slice_writer import (
    SLICES_SUBDIR,
    HeldOutSliceWriter,
)
from spectramr.infrastructure.training.validation_rows import IDENTIFIER_KEYS

logger = logging.getLogger(__name__)

TEST_SUBDIR = "test"
FINAL_EVAL_JSON = "final_eval.json"
TEST_SUMMARY_JSON = "test_metrics.json"
#: Columns of ``validation_batches.csv`` that are not metrics.
_ROW_META = frozenset({"iteration", "epoch", "batch_index", "n_samples", *IDENTIFIER_KEYS})


def _bare(metric: str) -> str:
    for prefix in SPLIT_PREFIXES:
        if metric.startswith(prefix):
            return metric[len(prefix) :]
    return metric


def case_id(row: dict[str, str]) -> str:
    """The row's identity: the loader's identifiers, so the same slice pairs across arms."""
    parts = [str(row.get(k, "")) for k in ("subject_id", "contrast", "file_id", "slice_index")]
    named = "/".join(p for p in parts if p)
    return named or f"batch{row.get('batch_index', '')}"


def final_eval_from_rows(rows_csv: Path) -> dict[str, dict[str, float]]:
    """``{metric: {case_id: value}}`` from a pass's per-batch rows; non-finite values are left out."""
    out: dict[str, dict[str, float]] = {}
    with rows_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cid = case_id(row)
            for name, raw in row.items():
                if name in _ROW_META or raw in ("", None):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value):
                    out.setdefault(_bare(name), {})[cid] = value
    return out


def run_test_pass(
    loop: Any,
    pipeline: Any,
    strategy: Any,
    config: Any,
    run_dir: Path,
    *,
    iteration: int,
    weights: str,
    weights_iteration: int | None = None,
    write: bool = True,
) -> dict[str, Any] | None:
    """Score the held-out test set once; ``None`` when the arm declares none.

    Under DDP every rank scores its sampler shard and the means are all-reduced, but the per-case
    rows and ``final_eval.json`` are rank 0's shard only. ``weights`` says which weights were scored ("best", "final", or the checkpoint a validate-only
    run loaded). Per-case rows need ``validation.per_batch_rows``; without them ``final_eval.json`` holds
    the pass means as scalars, which the report reads the same way but cannot pair.
    """
    if not declares_test_set(config):
        return None
    out_dir = Path(run_dir) / TEST_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # Beside its own training CSV name, so the pass writes `test/validation_*.csv`, not the run's.
    csv_file = out_dir / "training_metrics.csv"
    for stale in ("validation_batches.csv", "validation_metrics.csv"):  # appended to, not replaced
        (out_dir / stale).unlink(missing_ok=True)

    loaders = pipeline.data_loaders
    # Built with the environment (fail-fast on a missing manifest, sampler-wrapped under DDP).
    if loaders.get("test") is None:
        raise ValueError(
            "data.source.test_index_path is declared but the environment holds no test loader."
        )
    # The report's case sinks are fed by the validation loop; this pass must not add to them.
    sinks = {
        k: getattr(strategy, k, None) for k in ("_per_case_metric_sink", "_report_case_recorder")
    }
    for key in sinks:
        if sinks[key] is not None:
            setattr(strategy, key, None)
    # Every slice this rank scores is rendered; the guard reads the first batch.
    writer = None
    if write:
        shutil.rmtree(out_dir / SLICES_SUBDIR, ignore_errors=True)
        writer = HeldOutSliceWriter(out_dir / SLICES_SUBDIR, DegradationGuard.from_config(config))
    strategy._test_slice_writer = writer
    try:
        metrics = loop.evaluate(iteration=iteration, epoch=0, csv_file=csv_file, split="test")
    finally:
        strategy._test_slice_writer = None
        for key, sink in sinks.items():
            if sink is not None:
                setattr(strategy, key, sink)
    if writer is not None and writer.poisoned is not None:
        raise ValueError(f"[Test] refusing to report the test pass: {writer.poisoned}.")

    if not write:
        return {"metrics": {k: float(v) for k, v in metrics.items()}}
    rows_csv = out_dir / "validation_batches.csv"
    per_case = final_eval_from_rows(rows_csv) if rows_csv.is_file() else {}
    payload: dict[str, Any] = per_case or {_bare(k): float(v) for k, v in metrics.items()}
    (Path(run_dir) / FINAL_EVAL_JSON).write_text(json.dumps(payload, indent=2, sort_keys=True))
    n_cases = max((len(v) for v in per_case.values() if isinstance(v, dict)), default=0)
    summary = {
        "split": "test",
        "manifest": str(
            config.data.source.test_index_path
            or f"{config.data.source.paired_manifest_path} (config-owned test partition)"
        ),
        "n_cases": n_cases,
        "per_case": bool(per_case),
        "iteration": int(iteration),
        "weights": weights,
        # The step whose weights were scored: `iteration` is where the loop stopped,
        # which is not the restored best checkpoint (84001 vs 54000 on one arm).
        "weights_iteration": None if weights_iteration is None else int(weights_iteration),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    (Path(run_dir) / TEST_SUMMARY_JSON).write_text(json.dumps(summary, indent=2))
    if writer is not None:
        writer.write_index()
        if not writer.rows:
            logger.info(
                "[Test] %s does not feed the slice writer from its validation_step; no test "
                "slice was rendered.",
                type(strategy).__name__,
            )
        write_test_report(out_dir, Path(run_dir) / TEST_SUMMARY_JSON)
    logger.info(
        "[Test] %d cases from %s scored on the %s weights", n_cases, summary["manifest"], weights
    )
    return summary


__all__ = ["case_id", "final_eval_from_rows", "run_test_pass"]
