r"""Tidy-DataFrame aggregator for the reporting pipeline.

Per `TODO/report_step` Phase 3: a single ``aggregate(experiment_dir)``
call merges the per-step training log, validation log, any final
evaluation JSON, the per-arm ``final_metrics.json`` (best/final scalars),
and the run-level ``run_summary.json`` facts (params, throughput,
wall-time) into a long-format DataFrame:

    columns: run_id, method, seed, epoch, step, metric, value, split

``split`` values: ``train`` / ``val`` (per-step), ``test`` (eval JSON),
``best`` / ``final`` (final_metrics.json), ``run`` (run_summary.json).

Plotters consume only this canonical shape, so adding a new figure type
never requires re-parsing logs.

The aggregator is intentionally tolerant — if any of the expected
artifacts are missing, the corresponding rows are simply absent
(plotters that need them will report a clear "no data" message rather
than crashing the run).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from spectramr.core.run_artifacts import INFERENCE_SUBDIR

from .metric_names import annotate_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Discovered artifact paths for a single experiment run."""

    experiment_dir: Path
    training_metrics_csv: Path | None
    validation_metrics_csv: Path | None
    final_eval_json: Path | None
    final_metrics_json: Path | None
    config_yaml: Path | None
    run_summary_json: Path | None
    report_cases_dir: Path | None
    #: Where ``final_eval_json`` came from; see :func:`eval_source`.
    eval_source: str = "none"


def eval_source(experiment_dir: str | Path) -> str:
    """``inference`` when ``campaign infer`` re-ran the test pass, ``training`` for the run's own, else ``none``."""
    p = Path(experiment_dir)
    if (p / INFERENCE_SUBDIR / "final_eval.json").exists():
        return "inference"
    return "training" if (p / "final_eval.json").exists() else "none"


def discover_artifacts(experiment_dir: str | Path) -> ExperimentArtifacts:
    """Walk an experiment output directory looking for the canonical files.

    Recognises the layout produced by `src/pipelines/train.py`:

        <experiment_dir>/
          ├── logs/
          │   ├── training_metrics.csv
          │   ├── validation_metrics.csv
          │   └── run_summary.json   (optional)
          ├── final_eval.json        (optional, written by inference pipelines)
          ├── inference/final_eval.json  (optional, `campaign infer`; read in preference)
          ├── final_metrics.json     (optional, per-arm campaign contract)
          ├── run_summary.json       (optional, run-level facts)
          └── config.yaml            (optional snapshot)
    """
    p = Path(experiment_dir)
    logs = p / "logs"
    source = eval_source(p)
    # The re-run test pass (`campaign infer`) is the report's inference phase when it exists; the
    # training-time one stays on disk beside it for comparison.
    final_eval = p / INFERENCE_SUBDIR / "final_eval.json" if source == "inference" else p / "final_eval.json"
    # run_summary.json historically lived under logs/; the training pipeline
    # now writes it at the run root — accept either location.
    run_summary = next(
        (c for c in (p / "run_summary.json", logs / "run_summary.json") if c.exists()),
        None,
    )
    return ExperimentArtifacts(
        experiment_dir=p,
        training_metrics_csv=(logs / "training_metrics.csv")
        if (logs / "training_metrics.csv").exists()
        else None,
        validation_metrics_csv=(logs / "validation_metrics.csv")
        if (logs / "validation_metrics.csv").exists()
        else None,
        final_eval_json=final_eval if source != "none" else None,
        eval_source=source,
        final_metrics_json=(p / "final_metrics.json")
        if (p / "final_metrics.json").exists()
        else None,
        config_yaml=(p / "config.yaml") if (p / "config.yaml").exists() else None,
        run_summary_json=run_summary,
        report_cases_dir=(p / "report_cases") if (p / "report_cases").exists() else None,
    )


def _melt_metrics_csv(csv_path: Path | None, split: str, run_id: str, method: str) -> pd.DataFrame:
    """Read a wide metrics CSV and melt to long form."""
    if not csv_path or not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    id_vars = [c for c in ("epoch", "step", "iteration") if c in df.columns]
    value_vars = [c for c in df.columns if c not in id_vars]
    melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="metric", value_name="value")
    melted["split"] = split
    melted["run_id"] = run_id
    melted["method"] = method
    # Normalise epoch/step columns into a single 'step' (use first non-null id_var)
    if "step" not in melted.columns and "iteration" in melted.columns:
        melted["step"] = melted["iteration"]
    if "step" not in melted.columns and "epoch" in melted.columns:
        melted["step"] = melted["epoch"]
    # Drop metrics that are entirely empty (e.g. the val_loss / val_total_loss /
    # val_recon_loss headers pre-seeded into the training CSV but never written)
    # so they never surface as phantom all-NaN series in the learning-curve figure.
    if not melted.empty:
        melted = melted[melted.groupby("metric")["value"].transform("count") > 0]
    return melted


def _flatten_eval_json(eval_path: Path | None, run_id: str, method: str) -> pd.DataFrame:
    """Flatten a final-evaluation JSON into long-form rows.

    Supports two shapes:
      1. ``{metric: scalar, ...}`` — one row per metric.
      2. ``{metric: {subject_id: scalar, ...}, ...}`` — one row per (metric, subject).
    """
    if not eval_path or not eval_path.exists():
        return pd.DataFrame()
    with eval_path.open() as f:
        payload: Any = json.load(f)
    rows: list[dict] = []
    for metric, value in payload.items():
        if isinstance(value, dict):
            for subject_id, scalar in value.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "metric": metric,
                        "subject_id": subject_id,
                        "value": float(scalar) if scalar is not None else None,
                        "split": "test",
                        "step": -1,
                    }
                )
        else:
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "metric": metric,
                    "value": float(value) if value is not None else None,
                    "split": "test",
                    "step": -1,
                }
            )
    return pd.DataFrame(rows)


def _flatten_final_metrics(path: Path | None, run_id: str, method: str) -> pd.DataFrame:
    """Flatten a ``final_metrics.json`` (per-arm campaign contract, schema v1).

    ``best.{metric}_best`` entries become ``(metric, split="best", step=-1)``
    rows (the ``_best`` suffix is stripped — the split already encodes it);
    ``final_loss`` / ``early_stopping_best_value`` become ``split="final"`` rows.
    """
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        with path.open() as f:
            payload: Any = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("final_metrics.json unreadable at %s: %s", path, exc)
        return pd.DataFrame()
    rows: list[dict] = []

    def _row(metric: str, value: Any, split: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "metric": metric,
                    "value": float(value),
                    "split": split,
                    "step": -1,
                }
            )

    for key, value in (payload.get("best") or {}).items():
        _row(key.removesuffix("_best"), value, "best")
    for key in ("final_loss", "early_stopping_best_value", "early_stopping_best_iteration"):
        _row(key, payload.get(key), "final")
    return pd.DataFrame(rows)


# run_summary.json keys folded into the tidy frame, as
# ``source_key: (metric_name, scale)``. ``params_m`` feeds the headline-Pareto
# cost axis; the rest feed the computational profile + run-summary card.
_RUN_SUMMARY_FACTS: dict[str, tuple[str, float]] = {
    "model_params": ("params_m", 1e-6),
    "iterations_per_sec": ("iterations_per_sec", 1.0),
    "duration_sec": ("duration_min", 1.0 / 60.0),
    "effective_batch": ("effective_batch", 1.0),
}


def _flatten_run_summary(path: Path | None, run_id: str, method: str) -> pd.DataFrame:
    """Flatten run-level facts from ``run_summary.json`` into ``split="run"`` rows."""
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        with path.open() as f:
            payload: Any = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("run_summary.json unreadable at %s: %s", path, exc)
        return pd.DataFrame()
    rows: list[dict] = []
    for key, (metric, scale) in _RUN_SUMMARY_FACTS.items():
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "metric": metric,
                    "value": float(value) * scale,
                    "split": "run",
                    "step": -1,
                }
            )
    return pd.DataFrame(rows)


def aggregate(
    experiment_dir: str | Path,
    *,
    method_name: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Build the canonical long-format DataFrame for a single experiment.

    Args:
        experiment_dir: Root output directory of one training run.
        method_name: Tag for the ``method`` column. Defaults to the
            directory's basename.
        run_id: Tag for the ``run_id`` column. Defaults to ``method_name``.

    Returns:
        Long-format DataFrame with columns
        ``[run_id, method, split, step, metric, value]`` plus the parsed
        ``base_metric`` / ``acceleration`` / ``headline`` columns
        (:mod:`.metric_names`), and ``subject_id`` where the eval JSON provided
        per-subject data.
    """
    artifacts = discover_artifacts(experiment_dir)
    method = method_name or artifacts.experiment_dir.name
    rid = run_id or method

    parts = [
        _melt_metrics_csv(artifacts.training_metrics_csv, "train", rid, method),
        _melt_metrics_csv(artifacts.validation_metrics_csv, "val", rid, method),
        _flatten_eval_json(artifacts.final_eval_json, rid, method),
        _flatten_final_metrics(artifacts.final_metrics_json, rid, method),
        _flatten_run_summary(artifacts.run_summary_json, rid, method),
    ]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        logger.warning("No metric artifacts found under %s", experiment_dir)
        return pd.DataFrame(columns=["run_id", "method", "split", "step", "metric", "value"])
    return annotate_frame(pd.concat(parts, ignore_index=True, sort=False))


def aggregate_many(
    experiment_dirs: list[str | Path],
    *,
    method_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame across multiple runs (e.g. multiple seeds).

    If ``method_names`` is None, each run's directory basename becomes its
    method tag — group by ``method`` to aggregate over seeds at plot time.
    """
    if method_names is not None and len(method_names) != len(experiment_dirs):
        raise ValueError(
            f"method_names len {len(method_names)} != experiment_dirs len {len(experiment_dirs)}"
        )
    frames: list[pd.DataFrame] = []
    for i, d in enumerate(experiment_dirs):
        m = method_names[i] if method_names is not None else None
        frames.append(aggregate(d, method_name=m, run_id=f"{m or Path(d).name}_run{i}"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
