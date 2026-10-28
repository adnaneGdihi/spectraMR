r"""The CSV and JSON a contrast folder holds under ``data/``.

Split from :mod:`.contrast_folder`, which owns the layout, so each file stays under the
size ceiling and the tables can be tested without rendering a figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contrast_stats import PAIRED_COLUMNS, ContrastStats, base_of, paired_subject_frame
from .contrasts import ContrastSet
from .metric_selection import discover_metrics, select_metrics
from .phase_tables import inference_unavailable, phase_table, wide


def all_metrics(arms: Any) -> pd.DataFrame:
    """Every validation metric of each arm at the pass the report reads, one row per arm."""
    val = arms.df[arms.df["split"] == "val"]
    rows = []
    for lab in arms.labels:
        step = arms.steps.get(lab)
        mine = val[(val["method"] == lab) & (val["step"] == step)]
        rows.append({"arm": lab, "step": step, **mine.groupby("metric")["value"].last().to_dict()})
    return pd.DataFrame(rows)


def inference_skips(arms: Any, phases: tuple[str, ...], known: list[dict]) -> list[dict]:
    """Metrics an inference run declared and skipped, with the reason it recorded.

    Listed per arm whether or not the contrast asked for the metric, so the folder
    says why a full-reference metric is absent from the inference table.
    """
    if "test" not in phases:
        return []
    seen = {(u["phase"], u["metric"], u["variant"]) for u in known}
    found = []
    for arm in arms.labels:
        for u in inference_unavailable(arms.run_dirs[arm], arm):
            if (u["phase"], u["metric"], arm) not in seen:
                found.append(
                    {"phase": "test", "metric": u["metric"], "variant": arm, "reason": u["reason"]}
                )
    return found


def write_phase_tables(data: Path, arms: Any, contrast_set: ContrastSet, phases, window) -> dict:
    """The per-phase metric tables; returns the metric list actually used in each phase."""
    found = discover_metrics(arms, phases)
    spec = [base_of(m) for m in contrast_set.metrics] if contrast_set.metrics else "all"
    selected = select_metrics(spec, found) or {}
    table = phase_table(arms, selected, train_window=window)
    table.to_csv(data / "metrics_long.csv", index=False)
    for phase in selected:
        wide(table, phase).to_csv(data / f"metrics_{phase}.csv", index=False)
    return selected


def write_data(
    folder: Path,
    arms: Any,
    stats: ContrastStats,
    rows: list[dict[str, Any]],
    baseline: str,
    phase_args: tuple[ContrastSet, tuple[str, ...], int],
) -> dict:
    data = folder / "data"
    (data / "paired_subjects").mkdir(parents=True, exist_ok=True)
    arms.df.to_csv(data / "arm_metrics.csv", index=False)
    all_metrics(arms).to_csv(data / "all_metrics.csv", index=False)
    arms.per_case.to_csv(data / "per_case.csv", index=False)
    pd.DataFrame(rows, columns=list(PAIRED_COLUMNS)).to_csv(data / "paired_stats.csv", index=False)
    for (phase, metric, variant), paired in stats.paired.items():
        name = f"{phase}__{metric}__{variant}_minus_{baseline}.csv"
        paired_subject_frame(paired, variant, baseline).to_csv(data / "paired_subjects" / name)
    skipped = inference_skips(arms, phase_args[1], stats.unavailable)
    (data / "unavailable.json").write_text(json.dumps(stats.unavailable + skipped, indent=2) + "\n")
    return write_phase_tables(data, arms, *phase_args)


__all__ = ["all_metrics", "inference_skips", "write_data", "write_phase_tables"]
