"""Per-slice test metrics, summarised overall, per contrast and per acquisition situation.

The test pass scores one slice per batch row (``test/validation_batches.csv``); the slice
writer's ``slice_index.csv`` adds the rendered panel and the realised acceleration. This module
joins the two into ``per_slice_metrics.csv`` and reduces it to ``summary_by_group.csv``: count,
mean, std, median, quartiles and range of every metric for each grouping the data supports.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.test_split.slice_writer import (
    SLICE_INDEX_CSV,
    SLICES_SUBDIR,
)

ROWS_CSV = "validation_batches.csv"
PER_SLICE_CSV = "per_slice_metrics.csv"
SUMMARY_CSV = "summary_by_group.csv"
#: A per-rung metric column (``val_psnr_8x``) names its acceleration in the suffix.
_RUNG = re.compile(r"^(?P<base>.+)_(?P<rate>\d+(?:\.\d+)?)x$")
_STATS = ("n", "mean", "std", "median", "q25", "q75", "min", "max")


def per_slice_table(test_dir: Path) -> pd.DataFrame:
    """The test rows joined to the slice index on ``batch_index``; empty without rows."""
    test_dir = Path(test_dir)
    rows_path = test_dir / ROWS_CSV
    if not rows_path.is_file():
        return pd.DataFrame()
    rows = pd.read_csv(rows_path, dtype={"subject_id": str, "contrast": str, "file_id": str})
    index_path = test_dir / SLICES_SUBDIR / SLICE_INDEX_CSV
    if index_path.is_file():
        index = pd.read_csv(index_path, dtype=str)
        index["batch_index"] = pd.to_numeric(index["batch_index"], errors="coerce")
        first = index.groupby("batch_index", as_index=False).agg(
            image=("image", "first"),
            realised_acceleration=(
                "realised_acceleration",
                lambda s: pd.to_numeric(s, errors="coerce").mean(),
            ),
        )
        rows = rows.merge(first, on="batch_index", how="left")
    return rows


def long_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (slice, metric) with ``metric``, ``acceleration`` (rung) and ``value``."""
    metric_cols = [c for c in table.columns if c.startswith("val_")]
    keep = [
        c
        for c in ("batch_index", "subject_id", "contrast", "slice_index", "realised_acceleration")
        if c in table.columns
    ]
    long = table.melt(id_vars=keep, value_vars=metric_cols, var_name="column", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    parsed = long["column"].str.removeprefix("val_").str.extract(_RUNG)
    long["metric"] = parsed["base"].fillna(long["column"].str.removeprefix("val_"))
    long["acceleration"] = pd.to_numeric(parsed["rate"], errors="coerce")
    # Whole R: a radial mask realises 15.55 on one slice and 15.62 on the next.
    if "realised_acceleration" in long.columns:
        long["realised_acceleration"] = pd.to_numeric(
            long["realised_acceleration"], errors="coerce"
        ).round(0)
    return long.drop(columns="column")


def situation_columns(long: pd.DataFrame) -> list[str]:
    """The acquisition conditions that actually vary across the test set."""
    out = []
    if long["acceleration"].notna().any():
        out.append("acceleration")
    if (
        "realised_acceleration" in long.columns
        and long["realised_acceleration"].nunique(dropna=True) > 1
    ):
        out.append("realised_acceleration")
    return out


def _stats(values: pd.Series) -> dict[str, float]:
    v = values.dropna().to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, **{k: float("nan") for k in _STATS[1:]}}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Stats per metric for every grouping: overall, contrast, each situation, contrast x situation."""
    situations = situation_columns(long)
    groupings: list[list[str]] = [[]]
    if "contrast" in long.columns and long["contrast"].nunique(dropna=True) > 1:
        groupings.append(["contrast"])
    groupings += [[s] for s in situations]
    if len(groupings) > 1 and "contrast" in long.columns:
        groupings += [["contrast", s] for s in situations]
    records = []
    for keys in groupings:
        grouped = long.groupby(["metric", *keys], dropna=False)
        for name, part in grouped:
            name = name if isinstance(name, tuple) else (name,)
            label = dict(zip(["metric", *keys], name, strict=True))
            records.append(
                {
                    "grouping": "+".join(keys) or "overall",
                    "metric": label.pop("metric"),
                    "group": ", ".join(f"{k}={v}" for k, v in label.items()) or "all",
                    **_stats(part["value"]),
                }
            )
    return pd.DataFrame.from_records(records)


def write_summary(test_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write ``per_slice_metrics.csv`` and ``summary_by_group.csv``; return (table, long, summary)."""
    test_dir = Path(test_dir)
    table = per_slice_table(test_dir)
    if table.empty:
        return table, pd.DataFrame(), pd.DataFrame()
    table.to_csv(test_dir / PER_SLICE_CSV, index=False)
    long = long_metrics(table)
    summary = summarise(long)
    summary.to_csv(test_dir / SUMMARY_CSV, index=False)
    return table, long, summary


__all__ = [
    "PER_SLICE_CSV",
    "SUMMARY_CSV",
    "long_metrics",
    "per_slice_table",
    "situation_columns",
    "summarise",
    "write_summary",
]
