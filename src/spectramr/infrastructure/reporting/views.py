r"""Which validation pass a report reads each run at.

A run's validation log has one row per validation *pass*, so every number a
table or a cross-run figure quotes is one of those passes, picked by a rule:

``selected``
    The pass the run's own early stopping chose (``early_stopping_best_iteration``
    in ``final_metrics.json``) -- what its best checkpoint holds. Every metric is
    read at that one pass. ``final_metrics.json``'s ``best`` block is *not* used:
    it takes each metric's own optimum, which for different metrics lies at
    different checkpoints and so describes no model that exists.
``final``
    Each run's last validation pass.
``common``
    The last iteration every run reached -- one training budget for all arms.

A single run has no spread across seeds; the view returns one value per metric
and leaves the std to the caller, which must report it as absent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: fallbacks already reported, so one run's report logs each once rather than per figure
_WARNED: set[tuple[str, str]] = set()

VIEWS: tuple[str, ...] = ("selected", "final", "common")

#: ``final_metrics.json`` key carrying the early-stopping pass (``split="final"``).
SELECTED_KEY = "early_stopping_best_iteration"


def _validate(view: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"unknown checkpoint view {view!r}; choose from {list(VIEWS)}")


def _with_run_id(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-built frames may omit ``run_id``; one per ``method``, else one run."""
    if df is None or df.empty or "run_id" in df.columns:
        return df
    df = df.copy()
    df["run_id"] = df["method"] if "method" in df.columns else "run"
    return df


def _val(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not {"split", "step"} <= set(df.columns):
        return pd.DataFrame(columns=getattr(df, "columns", []))
    out = df[(df["split"] == "val") & df["value"].notna()]
    return out[out["step"] >= 0]


def _selected_step(run: pd.DataFrame, val_steps: np.ndarray) -> int | None:
    rows = run[(run["split"] == "final") & (run["metric"] == SELECTED_KEY)]["value"].dropna()
    if rows.empty:
        return None
    step = int(rows.iloc[-1])
    if step in val_steps:
        return step
    # The early-stopping monitor can fire on an iteration between logged passes;
    # the checkpoint then holds the nearest pass at or before it.
    earlier = val_steps[val_steps <= step]
    return int(earlier.max()) if earlier.size else None


def checkpoint_steps(df: pd.DataFrame, view: str = "selected") -> dict[str, int]:
    """``{run_id: step}`` for each run in ``df`` under ``view``.

    A run without validation rows is omitted. Under ``selected`` a run that
    recorded no early-stopping iteration falls back to its final pass, and the
    fallback is logged -- it is a different checkpoint and must not be silent.
    """
    _validate(view)
    df = _with_run_id(df)
    val = _val(df)
    if val.empty:
        return {}
    last = {rid: int(g["step"].max()) for rid, g in val.groupby("run_id")}
    if view == "final":
        return last
    if view == "common":
        budget = min(last.values())
        out: dict[str, int] = {}
        for rid, g in val.groupby("run_id"):
            steps = np.sort(g["step"].unique())
            reached = steps[steps <= budget]
            if reached.size:
                out[rid] = int(reached.max())
            else:
                logger.warning(
                    "run %s has no validation pass at or before the common budget %d "
                    "(first pass %d); it is left out of the common view",
                    rid,
                    budget,
                    int(steps.min()),
                )
        return out
    out = {}
    for rid, g in val.groupby("run_id"):
        run = df[df["run_id"] == rid]
        step = _selected_step(run, np.sort(g["step"].unique()))
        if step is None:
            recorded = run[(run["split"] == "final") & (run["metric"] == SELECTED_KEY)]
            why = (
                f"its {SELECTED_KEY} ({int(recorded['value'].iloc[-1])}) precedes every pass"
                if not recorded.empty
                else f"it records no {SELECTED_KEY}"
            )
            # A one-pass run with no selection (validate-only grading) falls back to that same pass.
            silent = recorded.empty and g["step"].nunique() == 1
            if not silent and (str(rid), why) not in _WARNED:
                _WARNED.add((str(rid), why))
                logger.warning("run %s: %s; reading its final validation pass instead", rid, why)
            step = last[rid]
        out[rid] = step
    return out


def recorded_steps(df: pd.DataFrame) -> dict[str, int]:
    """``{run_id: step}`` only for runs whose early stopping recorded a selection.

    The ``selected`` view falls back to the final pass for a run that recorded none;
    a figure that marks "the selected checkpoint" must not mark that fallback.
    """
    df = _with_run_id(df)
    val = _val(df)
    if val.empty:
        return {}
    out = {}
    for rid, g in val.groupby("run_id"):
        step = _selected_step(df[df["run_id"] == rid], np.sort(g["step"].unique()))
        if step is not None:
            out[rid] = step
    return out


def snapshot(df: pd.DataFrame, view: str = "selected") -> pd.DataFrame:
    """Validation rows of every run at its ``view`` step, one row per metric.

    Carries ``step`` so a caption can state which pass each number came from.
    """
    df = _with_run_id(df)
    steps = checkpoint_steps(df, view)
    val = _val(df)
    if not steps or val.empty:
        return val.iloc[0:0]
    keep = val["run_id"].map(steps) == val["step"]
    out = val[keep]
    # A pass logged twice (resume replay) keeps its last row.
    return out.drop_duplicates(subset=["run_id", "metric"], keep="last").reset_index(drop=True)


__all__ = ["SELECTED_KEY", "VIEWS", "checkpoint_steps", "recorded_steps", "snapshot"]
