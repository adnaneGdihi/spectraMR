"""Which values the QC group strip distributes: one pass, per case, never a trajectory."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.per_case import ID_COLUMNS, nearest_step

logger = logging.getLogger(__name__)

# Columns that are identifiers/bookkeeping, never a metric to distribute.
_NON_METRIC = ID_COLUMNS | {"method", "run_id", "metric", "value"}


def _series_from_wide(frame: pd.DataFrame) -> dict:
    """``per_call_metrics.csv`` shape: numeric non-id columns are metrics."""
    out: dict[str, tuple] = {}
    labels = frame["case_id"].astype(str).to_numpy() if "case_id" in frame else None
    accel = (
        pd.to_numeric(frame["acceleration_level"], errors="coerce").to_numpy()
        if "acceleration_level" in frame
        else None
    )
    if accel is not None and np.unique(accel[np.isfinite(accel)]).size < 2:
        accel = None
    for col in frame.columns:
        if col in _NON_METRIC or pd.api.types.is_bool_dtype(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy()
        mask = np.isfinite(vals)
        if mask.sum() >= 1:
            out[str(col)] = (
                vals[mask],
                labels[mask] if labels is not None else None,
                accel[mask] if accel is not None else None,
            )
    return out


def _series_from_long(frame: pd.DataFrame) -> dict:
    """Long frame (``metric``, ``value``[, ``subject_id``]) → per-metric arrays."""
    out: dict[str, tuple] = {}
    sub = frame
    if "split" in sub.columns:
        keep = sub["split"].isin(["test", "val"])
        sub = sub[keep] if keep.any() else sub
    for metric, grp in sub.groupby("metric"):
        vals = pd.to_numeric(grp["value"], errors="coerce").to_numpy()
        labs = grp["subject_id"].astype(str).to_numpy() if "subject_id" in grp else None
        mask = np.isfinite(vals)
        if mask.sum() >= 1:
            out[str(metric)] = (vals[mask], labs[mask] if labs is not None else None, None)
    return out


def _one_pass(df, per_call_df: pd.DataFrame, checkpoint_step=None, view="selected"):
    """Rows of the checkpoint pass only; a strip across passes is a trajectory."""
    if "step" not in per_call_df.columns:
        return per_call_df
    from spectramr.infrastructure.reporting.views import checkpoint_steps

    step = checkpoint_step
    if step is None and df is not None and not df.empty:
        steps = checkpoint_steps(df, view)
        step = next(iter(steps.values())) if len(steps) == 1 else None
    chosen = nearest_step(per_call_df, step)
    if chosen is None:
        logger.info("qc_group_strip: no per-call pass at or before step %s", step)
        return per_call_df.iloc[0:0]
    return per_call_df[pd.to_numeric(per_call_df["step"], errors="coerce") == chosen]


def _collect(df, per_call_df, predictions_df, checkpoint_step=None, view="selected") -> dict:
    if per_call_df is not None and not per_call_df.empty:
        s = _series_from_wide(_one_pass(df, per_call_df, checkpoint_step, view))
        if s:
            return s
    if (
        predictions_df is not None
        and not predictions_df.empty
        and "metric" in predictions_df.columns
    ):
        s = _series_from_long(predictions_df)
        if s:
            return s
    # The aggregator frame holds one row per validation PASS, not per case, so a
    # strip drawn from it is a training trajectory presented as a cohort spread.
    # Only the test split, which ``final_eval.json`` may key per subject, qualifies.
    if df is not None and not df.empty and {"metric", "split"} <= set(df.columns):
        test = df[df["split"] == "test"]
        if not test.empty:
            return _series_from_long(test)
    return {}
