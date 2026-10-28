r"""Per-case metric rows at one checkpoint -- the only honest source of error bars.

``per_call_metrics.csv`` holds one row per validation *call* (one batch at one
acceleration) for every validation pass of the run. A spread taken over all its
rows mixes early and late checkpoints, so it measures training progress, not
variation across cases. This module cuts the file to one pass and melts it to
``[run_id, method, case_id, pair_key, acceleration, metric, base_metric, value]``.

A row is a batch mean when the validation batch held more than one slice, so
the unit of ``n`` is the *validation call*; figures label it that way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metric_names import parse_metric_name

PER_CALL_CSV = "per_call_metrics.csv"

#: Bookkeeping columns of ``per_call_metrics.csv`` that are not metrics.
ID_COLUMNS: frozenset[str] = frozenset(
    {
        "case_id",
        "split",
        "step",
        "epoch",
        "iteration",
        "acceleration_level",
        "acceleration_realized",
        "timestep",
        "heldout",
        "batch_index",
        "subject_id",
    }
)


def validation_batch_size(run_dir: str | Path) -> int | None:
    """``validation.loader.batch_size`` from the run's ``resolved_config.json``.

    The training batch in ``run_summary.json`` is a different number (it carries
    accumulation and the training loader), so it cannot stand in for this one.
    """
    import json

    path = Path(run_dir) / "resolved_config.json"
    try:
        size = json.loads(path.read_text())["validation"]["loader"]["batch_size"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return int(size) if isinstance(size, int) and size > 0 else None


def call_unit(batch_sizes: list[int | None]) -> str:
    """What one per-call row is, for an ``n = ...`` label.

    A row is the mean over one validation batch, so ``n`` counts calls, not
    slices, unless every batch held a single slice.
    """
    known = {b for b in batch_sizes if b is not None}
    if known == {1}:
        return "slices"
    if len(known) == 1 and None not in batch_sizes:
        return f"validation calls (means of {known.pop()} slices)"
    return "validation calls (batch means)"


def read_per_call(run_dir: str | Path) -> pd.DataFrame | None:
    """The run's per-call CSV, or ``None`` when it was not recorded."""
    path = Path(run_dir) / PER_CALL_CSV
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


def nearest_step(frame: pd.DataFrame, step: int | None) -> int | None:
    """The logged pass at or before ``step``; the last pass when ``step`` is None."""
    if frame is None or "step" not in frame.columns or frame.empty:
        return None
    steps = pd.to_numeric(frame["step"], errors="coerce").dropna().astype(int)
    if steps.empty:
        return None
    if step is None:
        return int(steps.max())
    earlier = steps[steps <= step]
    return int(earlier.max()) if not earlier.empty else None


def per_case_rows(
    frame: pd.DataFrame | None,
    *,
    step: int | None = None,
    run_id: str = "run",
    method: str = "run",
    split: str = "val",
) -> pd.DataFrame:
    """Melt the per-call rows of one pass into a long per-case frame.

    ``step`` is the checkpoint the caller reads the run at (see
    :mod:`.views`); ``None`` takes the last logged pass.
    """
    empty = pd.DataFrame(
        columns=["run_id", "method", "case_id", "acceleration", "metric", "base_metric", "value"]
    )
    if frame is None or frame.empty:
        return empty
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == split]
    chosen = nearest_step(frame, step)
    if chosen is None and "step" in frame.columns:
        # No pass at or before the checkpoint: nothing to report, never every pass.
        return empty
    if chosen is not None:
        frame = frame[pd.to_numeric(frame["step"], errors="coerce") == chosen]
    if frame.empty:
        return empty
    metrics = [
        c
        for c in frame.columns
        if c not in ID_COLUMNS
        and not pd.api.types.is_bool_dtype(frame[c])
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    ids = frame.reset_index(drop=True)
    case_id = ids["case_id"].astype(str) if "case_id" in ids else pd.Series("call", index=ids.index)
    batch = (
        pd.to_numeric(ids["batch_index"], errors="coerce")
        if "batch_index" in ids
        else pd.Series(float("nan"), index=ids.index)
    )
    has_batch = batch.notna()
    # ``case_id`` names the pass and R (``val_step17500_R2x``) and repeats across
    # batches, so the batch index is what makes a row one case. A row that came
    # without one (strategies feeding the sink themselves) is its own case.
    tag = [
        str(int(b)) if ok else f"row{i}"
        for i, (b, ok) in enumerate(zip(batch, has_batch, strict=True))
    ]
    accel = (
        pd.to_numeric(ids["acceleration_level"], errors="coerce")
        if "acceleration_level" in ids
        else pd.Series(float("nan"), index=ids.index)
    )
    long = (
        ids[metrics]
        .assign(
            _case=case_id + "#" + pd.Series(tag, index=ids.index),
            _accel=accel,
            _tag=tag,
            _has_batch=has_batch,
        )
        .melt(
            id_vars=["_case", "_accel", "_tag", "_has_batch"], var_name="metric", value_name="value"
        )
        .dropna(subset=["value"])
    )
    parsed = long["metric"].map(parse_metric_name)
    # A cascade aggregate (``_mean``, ``_accel_gap``) summarises cases; it is not one.
    long = long[parsed.map(lambda n: n.aggregate is None)]
    parsed = parsed[long.index]
    # The rung comes from the row when the sink recorded it, else from the name
    # (``val_psnr_8x``): one call can log every rung, and pooling them would mix R.
    long["acceleration"] = long["_accel"].where(
        long["_accel"].notna(), parsed.map(lambda n: n.acceleration).astype(float)
    )
    long["base_metric"] = parsed.map(lambda n: n.base)
    long["case_id"] = long["_case"]
    # ``pair_key`` names the same validation case in another arm: the batch at this
    # R, whatever pass each arm is read at (a seeded, unshuffled loader repeats
    # its batches). Rows without a batch index cannot be paired across arms.
    r_tag = long["acceleration"].map(lambda r: "all" if pd.isna(r) else f"R{r:g}")
    long["pair_key"] = (r_tag + "#" + long["_tag"]).where(long["_has_batch"], long["_case"])
    long["run_id"] = run_id
    long["method"] = method
    long["step"] = chosen
    # A pass replayed after a resume repeats its batches; keep the last. Only rows
    # with a batch index can be recognised as a replay.
    batched = long[long["_has_batch"]].drop_duplicates(subset=["pair_key", "metric"], keep="last")
    long = pd.concat([long[~long["_has_batch"]], batched]).sort_index()
    long = long.drop(columns=["_case", "_accel", "_tag", "_has_batch"])
    return long.reset_index(drop=True)


__all__ = [
    "ID_COLUMNS",
    "PER_CALL_CSV",
    "call_unit",
    "nearest_step",
    "per_case_rows",
    "read_per_call",
    "validation_batch_size",
]
