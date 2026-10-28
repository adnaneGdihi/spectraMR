r"""Metric vs acceleration factor R, one line per method.

Two sources, best first:

1. ``per_call_df`` (``per_call_metrics.csv``) cut to the checkpoint pass: the
   mean over validation calls at each R, with a bootstrap 95 % CI band and the
   call count printed on the axis.
2. The aggregator frame's per-R columns (``val_psnr_8x`` -> ``acceleration=8``)
   read at the checkpoint pass: one point per R and *no band* -- one number per
   R has no spread, and a band drawn over validation passes would be the
   training trajectory, not uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metric_names import annotate_frame
from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.per_case import per_case_rows
from spectramr.infrastructure.reporting.plotters._stats import bootstrap_ci
from spectramr.infrastructure.reporting.style import (
    colour_for,
    column_width,
    pretty_label,
    save_figure,
    use_default_style,
)
from spectramr.infrastructure.reporting.views import checkpoint_steps, snapshot

_PREFERRED = ("psnr", "ssim", "nmse", "hfen")


def _from_per_call(df, per_call_df, view, checkpoint_step=None) -> pd.DataFrame:
    if per_call_df is None or per_call_df.empty or "acceleration_level" not in per_call_df:
        return pd.DataFrame()
    step = checkpoint_step
    if step is None and df is not None and not df.empty:
        # without the caller's run, only a single-run frame names one checkpoint
        steps = checkpoint_steps(df, view)
        step = next(iter(steps.values())) if len(steps) == 1 else None
    method = str(df["method"].iloc[0]) if df is not None and "method" in df else "run"
    rows = per_case_rows(per_call_df, step=step, method=method)
    return rows[rows["acceleration"].notna()]


def _from_frame(df, view) -> pd.DataFrame:
    """Logged per-R columns at the ``view`` pass (one value per R)."""
    if df is None or df.empty or "acceleration" not in df.columns or "step" not in df.columns:
        return pd.DataFrame()
    rows = snapshot(df, view)
    return rows[rows["acceleration"].notna()] if not rows.empty else rows


def _from_test(df) -> pd.DataFrame:
    """Per-case evaluation rows that carry an explicit ``acceleration``."""
    if df is None or df.empty or "acceleration" not in df.columns:
        return pd.DataFrame()
    if "step" in df.columns and "split" in df.columns and not (df["split"] == "test").any():
        return pd.DataFrame()
    rows = df[df["split"] == "test"] if "split" in df.columns else df
    rows = rows[rows["acceleration"].notna()]
    if rows.empty:
        return rows
    if "base_metric" not in rows.columns:
        rows = annotate_frame(rows.drop(columns="acceleration")).assign(
            acceleration=rows["acceleration"].to_numpy()
        )
    return rows


def _pick_metric(rows: pd.DataFrame, metric: str | None) -> str | None:
    available = set(rows["base_metric"])
    if metric is not None:
        return metric if metric in available else None
    return next((m for m in _PREFERRED if m in available), None)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metric: str | None = None,
    per_call_df: pd.DataFrame | None = None,
    per_case_df: pd.DataFrame | None = None,
    view: str = "selected",
    call_unit: str = "cases",
    checkpoint_step: int | None = None,
    expected_methods: list[str] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    # ``per_case_df`` is already one pass per arm (the cross-run report melts it).
    rows = (
        per_case_df[per_case_df["acceleration"].notna()]
        if per_case_df is not None and not per_case_df.empty
        else _from_per_call(df, per_call_df, view, checkpoint_step)
    )
    if rows.empty:
        rows = _from_test(df)
    per_case = not rows.empty
    if not per_case:
        rows = _from_frame(df, view)
    if rows.empty:
        return None
    metric = _pick_metric(rows, metric)
    if metric is None:
        return None
    sub = rows[rows["base_metric"] == metric]
    accels = sorted(sub["acceleration"].unique())
    if len(accels) < 2:
        return None
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single") * 0.75))
    counts: list[int] = []
    order = [str(m) for m in pd.unique(df["method"])] if "method" in df.columns else []
    order += [m for m in map(str, sub["method"].unique()) if m not in order]
    for method in [m for m in order if m in set(sub["method"].astype(str))]:
        i = order.index(method)
        md = sub[sub["method"] == method]
        colour = colour_for(str(method), i)
        means, los, his = [], [], []
        for r in accels:
            vals = md.loc[md["acceleration"] == r, "value"].to_numpy(dtype=float)
            means.append(np.nanmean(vals) if vals.size else np.nan)
            lo, hi = bootstrap_ci(vals) if per_case and vals.size > 1 else (np.nan, np.nan)
            los.append(lo)
            his.append(hi)
            counts.append(int(np.isfinite(vals).sum()))  # every arm, every R
        ax.plot(accels, means, marker="o", color=colour, label=str(method))
        if per_case:
            ax.fill_between(accels, los, his, color=colour, alpha=0.2, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(accels)
    ax.set_xticklabels([f"{r:g}\u00d7" for r in accels])
    ax.set_xlabel("acceleration factor R")
    ax.set_ylabel(pretty_label(metric))
    note = (
        f"mean ± 95% bootstrap CI over {call_unit} (n = "
        + (f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}")
        + " per R)"
        if per_case
        else f"one value per R at the {view} checkpoint; no spread available"
    )
    absent = [m for m in (expected_methods or []) if m not in set(sub["method"].astype(str))]
    if absent:
        note += "; no per-case rows: " + ", ".join(absent)
    ax.set_title(note, fontsize="x-small", loc="left")
    if sub["method"].nunique() > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    plt.close(fig)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
