r"""Figures a comparison adds when it reports every metric, not the headline panels.

* ``acceleration_sweep_<metric>``: one sweep per metric with two or more accelerations.
* ``fig_1_2_learning_curves_<k>``: curves beyond the first ``max_panels`` panels; the
  explicit-list learning-curve figure letters its panels ``a``, ``b``, ... and runs
  past ``z`` into control characters, so it is never asked for more than a chunk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .. import plotters
from .arms import Arms

logger = logging.getLogger(__name__)


def sweeps(
    arms: Arms, metrics: list[str], fig_dir: Path, formats, dpi, view, metadata, figures, reasons
):
    """One ``acceleration_sweep_<metric>`` per metric with at least two accelerations."""
    rows = arms.per_case
    if rows is None or rows.empty:
        return
    fn = plotters.get("acceleration_sweep")
    for metric in metrics:
        sub = (
            rows[rows["base_metric"] == metric] if "base_metric" in rows.columns else rows.iloc[:0]
        )
        if sub["acceleration"].dropna().nunique() < 2:
            continue
        fid = f"acceleration_sweep_{metric}"
        try:
            figures[fid] = fn(
                arms.df,
                fig_dir / f"{fid}.{formats[0]}",
                metric=metric,
                per_case_df=rows,
                view=view,
                call_unit=arms.call_unit,
                expected_methods=arms.labels,
                formats=formats,
                dpi=dpi,
                metadata=metadata,
            )
        except Exception as exc:  # report-time soft-fail, as plotters.dispatch_detailed
            logger.warning("plotter '%s' failed: %s", fid, exc)
            figures[fid], reasons[fid] = None, f"raised:{type(exc).__name__}: {exc}"


def more_curves(arms: Arms, names, fig_dir: Path, formats, dpi, size, metadata, figures) -> None:
    """Learning curves beyond the first ``size`` panels, in ``fig_1_2_learning_curves_2``, ..."""
    fn = plotters.get("fig_1_2_learning_curves")
    for k, start in enumerate(range(0, len(names), size), start=2):
        fid = f"fig_1_2_learning_curves_{k}"
        figures[fid] = fn(
            arms.df,
            fig_dir / f"{fid}.{formats[0]}",
            metrics=list(names[start : start + size]),
            formats=formats,
            dpi=dpi,
            metadata=metadata,
        )


def table_metrics(arms: Arms) -> set[str]:
    """Base metrics the results table can read at the arms' view."""
    from ..tables._headline import _headline_rows

    return set(_headline_rows(arms.df, arms.view)["metric"])


def curve_names(arms: Arms, metrics: list[str]) -> list[str]:
    """Raw training and validation column names of ``metrics`` that were logged.

    Ordered as ``metrics`` is, training before validation within a metric, so the
    first chunk holds the leading metrics rather than whatever sorts first.
    """
    frame = arms.df
    keep = frame["split"].isin(("train", "val")) & frame["base_metric"].isin(metrics)
    keep &= frame["headline"] & frame["step"].ge(0)
    found = frame.loc[keep, ["metric", "base_metric"]].drop_duplicates()
    order = {m: i for i, m in enumerate(metrics)}
    ranked = sorted(found.itertuples(index=False), key=lambda r: (order[r.base_metric], r.metric))
    return [r.metric for r in ranked]


def cohort_figures(
    arms: Arms,
    table: pd.DataFrame,
    stats: Any,
    baseline: str | None,
    ladder: Any,
    phases: tuple[str, ...],
    fig_dir: Path,
    formats,
    dpi,
    max_panels: int,
    metadata,
    groups: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Coverage, rank reversal, phase and group bars and, given paired statistics, the effect figures."""
    from .contrast_data import inference_skips

    known = stats.unavailable if stats is not None else []
    jobs: dict[str, dict] = {
        "cmp_coverage": {
            "phase_df": table,
            "unavailable": known + inference_skips(arms, phases, known),
            "arms": arms.labels,
        },
        "cmp_rank_reversal": {"phase_df": table, "arms": arms.labels},
        "cmp_phase_bars": {"phase_df": table, "arms": arms.labels, "max_panels": max_panels},
        "cmp_group_bars": {
            "phase_df": table,
            "arms": arms.labels,
            "groups": groups,
            "baseline": baseline,
            "max_panels": max_panels,
        },
    }
    if stats is not None and baseline is not None:
        common = {"arms": arms.labels, "baseline": baseline, "max_panels": max_panels}
        jobs["cmp_effect_forest"] = {"paired_df": pd.DataFrame(stats.rows), **common}
        jobs["cmp_subject_lines"] = {"paired": stats.paired, **common}
        jobs["cmp_effect_bars"] = {
            "paired_df": pd.DataFrame(stats.rows),
            "paired": stats.paired,
            "groups": groups,
            **common,
        }
        jobs["cmp_metric_group_bars"] = {
            "paired_df": pd.DataFrame(stats.rows),
            "paired": stats.paired,
            **common,
        }
        if ladder is not None:
            jobs["cmp_metric_ladder"] = {"paired": stats.paired, "ladder": ladder, **common}
    figures: dict = {}
    reasons: dict = {}
    for fid, kwargs in jobs.items():
        out = plotters.dispatch_detailed(
            arms.df, [fid], fig_dir, formats=formats, dpi=dpi, metadata=metadata, **kwargs
        )[fid]
        figures[fid] = out.path
        if out.reason:
            reasons[fid] = out.reason
        for extra in sorted(fig_dir.glob(f"{fid}_*.{formats[0]}")):
            figures[extra.stem] = extra
    return figures, reasons


__all__ = ["cohort_figures", "curve_names", "more_curves", "sweeps", "table_metrics"]
