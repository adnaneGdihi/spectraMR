r"""Per-arm metric distributions: violin, box and every case, one panel per metric.

Reads the per-slice rows of the selected validation pass (``slice_df``, keyed
``subject|contrast|slice|repetition``) when the run recorded them, else the test-split
per-case rows (``predictions_df``). The figure says which.

With slice identity and two or more arms, a bracket joins the two best arms (by median,
in the metric's own direction) with a Wilcoxon signed-rank test over subjects: each
subject's matched slices are averaged first, as the contrast statistics do
(:func:`..per_subject_stats.compare_subjects`). Without identity no bracket is drawn: rows
that are not keyed by slice cannot be paired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.core.metrics.metric_directions import resolve_direction
from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import (
    column_width,
    panel_label,
    panel_letter,
    pretty_label,
    use_default_style,
)


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _from_slices(slice_df) -> tuple[pd.DataFrame, bool] | None:
    tab = _frame.wide(slice_df)
    if tab is None:
        return None
    ids = pd.Series(tab.index.get_level_values("case_id").astype(str))
    return tab, bool((ids.str.count(r"\|") >= 2).all())


def _from_cases(frame) -> pd.DataFrame | None:
    """Test-split per-case rows as ``(method, case_id)`` x metric, or ``None``."""
    if frame is None or frame.empty or not {"metric", "value"} <= set(frame.columns):
        return None
    rows = frame[frame["split"] == "test"] if "split" in frame else frame
    if rows.empty:
        return None
    rows = rows.assign(
        method=rows.get("method", "run"),
        case_id=rows["case_id"] if "case_id" in rows else rows.groupby(["metric"]).cumcount(),
    )
    return rows.pivot_table(
        index=["method", "case_id"], columns="metric", values="value", aggfunc="mean"
    )


def _subject_test(tab: pd.DataFrame, metric: str, a: str, b: str) -> tuple[float, int] | None:
    """Wilcoxon p over subjects for arms ``a`` and ``b``, and the subject count."""
    from spectramr.infrastructure.reporting.per_subject_stats import (
        PairedSubjects,
        compare_subjects,
    )

    joined = pd.concat(
        {"a": tab[metric].xs(a, level="method"), "b": tab[metric].xs(b, level="method")},
        axis=1,
        join="inner",
    ).dropna()
    if joined.empty:
        return None
    joined.index = [str(c).split("|")[0] for c in joined.index]
    table = joined.groupby(level=0).mean()
    if len(table) < 2:
        return None
    paired = PairedSubjects(
        table=table,
        n_slices=len(joined),
        nonfinite_dropped={"a": 0, "b": 0},
        unpaired_slices={"a": 0, "b": 0},
        scale="metric",
    )
    p = compare_subjects(paired, n_bootstrap=200)["wilcoxon"].get("p_value", np.nan)
    return float(p), len(table)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    metrics=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    keyed = _from_slices(slice_df)
    if keyed is not None:
        tab, identity = keyed
        source = "validation, selected pass, per slice"
    else:
        tab, identity = _from_cases(predictions_df if predictions_df is not None else df), False
        source = "test split, per case"
    if tab is None or tab.empty:
        return None
    chosen = _frame.pick_metrics(tab, list(metrics) if metrics else None, k=3)
    if not chosen:
        return None
    arms = _frame.methods_of(tab)
    colours = arm_palette(arms)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(1, len(chosen), figsize=(width, width * 0.36), squeeze=False)
    for j, metric in enumerate(chosen):
        ax = axes[0][j]
        data = [tab[metric].xs(a, level="method").dropna().to_numpy() for a in arms]
        keep = [i for i, d in enumerate(data) if d.size]
        parts = ax.violinplot(
            [data[i] for i in keep], positions=[i + 1 for i in keep], showextrema=False
        )
        for i, body in zip(keep, parts["bodies"], strict=True):
            body.set_facecolor(colours[arms[i]])
            body.set_alpha(0.3)
        ax.boxplot(
            [data[i] for i in keep], positions=[i + 1 for i in keep], widths=0.15, showfliers=False
        )
        for i in keep:
            jitter = np.random.default_rng(i).uniform(-0.06, 0.06, data[i].size)
            ax.scatter(
                i + 1 + jitter,
                data[i],
                s=2,
                color=colours[arms[i]],
                alpha=0.4,
                linewidths=0,
                rasterized=True,
            )
        ax.set_xticks(
            range(1, len(arms) + 1),
            arms if len(arms) <= 4 else [""] * len(arms),
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(pretty_label(metric))
        sparse_ticks(ax, axes="y")
        if len(chosen) > 1:
            panel_label(ax, panel_letter(j))
        hib = resolve_direction(metric)
        if identity and len(keep) >= 2 and hib is not None:
            med = {i: float(np.median(data[i])) for i in keep}
            a, b = sorted(keep, key=lambda i: med[i], reverse=hib)[:2]
            res = _subject_test(tab, metric, arms[a], arms[b])
            if res is not None:
                p, n = res
                top = max(float(np.max(data[a])), float(np.max(data[b])))
                span = top - min(float(np.min(d)) for d in (data[a], data[b]))
                y = top + 0.04 * (span or 1.0)
                ax.plot(
                    [a + 1, a + 1, b + 1, b + 1],
                    [y, y + 0.02 * span, y + 0.02 * span, y],
                    lw=0.6,
                    color="#333333",
                )
                ax.text(
                    (a + b) / 2 + 1,
                    y + 0.03 * span,
                    f"{_stars(p)} (n={n} subjects)",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    fig.suptitle(
        source + ("; bracket: Wilcoxon over subjects, best two arms" if identity else ""),
        fontsize="x-small",
    )
    if len(arms) > 4:
        _frame.swatch_legend(fig, arms, colours)
    return _frame.finish(fig, out_path, formats, dpi, metadata, legend_rows=len(arms) > 4)
