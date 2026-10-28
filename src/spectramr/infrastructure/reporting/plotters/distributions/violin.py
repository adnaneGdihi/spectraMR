"""Figure 3.1 — per-slice metric violins.

One panel per metric; one violin per arm (or per contrast for a single run), with the
median and quartiles inside it and no value text. More than four groups drop the tick
labels and use one swatch legend instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import (
    column_width,
    panel_label,
    pretty_label,
    use_default_style,
)

_MAX_TICKED = 3


def _groups(table: pd.DataFrame, metric: str, by_contrast: bool) -> dict[str, np.ndarray]:
    col = table[metric].dropna()
    if by_contrast:
        keys = col.index.get_level_values("case_id").map(_frame.contrast_of)
        return {str(k): col[keys == k].to_numpy() for k in pd.unique(keys)}
    meth = col.index.get_level_values("method")
    return {str(m): col[meth == m].to_numpy() for m in pd.unique(meth)}


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    table = _frame.wide(slice_df)
    if table is None:
        return None
    cols = _frame.pick_metrics(table, metrics)
    if not cols:
        return None
    methods = _frame.methods_of(table)
    by_contrast = len(methods) == 1
    first = _groups(table, cols[0], by_contrast)
    colours = arm_palette(list(first))
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(1, len(cols), figsize=(width, width * 0.3), squeeze=False)
    for i, metric in enumerate(cols):
        ax = axes[0][i]
        groups = {k: v for k, v in _groups(table, metric, by_contrast).items() if v.size > 1}
        if not groups:
            ax.set_axis_off()
            continue
        parts = ax.violinplot(list(groups.values()), showextrema=False, widths=0.85)
        for body, key in zip(parts["bodies"], groups, strict=True):
            body.set_facecolor(colours.get(key, "#888888"))
            body.set_edgecolor("none")
            body.set_alpha(0.8)
        pos = np.arange(1, len(groups) + 1)
        q1, med, q3 = (
            np.array([np.percentile(v, q) for v in groups.values()]) for q in (25, 50, 75)
        )
        ax.vlines(pos, q1, q3, color="black", lw=1.2, zorder=3)
        ax.scatter(pos, med, s=7, color="white", edgecolor="black", linewidths=0.4, zorder=4)
        ax.set_xticks(pos)
        ax.set_xticklabels(
            list(groups) if len(groups) <= _MAX_TICKED else [], rotation=0, ha="center"
        )
        ax.set_ylabel(pretty_label(metric))
        sparse_ticks(ax, axes="y")
        if len(cols) > 1:
            panel_label(ax, chr(ord("a") + i))
    many = len(first) > _MAX_TICKED
    if many:
        _frame.swatch_legend(fig, list(first), colours)
    return _frame.finish(fig, out_path, formats, dpi, metadata, legend_rows=many)
