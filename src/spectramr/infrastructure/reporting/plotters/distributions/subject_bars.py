"""Figure 3.2 — per-subject means, sorted.

One bar per subject (mean over that subject's slices), ordered by value so the spread
and the tail read at a glance; colour is the arm. No tick labels or value text: a
subject is identified in the sidecar table, not on the canvas.
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
    cols = _frame.pick_metrics(table, metrics, k=2)
    if not cols:
        return None
    methods = _frame.methods_of(table)
    colours = arm_palette(methods)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(1, len(cols), figsize=(width, width * 0.28), squeeze=False)
    drawn = False
    for i, metric in enumerate(cols):
        ax = axes[0][i]
        for method in methods:
            sub = table.xs(method, level="method")[metric].dropna()
            if sub.empty:
                continue
            subjects = sub.index.map(_frame.subject_of)
            means = sub.groupby(subjects).mean().sort_values()
            ax.plot(np.arange(means.size), means.to_numpy(), color=colours[method], lw=1.0)
            drawn = True
        ax.set_xticks([])
        ax.set_xlabel("subjects, sorted")
        ax.set_ylabel(pretty_label(metric))
        sparse_ticks(ax, axes="y")
        if len(cols) > 1:
            panel_label(ax, chr(ord("a") + i))
    if not drawn:
        plt.close(fig)
        return None
    many = len(methods) > 1
    if many:
        _frame.swatch_legend(fig, methods, colours)
    return _frame.finish(fig, out_path, formats, dpi, metadata, legend_rows=many)
