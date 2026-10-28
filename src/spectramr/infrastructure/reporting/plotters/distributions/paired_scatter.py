"""Figure 3.4 — arm against baseline, slice by slice.

For each non-baseline arm, its per-slice value against the baseline's on the same slice
(``case_id``), with the identity line: points above it are slices the arm improved for a
higher-is-better metric. One panel per arm, one metric (PSNR by default).
"""

from __future__ import annotations

from pathlib import Path

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

_MAX_PANELS = 6


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    baseline: str | None = None,
    metric: str | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    table = _frame.wide(slice_df)
    if table is None:
        return None
    methods = _frame.methods_of(table)
    if len(methods) < 2:
        return None
    base = baseline if baseline in methods else methods[0]
    chosen = _frame.pick_metrics(table, [metric] if metric else None, k=1)
    if not chosen:
        return None
    m = chosen[0]
    colours = arm_palette(methods)
    others = [x for x in methods if x != base][:_MAX_PANELS]
    ref = table.xs(base, level="method")[m].dropna()
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    cols = min(len(others), 3)
    rows = -(-len(others) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width, width * 0.3 * rows), squeeze=False)
    drawn = False
    for i, arm in enumerate(others):
        ax = axes[i // cols][i % cols]
        cur = table.xs(arm, level="method")[m].dropna()
        shared = ref.index.intersection(cur.index)
        if len(shared) < 3:
            ax.set_axis_off()
            continue
        x, y = ref[shared].to_numpy(), cur[shared].to_numpy()
        ax.scatter(x, y, s=3, color=colours[arm], alpha=0.3, linewidths=0, rasterized=True)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color="#666666", lw=0.6, ls=":")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(base)
        ax.set_ylabel(arm)
        sparse_ticks(ax)
        if len(others) > 1:
            panel_label(ax, chr(ord("a") + i))
        drawn = True
    for k in range(len(others), rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    if not drawn:
        plt.close(fig)
        return None
    fig.suptitle(pretty_label(m), fontsize="small")
    return _frame.finish(fig, out_path, formats, dpi, metadata)
