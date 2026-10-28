"""Figure 3.3 — metric against metric, per slice.

A hexbin of two per-slice metrics (PSNR against SSIM by default) so the joint
distribution, not just each marginal, is visible; one small panel per arm.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.legend import sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import (
    column_width,
    pretty_label,
    use_default_style,
)

_MAX_PANELS = 4


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
    pair = _frame.pick_metrics(table, metrics, k=2)
    if len(pair) < 2:
        return None
    x, y = pair
    methods = _frame.methods_of(table)[:_MAX_PANELS]
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(
        1, len(methods), figsize=(width, width * 0.3), squeeze=False, sharex=True, sharey=True
    )
    artist = None
    for i, method in enumerate(methods):
        ax = axes[0][i]
        sub = table.xs(method, level="method")[[x, y]].dropna()
        if len(sub) < 3:
            ax.set_axis_off()
            continue
        artist = ax.hexbin(sub[x], sub[y], gridsize=28, mincnt=1, cmap="viridis", linewidths=0)
        ax.set_title(method, fontsize="x-small")
        ax.set_xlabel(pretty_label(x))
        if i == 0:
            ax.set_ylabel(pretty_label(y))
        sparse_ticks(ax)
    if artist is None:
        plt.close(fig)
        return None
    cbar = fig.colorbar(artist, ax=axes[0].tolist(), shrink=0.8, pad=0.02, aspect=18)
    cbar.set_label("slices")
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    primary = _frame.save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
    plt.close(fig)
    return primary
