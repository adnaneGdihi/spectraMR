"""Figure 3.5 — arm rank per metric.

Arms by metrics, colour is the arm's rank on that metric (1 = best, direction-aware, by
mean over shared slices). No numbers on the cells: the colour bar runs best to worst.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.core.metrics.metric_directions import resolve_direction
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import (
    column_width,
    pretty_label,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
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
    cols = [m for m in _frame.PREFERRED if m in table.columns]
    means = table.groupby(level="method")[cols].mean().reindex(methods)
    ranks = pd.DataFrame(index=methods)
    for col in cols:
        higher = resolve_direction(col)
        if higher is None or means[col].isna().any():
            continue
        ranks[col] = means[col].rank(ascending=not higher, method="min")
    if ranks.empty:
        return None
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, width * max(0.5, 0.07 * len(methods) + 0.3)))
    im = ax.imshow(ranks.to_numpy(), cmap="viridis_r", vmin=1, vmax=len(methods), aspect="auto")
    ax.set_xticks(range(ranks.shape[1]))
    ax.set_xticklabels([pretty_label(c) for c in ranks.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[1, len(methods)])
    cbar.ax.set_yticklabels(["best", "worst"])
    return _frame.finish(fig, out_path, formats, dpi, metadata)
