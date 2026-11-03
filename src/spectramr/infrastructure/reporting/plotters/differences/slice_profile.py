"""Figure 6.4 — slice profile: the primary metric against slice position, per arm.

Line = median across subjects at each slice index, band = interquartile range (drawn for
up to four arms). A floor or
a slope with slice index (edge slices, coil coverage, a position-dependent bias) shows here
and averages away everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _slices


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    metric: str | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    tab = _slices.table(slice_df)
    if (
        tab is None
        or not _slices.has_identity(tab)
        or (pick := _slices.primary(tab, metric)) is None
    ):
        return None
    name, _ = pick
    rows = tab[name].dropna().reset_index()
    rows["slice"] = rows["case_id"].map(_slices.slice_of)
    rows = rows.dropna(subset=["slice"])
    if rows["slice"].nunique() < 3:
        return None
    stats = rows.groupby(["method", "slice"])[name].quantile([0.25, 0.5, 0.75]).unstack()
    arms = _frame.methods_of(tab)
    colours = arm_palette(arms)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, width * 0.75))
    for arm in arms:
        if arm not in stats.index.get_level_values("method"):
            continue
        s = stats.xs(arm, level="method")
        if len(arms) <= 4:  # beyond four arms overlapping bands hide the medians
            ax.fill_between(s.index, s[0.25], s[0.75], color=colours[arm], alpha=0.15, lw=0)
        ax.plot(s.index, s[0.5], color=colours[arm], lw=1.2, label=arm)
    ax.set_xlabel("slice index")
    ax.set_ylabel(name)
    ax.set_title(f"Median and IQR across subjects ({_slices.PHASE_NOTE})", fontsize=8)
    sparse_ticks(ax)
    if len(arms) > 1:
        _frame.swatch_legend(fig, arms, colours)
    return _frame.finish(fig, out_path, formats, dpi, metadata, legend_rows=len(arms) > 1)
