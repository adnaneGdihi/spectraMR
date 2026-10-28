"""Figure 3.6 — bias against variance, per slice.

For the n2n protocol, each slice's squared bias against its repetition variance on log
axes, with the identity line: a slice below it is noise-limited, above it is
bias-limited. Needs ``bias_squared`` and ``rep_variance`` per slice; soft-skips otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

_NEED = ("bias_squared", "rep_variance")


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
    if table is None or not set(_NEED) <= set(table.columns):
        return None
    pos = table[list(_NEED)].dropna()
    pos = pos[(pos > 0).all(axis=1)]
    if len(pos) < 3:
        return None
    methods = _frame.methods_of(pos)
    colours = arm_palette(methods)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, width * 0.9))
    for method in methods:
        sub = pos.xs(method, level="method")
        ax.scatter(
            sub["rep_variance"],
            sub["bias_squared"],
            s=3,
            color=colours[method],
            alpha=0.3,
            linewidths=0,
            rasterized=True,
            label=method,
        )
    lo, hi = float(pos.to_numpy().min()), float(pos.to_numpy().max())
    ax.plot([lo, hi], [lo, hi], color="#666666", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("repetition variance")
    ax.set_ylabel("bias$^2$")
    sparse_ticks(ax)
    if len(methods) > 1:
        _frame.swatch_legend(fig, methods, colours)
    return _frame.finish(fig, out_path, formats, dpi, metadata, legend_rows=len(methods) > 1)
