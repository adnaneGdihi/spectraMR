"""Figure 6.3 — band fidelity: per-slice error in each k-space band, per arm.

One panel per band (low, mid, high), each on its own log axis because the bands differ by
orders of magnitude. Dot = slice median, bar = interquartile range. Shows which band an
arm loses, which a mean over the whole image hides.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.ticker import NullFormatter

from spectramr.infrastructure.reporting.legend import arm_palette
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _slices

BANDS = ("low", "mid", "high")
FAMILIES = (("band_corrected_mse", "corrected MSE"), ("band_bias_squared", "bias$^2$"))


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
    tab = _slices.table(slice_df)
    if tab is None:
        return None
    fam = next(
        ((s, lab) for s, lab in FAMILIES if all(f"{s}_{b}" in tab.columns for b in BANDS)), None
    )
    if fam is None:
        return None
    stem, label = fam
    arms = _frame.methods_of(tab)
    colours = arm_palette(arms)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(width, 1.1 + 0.28 * len(arms)))
    for ax, band in zip(axes, BANDS, strict=True):
        col = tab[f"{stem}_{band}"]
        for y, arm in enumerate(arms):
            v = col.xs(arm, level="method").dropna()
            v = v[v > 0]
            if v.empty:
                continue
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.plot([q1, q3], [y, y], color=colours[arm], lw=1.4, alpha=0.6)
            ax.plot(med, y, "o", color=colours[arm], ms=4)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{band} band", fontsize=8)
        ax.set_xlabel(label)
    axes[0].set_yticks(range(len(arms)), arms, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle(f"Per-slice error by k-space band ({_slices.PHASE_NOTE})", fontsize=8)
    return _frame.finish(fig, out_path, formats, dpi, metadata)
