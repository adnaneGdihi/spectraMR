r"""Paired effects against the baseline, one panel per metric.

Each row is an arm; the point is the mean paired difference (arm minus baseline) and
the bar its bootstrap 95 % interval, both read from the rows of ``paired_stats.csv``,
so the figure and the table cannot disagree. Validation is a circle, inference a
square. A filled marker is significant at 0.05: the Holm-adjusted p when the contrast
declared a family, the raw paired t-test p otherwise. A reciprocal metric
(``nex_equivalent``) is on the ``1/N_eq`` scale, where lower is better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..metadata import RunMetadata
from ..style import colour_for
from ._grid import MAX_PANELS, new_grid, write_chunks
from .metric_bars import _label

_MARKERS = {"val": "o", "test": "s"}
_OFFSET = {"val": -0.12, "test": 0.12}


def adjusted_p(row: pd.Series) -> float:
    """The p-value the marker reports: Holm-adjusted if the row has one, else raw."""
    return float(row["p_holm"]) if pd.notna(row["p_holm"]) else float(row["p_ttest"])


def _draw(chunk: list[str], paired: pd.DataFrame, arms: list[str], baseline: str):
    rows = [a for a in arms if a != baseline]
    fig, axes = new_grid(len(chunk))
    colours = {a: colour_for(a, i) for i, a in enumerate(arms)}
    for ax, metric in zip(axes, chunk, strict=True):
        one = paired[paired["metric"] == metric]
        for _, r in one.iterrows():
            y = rows.index(r["variant"]) + _OFFSET.get(r["phase"], 0.0)
            colour = colours[r["variant"]]
            ax.errorbar(
                r["mean_diff"],
                y,
                xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
                fmt=_MARKERS.get(r["phase"], "o"),
                ms=4,
                color=colour,
                mfc=colour if adjusted_p(r) < 0.05 else "none",
                mec=colour,
                capsize=2,
                lw=0.9,
            )
        ax.axvline(0.0, color="#555555", lw=0.7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=6)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        scale = one["scale"].iloc[0] if len(one) else "metric"
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.set_xlabel(_label(metric) + (" (1/N_eq)" if "1/N_eq" in scale else ""))
    handles = [
        Line2D([], [], marker="o", ls="", color="#444444", label="validation"),
        Line2D([], [], marker="s", ls="", color="#444444", label="inference"),
        Line2D([], [], marker="o", ls="", color="#444444", label="filled: p < 0.05"),
        Line2D([], [], marker="o", ls="", color="#444444", mfc="none", label="hollow: not"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize="x-small")
    fig.suptitle(
        f"difference = arm minus {baseline}; bars: bootstrap 95 % interval of the mean "
        "paired difference; p is Holm-adjusted within a declared family, else uncorrected",
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    paired_df: pd.DataFrame | None = None,
    arms: list[str] | None = None,
    baseline: str | None = None,
    max_panels: int = MAX_PANELS,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """One panel per metric of ``paired_df`` (the ``paired_stats.csv`` rows)."""
    if paired_df is None or paired_df.empty or not arms or baseline is None:
        return None
    finite = paired_df[np.isfinite(paired_df[["mean_diff", "ci_low", "ci_high"]]).all(axis=1)]
    if finite.empty:
        return None
    metrics = list(dict.fromkeys(finite["metric"]))
    return write_chunks(
        out_path,
        metrics,
        max_panels,
        lambda chunk: _draw(chunk, finite, arms, baseline),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["adjusted_p", "make"]
