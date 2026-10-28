r"""Raw metric bars, arms grouped, one panel per metric.

x groups are the arm groups declared in the contrast (supervised, self-supervised, ...);
one bar per arm. The value is the arm's mean at one phase, read from the phase table. Bars
start at zero and a dotted line carries the baseline arm's value, so a difference of a
fraction of the metric stays readable without truncating the axis. Validation is a single
pass and has no whisker; training and inference carry one standard deviation of their rows or
cases. Different metrics sit in different panels because their scales are not comparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..metadata import RunMetadata
from ._bars import GUIDE, arm_colours, arm_groups, decorate, draw_bar, grouped_positions, wrap_label
from ._grid import MAX_PANELS, new_grid, write_chunks
from .metric_bars import _label


def _draw(chunk, table, arms, groups, baseline, phase):
    layout = arm_groups(arms, groups)
    positions, spans = grouped_positions(layout)
    colours = arm_colours(arms)
    fig, axes = new_grid(len(chunk))
    for ax, metric in zip(axes, chunk, strict=True):
        one = table[(table["metric"] == metric) & (table["phase"] == phase)].set_index("arm")
        for (_group, arm), x in positions.items():
            if arm not in one.index:
                continue
            row = one.loc[arm]
            spread = float(row["std"]) if np.isfinite(row["std"]) else np.nan
            mean = float(row["mean"])
            draw_bar(ax, x, mean, colours[arm], lo=mean - spread, hi=mean + spread)
        if baseline in one.index and np.isfinite(one.loc[baseline, "mean"]):
            ax.axhline(float(one.loc[baseline, "mean"]), color=GUIDE, ls=":", lw=1.0, zorder=0)
        decorate(ax, positions, spans, layout)
        ax.set_ylabel(wrap_label(_label(metric)))
    note = (
        "validation: one pass, no whisker" if phase == "val" else "whisker: one SD of rows or cases"
    )
    fig.suptitle(
        f"{phase}: bars start at zero; dotted line = {baseline}; {note}",
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    phase_df: pd.DataFrame | None = None,
    arms: list[str] | None = None,
    groups: dict[str, str] | None = None,
    baseline: str | None = None,
    phase: str = "val",
    max_panels: int = MAX_PANELS,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """One panel per metric of ``phase_df`` at ``phase``; ``None`` when that phase has no rows."""
    if phase_df is None or phase_df.empty or not arms:
        return None
    rows = phase_df[(phase_df["phase"] == phase) & phase_df["mean"].notna()]
    metrics = list(dict.fromkeys(rows["metric"]))
    if not metrics:
        return None
    return write_chunks(
        out_path,
        metrics,
        max_panels,
        lambda chunk: _draw(chunk, rows, arms, groups, baseline, phase),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["make"]
