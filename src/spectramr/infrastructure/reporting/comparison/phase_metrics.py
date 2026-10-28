r"""Every metric across training, validation and inference, one panel each.

x = phase, one colour per arm. Validation is the metric at the arm's selected
pass (a dot, one value). Training is the mean of its last logged rows and
inference the mean over its cases; both carry a bar of one standard deviation
of those rows or cases, so the bar is spread, not an interval for the mean.
The panel says which. A metric logged in a single phase still gets a panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metadata import RunMetadata, stamp_figure, write_sidecar
from ..style import (
    colour_for,
    column_width,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)
from ._grid import MAX_PANELS, PANEL_COLUMNS
from ._grid import chunks as _chunks
from .metric_bars import _label as direction_label

PHASE_NAMES = {"train": "training", "val": "validation", "test": "inference"}


def _panel(ax, rows: pd.DataFrame, arms: list[str], colours: dict[str, str]) -> None:
    phases = [p for p in ("train", "val", "test") if p in set(rows["phase"])]
    width = 0.8 / len(arms)
    for a_i, arm in enumerate(arms):
        for p_i, phase in enumerate(phases):
            one = rows[(rows["arm"] == arm) & (rows["phase"] == phase)]
            if one.empty:
                continue
            mean, std = float(one["mean"].iloc[0]), one["std"].iloc[0]
            x = p_i - 0.4 + width * (a_i + 0.5)
            ax.errorbar(
                x,
                mean,
                yerr=None if not np.isfinite(std) else std,
                fmt="o",
                ms=3.5,
                color=colours[arm],
                mec="black",
                mew=0.3,
                capsize=2,
                label=arm if p_i == 0 else None,
            )
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([PHASE_NAMES[p] for p in phases], fontsize="x-small")


def _figure(chunk: list[str], table: pd.DataFrame, arms: list[str], colours: dict[str, str]):
    use_default_style()
    cols = min(len(chunk), PANEL_COLUMNS)
    rows_n = -(-len(chunk) // cols)
    width = column_width("double")
    fig, axes = plt.subplots(rows_n, cols, figsize=(width, width * 0.36 * rows_n), squeeze=False)
    for i, metric in enumerate(chunk):
        ax = axes[i // cols][i % cols]
        _panel(ax, table[table["metric"] == metric], arms, colours)
        ax.set_ylabel(direction_label(metric))
        ax.set_title(pretty_label(metric), fontsize="x-small", loc="left")
        if len(chunk) > 1:
            panel_label(ax, chr(ord("a") + i))
    for j in range(len(chunk), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    handles = next((h for h in (a.get_legend_handles_labels() for a in axes.ravel()) if h[0]), None)
    if handles:
        fig.legend(*handles, loc="lower center", ncol=min(len(arms), 4), frameon=False)
    fig.suptitle(
        "validation: one pass; training: last rows, inference: cases; bars are one standard "
        "deviation of those rows or cases",
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    phase_df: pd.DataFrame | None = None,
    arms: list[str] | None = None,
    max_panels: int = MAX_PANELS,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """One panel per metric of ``phase_df`` (:func:`.phase_tables.phase_table`)."""
    if phase_df is None or phase_df.empty or not arms:
        return None
    if max_panels < 1:
        raise ValueError(f"max_panels must be at least 1, got {max_panels}")
    colours = {arm: colour_for(arm, i) for i, arm in enumerate(arms)}
    names = list(dict.fromkeys(phase_df["metric"]))
    out_path, primary = Path(out_path), None
    for k, chunk in enumerate(_chunks(names, max_panels), start=1):
        target = out_path if k == 1 else out_path.with_name(f"{out_path.stem}_{k}{out_path.suffix}")
        fig = _figure(chunk, phase_df, arms, colours)
        if metadata is not None:
            stamp_figure(fig, metadata)
        written = save_figure(fig, target.parent, target.stem, formats=formats, dpi=dpi)
        plt.close(fig)
        path = written.get(formats[0])
        if metadata is not None and path is not None:
            write_sidecar(path, metadata)
        primary = primary or path
    return primary
