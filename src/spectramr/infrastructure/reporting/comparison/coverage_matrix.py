r"""What was measured: arms by metrics, one panel per phase.

A cell is *measured* (the arm has a finite value for that metric in that phase),
*not computed* (the run declared the metric and recorded that it could not compute
it, e.g. a full-reference metric on an inference run given no reference), or *not
logged*. The matrix says which claims a comparison can support before any number
is read: an arm with no value on a metric cannot be ranked on it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..metadata import RunMetadata
from ..style import column_width, pretty_label, use_default_style
from ._grid import chunks, write_chunks

#: Metrics per figure; a wider matrix continues in ``<stem>_2``.
MAX_COLUMNS = 30
STATES = ("not logged", "measured", "not computed")
_COLOURS = ["#ffffff", "#2a6f97", "#e9c46a"]
PHASE_NAMES = {"train": "training", "val": "validation", "test": "inference"}


def coverage_grid(
    table: pd.DataFrame, unavailable: list[dict], arms: list[str], phase: str, metrics: list[str]
) -> np.ndarray:
    """``(arms, metrics)`` of 0 not logged, 1 measured, 2 not computed, for one phase."""
    grid = np.zeros((len(arms), len(metrics)), dtype=int)
    one = table[(table["phase"] == phase) & table["mean"].notna()]
    seen = set(zip(one["arm"], one["metric"], strict=True))
    skipped = {
        (u.get("variant", u.get("arm")), u["metric"])
        for u in unavailable
        if u.get("phase") == phase and u.get("metric") != "(all)"
    }
    for i, arm in enumerate(arms):
        for j, metric in enumerate(metrics):
            if (arm, metric) in seen:
                grid[i, j] = 1
            elif (arm, metric) in skipped:
                grid[i, j] = 2
    return grid


def _draw(names, table, unavailable, arms, phases):
    use_default_style()
    width = column_width("double")
    height = max(1.2, 0.22 * len(arms) + 0.7) * len(phases)
    fig, axes = plt.subplots(len(phases), 1, figsize=(width, height), squeeze=False)
    for ax, phase in zip(axes.ravel(), phases, strict=True):
        grid = coverage_grid(table, unavailable, arms, phase, names)
        ax.imshow(grid, cmap=ListedColormap(_COLOURS), vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([pretty_label(m) for m in names], rotation=60, ha="right", fontsize=5)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms, fontsize=6)
        ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(arms)), minor=True)
        ax.grid(which="minor", color="#bbbbbb", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        ax.set_title(PHASE_NAMES[phase], fontsize="x-small", loc="left")
    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor="#777777", label=s)
            for c, s in zip(_COLOURS, STATES, strict=True)
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    phase_df: pd.DataFrame | None = None,
    unavailable: list[dict] | None = None,
    arms: list[str] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Coverage of ``phase_df`` (:func:`.phase_tables.phase_table`) by arm, metric and phase."""
    if phase_df is None or phase_df.empty or not arms:
        return None
    unavailable = unavailable or []
    phases = [p for p in ("train", "val", "test") if p in set(phase_df["phase"])]
    named = [u["metric"] for u in unavailable if u.get("metric") != "(all)"]
    metrics = list(dict.fromkeys([*phase_df["metric"], *named]))
    return write_chunks(
        out_path,
        metrics,
        MAX_COLUMNS,
        lambda names: _draw(names, phase_df, unavailable, arms, phases),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["MAX_COLUMNS", "STATES", "chunks", "coverage_grid", "make"]
