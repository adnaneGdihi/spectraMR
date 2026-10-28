r"""Raw metric bars by phase: training, validation, inference, one panel per metric.

x groups are the phases a metric was logged in, one bar per arm; the inference bar is
hatched. Each whisker is one standard deviation of the phase's rows: logged rows in training, the
slices of the read pass in validation, cases in inference, with n under the phase name. The phases are not on one footing (training is a windowed
mean over patches, validation and inference are full slices), so the bars are read across arms
within a phase, and a gap between phases is not a generalisation gap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..metadata import RunMetadata
from ._bars import HATCH, arm_colours, decorate, draw_bar, grouped_positions, wrap_label
from ._grid import MAX_PANELS, new_grid, write_chunks
from .metric_bars import _label

PHASE_NAMES = {"train": "training", "val": "validation", "test": "inference"}


def _tick(one, phase):
    """Phase name, with the largest n behind its bars when the phase has one above 1."""
    n = int(one.loc[one["phase"] == phase, "n"].max())
    return PHASE_NAMES[phase] + (f"\nn={n}" if n > 1 else "")


def _draw(chunk, table, arms):
    colours = arm_colours(arms)
    fig, axes = new_grid(len(chunk))
    for ax, metric in zip(axes, chunk, strict=True):
        one = table[table["metric"] == metric]
        phases = [p for p in ("train", "val", "test") if p in set(one["phase"])]
        layout = {
            PHASE_NAMES[p]: [a for a in arms if a in set(one.loc[one["phase"] == p, "arm"])]
            for p in phases
        }
        positions, spans = grouped_positions(layout)
        for phase in phases:
            for arm in layout[PHASE_NAMES[phase]]:
                row = one[(one["phase"] == phase) & (one["arm"] == arm)].iloc[0]
                mean, spread = float(row["mean"]), row["std"]
                spread = float(spread) if np.isfinite(spread) else np.nan
                draw_bar(
                    ax,
                    positions[(PHASE_NAMES[phase], arm)],
                    mean,
                    colours[arm],
                    lo=mean - spread,
                    hi=mean + spread,
                    hatch=HATCH if phase == "test" else None,
                )
        decorate(ax, positions, spans, layout, tick_labels=False, headers=False)
        ax.set_xticks([sum(spans[PHASE_NAMES[p]]) / 2 for p in phases])
        ax.set_xticklabels([_tick(one, p) for p in phases], fontsize=6)
        ax.set_ylabel(wrap_label(_label(metric)))
    fig.legend(
        handles=[Patch(facecolor=colours[a], edgecolor="#555555", label=a) for a in arms],
        loc="lower center",
        ncol=min(len(arms), 6),
        frameon=False,
        fontsize="x-small",
    )
    fig.suptitle(
        "bars start at zero; whisker: one SD of rows, slices or cases; hatched: inference; "
        "read differences between arms, not between phases",
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
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
    rows = phase_df[phase_df["mean"].notna()]
    metrics = list(dict.fromkeys(rows["metric"]))
    if not metrics:
        return None
    return write_chunks(
        out_path,
        metrics,
        max_panels,
        lambda chunk: _draw(chunk, rows, arms),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["make"]
