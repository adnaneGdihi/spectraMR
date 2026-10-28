r"""Change from the baseline as bars, arms grouped, one panel per metric.

A bar is the mean paired difference (arm minus baseline) and its whisker the bootstrap 95 %
interval, both read from the rows of ``paired_stats.csv``, so the figure and the table agree.
The small points are the subject differences behind the mean. Validation is solid and inference
hatched; a bar is faded when its p-value is not below 0.05 (Holm-adjusted within a declared family,
otherwise the raw paired t-test p). Bars start at zero, which here is "no change", so the whole
axis is meaningful. A reciprocal metric is on the ``1/N_eq`` scale, where lower is better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..metadata import RunMetadata
from ._bars import HATCH, arm_colours, arm_groups, decorate, draw_bar, grouped_positions, wrap_label
from ._grid import MAX_PANELS, new_grid, write_chunks
from .effect_forest import adjusted_p
from .metric_bars import _label

_OFFSET = {"val": -0.21, "test": 0.21}


def _draw(chunk, paired_df, paired, arms, groups, baseline):
    variants = [a for a in arms if a != baseline]
    layout = arm_groups(variants, groups)
    positions, spans = grouped_positions(layout)
    group_of = {arm: g for g, members in layout.items() for arm in members}
    colours = arm_colours(arms)
    fig, axes = new_grid(len(chunk))
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, chunk, strict=True):
        one = paired_df[paired_df["metric"] == metric]
        both = len(set(one["phase"])) > 1
        for _, r in one.iterrows():
            arm = r["variant"]
            if arm not in group_of:
                continue
            centre = positions[(group_of[arm], arm)]
            x = centre + (_OFFSET.get(r["phase"], 0.0) if both else 0.0)
            draw_bar(
                ax,
                x,
                r["mean_diff"],
                colours[arm],
                lo=r["ci_low"],
                hi=r["ci_high"],
                hatch=HATCH if r["phase"] == "test" else None,
                width=0.4 if both else 0.8,
                alpha=1.0 if adjusted_p(r) < 0.05 else 0.4,
            )
            pair = paired.get((r["phase"], metric, arm))
            if pair is not None:
                diff = (pair.table["a"] - pair.table["b"]).to_numpy(dtype=float)
                ax.scatter(
                    x + rng.uniform(-0.12, 0.12, diff.size),
                    diff,
                    s=2,
                    color="black",
                    alpha=0.5,
                    linewidths=0,
                    zorder=5,
                )
        decorate(ax, positions, spans, layout)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        scale = one["scale"].iloc[0] if len(one) else "metric"
        label = _label(metric) + (" (1/N_eq)" if "1/N_eq" in scale else "")
        ax.set_ylabel(wrap_label(label) + "\narm - baseline")
    fig.legend(
        handles=[
            Patch(facecolor="#888888", edgecolor="#555555", label="validation"),
            Patch(facecolor="#888888", edgecolor="#555555", hatch=HATCH, label="inference"),
            Patch(facecolor="#888888", edgecolor="#555555", alpha=0.4, label="p >= 0.05"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="x-small",
    )
    fig.suptitle(
        f"difference = arm minus {baseline}; whisker: bootstrap 95 % interval; points: subjects "
        "(inference: cases); faded: p >= 0.05 (Holm-adjusted within a declared family)",
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    paired_df: pd.DataFrame | None = None,
    paired: dict | None = None,
    arms: list[str] | None = None,
    groups: dict[str, str] | None = None,
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
    return write_chunks(
        out_path,
        list(dict.fromkeys(finite["metric"])),
        max_panels,
        lambda chunk: _draw(chunk, finite, paired or {}, arms, groups, baseline),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["make"]
