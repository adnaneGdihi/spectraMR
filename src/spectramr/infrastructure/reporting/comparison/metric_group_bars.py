r"""Improvement over the baseline in percent, metrics grouped, arms as bars.

Raw values of different metrics share no scale, so each arm is shown as its improvement over
the baseline: ``100 * sign * mean difference / |baseline mean|``, where ``sign`` makes up mean
better by the metric's declared direction (a reciprocal metric is read on ``1/N_eq``, where the
direction is the opposite). The whisker is the bootstrap 95 % interval of the mean difference on
the same scale. One panel per phase. A metric with no declared direction, or with a baseline mean
of zero, has no percentage and is named in the note, never drawn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from spectramr.core.metrics.metric_directions import resolve_direction
from spectramr.core.metrics.reciprocal import is_reciprocal

from ..metadata import RunMetadata
from ..style import pretty_label
from ._bars import arm_colours, decorate, draw_bar, grouped_positions, wrap_label
from ._grid import new_grid, write_chunks

#: Metrics per figure (the x groups); more continue in numbered files.
MAX_METRICS = 8


def improvement_rows(paired_df: pd.DataFrame, paired: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Percent improvement and its interval per (phase, metric, arm), and the metrics left out."""
    rows, omitted = [], {}
    for _, r in paired_df.iterrows():
        metric = r["metric"]
        higher = resolve_direction(metric)
        if higher is None:
            omitted[metric] = "no declared direction"
            continue
        if is_reciprocal(metric):
            higher = not higher  # analysed as 1/N_eq, which runs the other way
        pair = paired.get((r["phase"], metric, r["variant"]))
        base = abs(float(pair.table["b"].mean())) if pair is not None else np.nan
        if not np.isfinite(base) or base == 0.0:
            omitted[metric] = "baseline mean is zero or unknown"
            continue
        sign = 1.0 if higher else -1.0
        edge = sorted([sign * 100 * r["ci_low"] / base, sign * 100 * r["ci_high"] / base])
        rows.append(
            {
                "phase": r["phase"],
                "metric": metric,
                "variant": r["variant"],
                "pct": sign * 100 * r["mean_diff"] / base,
                "lo": edge[0],
                "hi": edge[1],
            }
        )
    return pd.DataFrame(rows, columns=["phase", "metric", "variant", "pct", "lo", "hi"]), omitted


def _draw(chunk, rows, variants, omitted, phases):
    colours = arm_colours(variants)
    fig, axes = new_grid(len(phases), row_height=0.3, columns=1)
    for ax, phase in zip(axes, phases, strict=True):
        one = rows[rows["phase"] == phase]
        shown = [m for m in chunk if m in set(one["metric"])]  # no empty slot for a metric
        layout = dict.fromkeys(shown, variants)
        positions, spans = grouped_positions(layout)
        for metric in shown:
            for arm in variants:
                hit = one[(one["metric"] == metric) & (one["variant"] == arm)]
                if not hit.empty:
                    r = hit.iloc[0]
                    draw_bar(
                        ax, positions[(metric, arm)], r["pct"], colours[arm], lo=r["lo"], hi=r["hi"]
                    )
        decorate(ax, positions, spans, layout, tick_labels=False, headers=False)
        ax.set_xticks([sum(spans[m]) / 2 for m in shown])
        ax.set_xticklabels([wrap_label(pretty_label(m), 14) for m in shown], fontsize=6)
        ax.set_ylabel("improvement (%)")
        ax.set_title(
            "validation" if phase == "val" else "inference", fontsize="x-small", loc="left"
        )
    fig.legend(
        handles=[Patch(facecolor=colours[a], edgecolor="#555555", label=a) for a in variants],
        loc="lower center",
        ncol=min(len(variants), 6),
        frameon=False,
        fontsize="x-small",
    )
    note = "improvement over the baseline; up = better by each metric's declared direction; whisker: bootstrap 95 % interval"
    if omitted:
        note += "; not drawn: " + ", ".join(f"{m} ({why})" for m, why in list(omitted.items())[:4])
    fig.suptitle(note, fontsize="x-small")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    paired_df: pd.DataFrame | None = None,
    paired: dict | None = None,
    arms: list[str] | None = None,
    baseline: str | None = None,
    max_metrics: int = MAX_METRICS,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Percent-improvement bars of the ``paired_stats.csv`` rows; ``None`` if nothing has a percentage."""
    if paired_df is None or paired_df.empty or not paired or not arms or baseline is None:
        return None
    rows, omitted = improvement_rows(paired_df, paired)
    if rows.empty:
        return None
    variants = [a for a in arms if a != baseline and a in set(rows["variant"])]
    phases = [p for p in ("val", "test") if p in set(rows["phase"])]
    metrics = list(dict.fromkeys(rows["metric"]))
    return write_chunks(
        out_path,
        metrics,
        max_metrics,
        lambda chunk: _draw(chunk, rows, variants, omitted, phases),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["MAX_METRICS", "improvement_rows", "make"]
