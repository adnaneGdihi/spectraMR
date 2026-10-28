r"""Figure 1.3 — Loss decomposition stack.

Stacked-area chart of the individual loss terms over training, with the
total overlaid as a thin line. Demonstrates which term dominates and
justifies weighting choices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.legend import (
    collect_handles,
    reserve_legend_space,
    shared_legend,
    sparse_ticks,
)
from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    OKABE_ITO,
    pretty_label,
    save_figure,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    component_keyword: str = "loss",
    total_metric: str | None = "loss",
    method: str | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the stacked-area loss decomposition.

    Args:
        component_keyword: Substring used to identify component metrics
            (default: ``"loss"``). All metrics whose names contain it
            and are NOT ``total_metric`` are stacked.
        total_metric: Metric name to overlay as the total. Set to None
            to skip the overlay.
        method: Optionally restrict to a single method name.
    """
    out_path = Path(out_path)
    if df.empty:
        return None
    sub = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    if method is not None:
        sub = sub[sub["method"] == method]
    if sub.empty:
        return None
    metrics = [m for m in sub["metric"].unique() if component_keyword in m.lower()]
    if not metrics:
        return None
    component_metrics = [m for m in metrics if m != total_metric]
    if not component_metrics:
        return None

    use_default_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.2))

    # Build a per-step matrix of component values
    pivot = (
        sub[sub["metric"].isin(component_metrics)]
        .pivot_table(index="step", columns="metric", values="value", aggfunc="mean")
        .fillna(0)
        .sort_index()
    )
    if pivot.empty:
        plt.close(fig)
        return None
    colours = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(pivot.shape[1])]
    if pivot.shape[1] == 1:
        # one component is its own total: a line, not a stack with the same line over it
        ax.plot(pivot.index, pivot.iloc[:, 0].values, color=colours[0], linewidth=0.9,
                label=pretty_label(pivot.columns[0]))
        total_metric = None
    else:
        ax.stackplot(
            pivot.index,
            pivot.T.values,
            labels=[pretty_label(c) for c in pivot.columns],
            colors=colours,
            alpha=0.85,
            linewidth=0,
        )

    # Total overlay
    if total_metric is not None:
        total_df = sub[sub["metric"] == total_metric].groupby("step")["value"].mean()
        if not total_df.empty:
            ax.plot(
                total_df.index,
                total_df.values,
                color="black",
                linewidth=0.8,
                label=f"total ({pretty_label(total_metric)})",
            )

    ax.set_xlabel(pretty_label("step"))
    ax.set_ylabel("loss component")
    shared_legend(fig, *collect_handles([ax]))
    sparse_ticks(ax)
    fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 1))
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary


# Silence "imported but unused"
_ = np
