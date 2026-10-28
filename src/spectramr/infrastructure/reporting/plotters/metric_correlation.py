r"""Figure 1.17 — Validation-metric correlation heatmap.

Spearman rank correlation between validation metrics across training steps.
Two diagnostic reads:

* |ρ| ≈ 1 off-diagonal → the two metrics are redundant (report one).
* a metric uncorrelated with every other → it is measuring something the
  others don't (keep it), or it is degenerate/constant (the aggregator's
  NaN guard drops constant series — an absent row is the tell).

Spearman (rank) rather than Pearson: metric scales differ wildly
(dB vs [0,1] vs unbounded losses) and training curves are monotone-ish,
so rank correlation is the meaningful comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    column_width,
    pretty_label,
    save_figure,
    use_default_style,
)

_MIN_STEPS = 3  # fewer paired observations make rank correlation meaningless


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    split: str = "val",
    max_metrics: int = 10,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Render the Spearman correlation heatmap. ``None`` when under-determined."""
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns:
        return None
    sub = df[df["split"] == split] if "split" in df.columns else df
    if sub.empty:  # fall back to train-split metrics when no validation was run
        sub = df[df["split"] == "train"] if "split" in df.columns else df
    sub = sub.dropna(subset=["value"])
    sub = sub[sub["step"].fillna(-1) >= 0]
    if sub.empty:
        return None

    pivot = sub.pivot_table(
        index="step", columns="metric", values="value", aggfunc="mean"
    ).sort_index()
    # constant columns have zero rank variance → NaN ρ; drop them up front
    pivot = pivot.loc[:, pivot.nunique() > 1]
    if pivot.shape[1] > max_metrics:
        pivot = pivot.iloc[:, :max_metrics]
    if pivot.shape[0] < _MIN_STEPS or pivot.shape[1] < 2:
        return None
    corr = pivot.corr(method="spearman")

    use_default_style()
    size = max(2.2, 0.42 * corr.shape[0] + 1.0)
    # constrained layout: tight_layout mis-sizes the colorbar + rotated tick
    # labels and lets the title collide with the bar
    fig, ax = plt.subplots(
        figsize=(min(size * 1.3, column_width("double")), min(size, column_width("double"))),
        layout="constrained",
    )
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [pretty_label(str(c)) for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Spearman ρ")
    ax.set_title(f"{split} metric correlation", pad=8)
    if metadata is not None:
        stamp_figure(fig, metadata)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
