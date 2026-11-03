"""Figure 6.1 — critical-difference diagram: arms ranked per subject on the primary metric.

Each subject ranks the arms (1 = best, ties share); the axis is the mean rank. Arms joined
by a bar are within the Nemenyi critical difference: not distinguishable at 0.05. The
Friedman p-value and N subjects head the figure. Needs three or more arms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.core.design_tokens import colours as tokens
from spectramr.core.metrics.meta_evaluation.critical_difference import (
    cliques,
    friedman_p,
    nemenyi_cd,
    rank_rows,
)
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _slices


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    metric: str | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    tab = _slices.table(slice_df)
    if (
        tab is None
        or not _slices.has_identity(tab)
        or (pick := _slices.primary(tab, metric)) is None
    ):
        return None
    name, hib = pick
    subj = _slices.subject_means(tab, name)
    if subj.shape[1] < 3 or subj.shape[0] < 2:
        return None
    ranks = rank_rows(subj.to_numpy(), higher_is_better=hib)
    mean = ranks.mean(axis=0)
    order = np.argsort(mean, kind="stable")
    arms, mean = [str(subj.columns[i]) for i in order], mean[order]
    n, k = subj.shape
    cd = nemenyi_cd(k, n)
    ink = tokens("light")
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(width, 1.4 + 0.24 * half))
    ax.set_xlim(1, k)
    ax.set_ylim(-(half + 1.2), 2.0)
    ax.axhline(0, color=ink["ink"], lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.18], color=ink["ink"], lw=0.6)
        ax.text(r, 0.35, str(r), ha="center", fontsize=7)
    for i, (arm, r) in enumerate(zip(arms, mean, strict=True)):
        left = i < half
        row = -(i + 1) if left else -(k - i)
        xe = 1 if left else k
        ax.plot([r, r, xe], [0, row, row], color=ink["ink_2"], lw=0.6)
        ax.text(
            xe - 0.05 if left else xe + 0.05,
            row,
            f"{arm}  {r:.2f}" if left else f"{r:.2f}  {arm}",
            ha="right" if left else "left",
            va="center",
            fontsize=7,
        )
    for depth, (lo, hi) in enumerate(cliques(mean, cd)):
        y = -0.35 - 0.22 * depth
        ax.plot(
            [mean[lo] - 0.03, mean[hi] + 0.03],
            [y, y],
            color=ink["accent"],
            lw=2.2,
            solid_capstyle="round",
        )
    ax.plot([1, 1 + cd], [1.35, 1.35], color=ink["ink"], lw=1.2)
    ax.text(1 + cd / 2, 1.5, f"CD = {cd:.2f}", ha="center", fontsize=7)
    p = friedman_p(ranks)
    ax.set_title(
        f"{name}: mean rank over {n} subjects, Friedman p = {p:.2g} ({_slices.PHASE_NOTE})",
        fontsize=8,
    )
    ax.axis("off")
    return _frame.finish(fig, out_path, formats, dpi, metadata)
