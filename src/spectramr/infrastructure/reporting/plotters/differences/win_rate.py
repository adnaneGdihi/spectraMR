"""Figure 6.2 — win rate: share of subjects on which the row arm beats the column arm.

Ties count half. Cells are coloured around 50 % and ringed with an asterisk when a
two-sided sign test survives Holm correction over all pairs. Arms are ordered by their
overall win rate.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.core.design_tokens import colours as tokens
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _slices


def win_matrix(subj: pd.DataFrame, higher_is_better: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(win share, Holm-significant) for every ordered arm pair."""
    from scipy.stats import binomtest

    arms = list(subj.columns)
    sign = 1.0 if higher_is_better else -1.0
    win = pd.DataFrame(np.nan, index=arms, columns=arms)
    pvals = {}
    for a, b in combinations(arms, 2):
        d = sign * (subj[a] - subj[b])
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        win.loc[a, b] = (wins + 0.5 * (d == 0).sum()) / len(d)
        win.loc[b, a] = 1.0 - win.loc[a, b]
        pvals[(a, b)] = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    sig = pd.DataFrame(False, index=arms, columns=arms)
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    for rank, ((a, b), p) in enumerate(ordered):
        if p * (m - rank) >= 0.05:
            break
        sig.loc[a, b] = sig.loc[b, a] = True
    return win, sig


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
    if subj.shape[1] < 2 or subj.shape[0] < 3:
        return None
    win, sig = win_matrix(subj, hib)
    order = win.mean(axis=1).sort_values(ascending=False).index
    win, sig = win.loc[order, order], sig.loc[order, order]
    k = len(order)
    ink = tokens("light")
    use_default_style()
    import matplotlib.pyplot as plt

    side = min(column_width("double"), 1.2 + 0.42 * k)
    fig, ax = plt.subplots(figsize=(side + 1.0, side))
    im = ax.imshow(win.to_numpy(dtype=float), cmap="RdBu", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if i != j:
                v = win.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.0%}{'*' if sig.iat[i, j] else ''}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(v - 0.5) > 0.3 else ink["ink"],
                )
    ax.set_xticks(range(k), [str(c) for c in order], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(k), [str(c) for c in order], fontsize=7)
    ax.set_title(
        f"{name}: row beats column, share of {len(subj)} subjects ({_slices.PHASE_NOTE})",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=lambda v, _: f"{v:.0%}")
    return _frame.finish(fig, out_path, formats, dpi, metadata)
