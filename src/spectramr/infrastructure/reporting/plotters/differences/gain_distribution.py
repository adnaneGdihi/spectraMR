"""Figure 6.5 — gain over the input, per slice, per arm.

Each dot is one slice: the arm's metric minus the input's (zero-filled, or the noisy
excitation for n2n) on the same slice. The tick is the median; the label gives the share of
slices on which the arm is worse than its own input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.core.design_tokens import colours as tokens
from spectramr.infrastructure.reporting.legend import arm_palette
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _slices

PAIRS = (("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("sense_psnr", "SENSE PSNR (dB)"))


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    tab = _slices.table(slice_df)
    if tab is None:
        return None
    pick = next(((m, lab) for m, lab in PAIRS if {m, f"zf_{m}"} <= set(tab.columns)), None)
    if pick is None:
        return None
    metric, label = pick
    gain = (tab[metric] - tab[f"zf_{metric}"]).dropna()
    if gain.empty:
        return None
    arms = [a for a in _frame.methods_of(tab) if a in gain.index.get_level_values("method")]
    colours = arm_palette(arms)
    ink = tokens("light")
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, 0.7 + 0.32 * len(arms)))
    rng = np.random.default_rng(0)
    ax.axvline(0, color=ink["muted"], lw=0.8, ls="--")
    for y, arm in enumerate(arms):
        v = gain.xs(arm, level="method").to_numpy()
        ax.scatter(
            v,
            y + rng.uniform(-0.22, 0.22, v.size),
            s=3,
            color=colours[arm],
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        med = float(np.median(v))
        ax.plot([med, med], [y - 0.3, y + 0.3], color=ink["ink"], lw=1.4)
        ax.text(
            1.01,
            y,
            f"{(v < 0).mean():.0%} worse",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=6,
            color=ink["ink_2"],
        )
    ax.set_yticks(range(len(arms)), arms, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {label}, output - input")
    ax.set_title(f"Per-slice gain over the input ({_slices.PHASE_NOTE})", fontsize=8)
    return _frame.finish(fig, out_path, formats, dpi, metadata)
