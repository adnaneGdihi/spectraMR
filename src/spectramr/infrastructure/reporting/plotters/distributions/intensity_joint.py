"""Figure 3.8 — prediction against target, pixel by pixel.

Joint density of target and predicted intensity for one case, counts on a log colour
scale with the identity line. A bias shows as a band off the diagonal, a noisy reference
as a vertical smear at low target intensity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: list[dict] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not cases:
        return None
    case = cases[0]
    pred = np.asarray(case.get("prediction"), dtype=np.float64).ravel()
    target = np.asarray(case.get("target"), dtype=np.float64).ravel()
    if pred.size == 0 or pred.shape != target.shape:
        return None
    top = float(max(target.max(), pred.max()))
    if top <= 0:
        return None
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, width * 0.9))
    h = ax.hist2d(target, pred, bins=96, range=[[0, top], [0, top]], norm=LogNorm(), cmap="viridis")
    ax.plot([0, top], [0, top], color="white", lw=0.6, ls=":")
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_aspect("equal")
    cbar = fig.colorbar(h[3], ax=ax, shrink=0.8)
    cbar.set_label("pixels")
    return _frame.finish(fig, out_path, formats, dpi, metadata)
