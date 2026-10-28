r"""Forest plot of effect sizes (Δ vs baseline) with bootstrap CIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters import _stats
from spectramr.infrastructure.reporting.style import (
    colour_for,
    column_width,
    save_figure,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metric=None,
    baseline=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if df is None or df.empty or "metric" not in df.columns:
        return None
    if metric is None:
        metric = "psnr" if "psnr" in set(df["metric"]) else df["metric"].iloc[0]
    sub = df[df["metric"] == metric]
    if "subject_id" not in sub.columns:
        return None
    pivot = sub.pivot_table(index="subject_id", columns="method", values="value")
    methods = [m for m in pivot.columns]
    if baseline is None or baseline not in methods:
        baseline = methods[0]
    base = pivot[baseline].to_numpy()
    rows = []
    for m in methods:
        if m == baseline:
            continue
        d = pivot[m].to_numpy() - base
        d = d[~np.isnan(d)]
        if d.size < 2:
            continue
        lo, hi = _stats.bootstrap_ci(d, seed=0)
        rows.append((m, d.mean(), lo, hi))
    if not rows:
        return None
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), 0.4 * len(rows) + 0.8))
    for i, (m, eff, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=colour_for(m, i), lw=1.2)
        ax.plot(eff, i, "o", color=colour_for(m, i))
    ax.axvline(0, color="#999999", lw=0.7, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(f"Δ {metric.upper()} vs {baseline}")
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
