r"""Pairwise significance matrix.

Lower-triangular heatmap of Holm–Bonferroni-corrected paired-bootstrap
p-values across methods for one metric. Annotates each cell with the
corrected p (or a significance star).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters import _stats
from spectramr.infrastructure.reporting.style import column_width, save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metric=None,
    predictions_df=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    frame = predictions_df if predictions_df is not None else df
    if frame is None or frame.empty or "metric" not in frame.columns:
        return None
    if metric is None:
        metric = "psnr" if "psnr" in set(frame["metric"]) else frame["metric"].iloc[0]
    sub = frame[frame["metric"] == metric]
    if "subject_id" not in sub.columns:
        return None
    methods = list(sub["method"].unique())
    if len(methods) < 2:
        return None
    pivot = sub.pivot_table(index="subject_id", columns="method", values="value")
    pairs, raw = [], []
    for i in range(len(methods)):
        for k in range(i):
            a = pivot[methods[i]].to_numpy()
            b = pivot[methods[k]].to_numpy()
            m = ~(np.isnan(a) | np.isnan(b))
            pairs.append((i, k))
            raw.append(_stats.paired_bootstrap_pvalue(a[m], b[m], seed=0) if m.sum() > 1 else 1.0)
    adj = _stats.holm_bonferroni(raw)
    mat = np.full((len(methods), len(methods)), np.nan)
    for (i, k), q in zip(pairs, adj):
        mat[i][k] = q
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single")))
    im = ax.imshow(mat, cmap="viridis_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for (i, k), q in zip(pairs, adj):
        ax.text(
            k,
            i,
            f"{q:.3f}",
            ha="center",
            va="center",
            fontsize=5,
            color="white" if q < 0.05 else "black",
        )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Holm–Bonferroni p")
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
