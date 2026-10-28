r"""Per-method metric distributions with paired-significance annotation.

Box + violin + jittered strip ("raincloud") for each method on each metric,
with a paired-bootstrap significance bracket (Holm–Bonferroni corrected)
between the top two methods.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.legend import sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters import _stats
from spectramr.infrastructure.reporting.style import (
    colour_for,
    column_width,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)


def _p_to_star(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metrics=None,
    predictions_df=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    frame = predictions_df if predictions_df is not None else df
    if frame is None or frame.empty or "metric" not in frame.columns:
        return None
    test = frame[frame.get("split", "test") == "test"] if "split" in frame else frame
    if metrics is None:
        metrics = [m for m in ("psnr", "ssim", "nmse") if m in set(test["metric"])] or list(
            test["metric"].unique()[:3]
        )
    metrics = [m for m in metrics if m in set(test["metric"])]
    if not metrics:
        return None
    use_default_style()
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(column_width("double"), column_width("single") * 0.8),
        squeeze=False,
    )
    for j, metric in enumerate(metrics):
        ax = axes[0][j]
        sub = test[test["metric"] == metric]
        methods = list(sub["method"].unique())
        data = [sub[sub["method"] == m]["value"].dropna().to_numpy() for m in methods]
        parts = ax.violinplot(data, showextrema=False)
        for k, b in enumerate(parts["bodies"]):
            b.set_facecolor(colour_for(methods[k], k))
            b.set_alpha(0.25)
        ax.boxplot(data, widths=0.15, showfliers=False)
        for k, arr in enumerate(data):
            # seeded local RNG → reproducible jitter, never touches the global RNG
            jitter = np.random.default_rng(k).uniform(-0.05, 0.05, arr.shape)
            x = np.full(arr.shape, k + 1) + jitter
            ax.scatter(x, arr, s=2, color=colour_for(methods[k], k), alpha=0.5)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(pretty_label(metric))
        sparse_ticks(ax)
        if len(metrics) > 1:
            panel_label(ax, chr(ord("a") + j))
        # significance bracket between best two methods (by mean)
        if len(methods) >= 2:
            means = [np.nanmean(d) if d.size else np.nan for d in data]
            top = np.argsort(means)[-2:]
            a, b = data[top[0]], data[top[1]]
            n = min(a.size, b.size)
            if n > 1:
                p = _stats.paired_bootstrap_pvalue(a[:n], b[:n], seed=0)
                p = _stats.holm_bonferroni([p])[0]
                y = max(np.nanmax(a), np.nanmax(b)) * 1.02
                x1, x2 = top[0] + 1, top[1] + 1
                ax.plot([x1, x1, x2, x2], [y, y * 1.01, y * 1.01, y], lw=0.6, color="#333333")
                ax.text(
                    (x1 + x2) / 2, y * 1.01, _p_to_star(p), ha="center", va="bottom", fontsize=6
                )
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
