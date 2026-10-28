r"""Figure 1.11 — Stratified ("error by regime") performance bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._stats import t_ci_half_width
from spectramr.infrastructure.reporting.style import colour_for, save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    stratified_df: pd.DataFrame | None = None,
    metric: str = "psnr",
    group_col: str = "subgroup",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Bar plot of metric per subgroup with 95 % CI.

    Args:
        stratified_df: Long table with at minimum
            ``[method, group_col, metric, value]``. Aggregator can build
            this from a per-subject final-eval JSON joined with cohort
            metadata.
    """
    if stratified_df is None or stratified_df.empty:
        return None
    sub = stratified_df[stratified_df.get("metric", metric) == metric].copy()
    if sub.empty or group_col not in sub.columns:
        return None

    out_path = Path(out_path)
    use_default_style()

    methods = list(sub["method"].unique()) if "method" in sub.columns else ["overall"]
    groups = list(sub[group_col].unique())

    fig, ax = plt.subplots(figsize=(0.4 * len(groups) + 1.5, 2.6))
    bar_width = 0.8 / max(len(methods), 1)
    x = np.arange(len(groups))

    for i, m in enumerate(methods):
        slot = sub[sub["method"] == m] if "method" in sub.columns else sub
        means: list[float] = []
        ci_half: list[float] = []
        for g in groups:
            v = slot[slot[group_col] == g]["value"].dropna().to_numpy().astype(float)
            if v.size == 0:
                means.append(np.nan)
                ci_half.append(0)
            else:
                means.append(float(v.mean()))
                # Student-t CI half-width — z=1.96 is too narrow for small n
                # (see _stats.t_ci_half_width).
                ci_half.append(t_ci_half_width(v))
        offset = (i - (len(methods) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=ci_half,
            color=colour_for(str(m), fallback_index=i),
            label=str(m),
            alpha=0.85,
            error_kw={"elinewidth": 0.8, "capsize": 2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_xlabel(group_col)
    ax.legend(frameon=False, fontsize=6, loc="best")

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
