"""Figure B7 — Riemannian vs. Euclidean qMap credible regions (Idea 5).

Side-by-side comparison of per-pixel posterior credible regions on a
representative slice. Left: Euclidean baseline (MAE on (M0, T1, T2)).
Right: Riemannian Fisher-metric geodesic distance from the
``geodesic_qmap_error`` metric. The figure expects a dataframe with
columns ``(method, pixel_idx, error)`` plus optional 2-D slice arrays
in the ``slices`` kwarg for the heatmap overlay.

If the slice arrays are absent the plotter falls back to a violin /
histogram comparison so the figure still renders.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import (
    RunMetadata,
    stamp_figure,
    write_sidecar,
)
from spectramr.infrastructure.reporting.style import colour_for, use_default_style, save_plot

logger = logging.getLogger(__name__)


_METHODS = ("euclidean", "riemannian")


def _violins(ax, df: pd.DataFrame) -> None:
    methods = [m for m in _METHODS if m in df["method"].unique()]
    methods += [m for m in df["method"].unique() if m not in methods]
    data = [df[df["method"] == m]["error"].to_numpy() for m in methods]
    if not any(len(d) for d in data):
        ax.set_axis_off()
        return
    parts = ax.violinplot(
        data,
        showmeans=True,
        showmedians=True,
    )
    for body, m in zip(parts["bodies"], methods):
        body.set_facecolor(colour_for(m))
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Per-pixel error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def _heatmaps(axes, slices: dict[str, np.ndarray]) -> None:
    vmax = max((np.nanmax(s) for s in slices.values() if s is not None), default=1.0)
    for ax, method in zip(axes, _METHODS):
        arr = slices.get(method)
        if arr is None:
            ax.set_axis_off()
            continue
        im = ax.imshow(arr, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make(
    df: pd.DataFrame | None,
    out_path: str | Path,
    *,
    slices: dict[str, np.ndarray] | None = None,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Render the Riemannian vs. Euclidean qMap comparison.

    Args:
        df: Long-form dataframe with columns ``method`` and ``error``.
            One row per (method, pixel) pair. Mandatory for the
            violin / hist panel.
        out_path: Destination PDF.
        slices: Optional dict ``{"euclidean": np.ndarray, "riemannian": np.ndarray}``
            of per-pixel error maps for the heatmap overlay.
        metadata: Provenance stamp.

    Returns:
        Output path on success, ``None`` when there is no plottable data.
    """
    out_path = Path(out_path)
    if df is None or df.empty or "method" not in df.columns or "error" not in df.columns:
        logger.info("fig_b7: missing dataframe columns — skipping")
        return None

    use_default_style()
    has_slices = bool(slices)
    if has_slices:
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))
        _heatmaps(axes[:2], slices or {})
        _violins(axes[2], df)
        axes[2].set_title("Per-pixel distribution")
    else:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        _violins(ax, df)
        ax.set_title("Riemannian vs. Euclidean qMap error")

    if metadata is not None:
        stamp_figure(fig, metadata)
    fig.tight_layout()
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)


__all__ = ["make"]
