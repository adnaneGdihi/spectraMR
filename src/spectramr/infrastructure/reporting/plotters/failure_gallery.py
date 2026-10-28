r"""Figure 1.12 — Failure-case gallery.

A two-column layout per case (target with overlaid |residual| heatmap +
residual histogram), arranged as a grid of the top-K hardest cases.

Each panel is annotated with:
* PSNR / SSIM / L1 versus target (top-left, white-on-black).
* Case name + ranking (top of the histogram column).

Plotter input
-------------
``cases`` — a list of dicts with shape::

    {name, image|input, predicted, target, residual}

where each image value is a numpy array (the report case loader supplies them).
The ``residual`` field can also be a scalar (e.g. mean L1) used to
sort cases.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._comparison_helpers import (
    _format_metric_str,
    add_image_panel,
    add_residual_histogram,
    l1,
)
from spectramr.infrastructure.reporting.style import use_default_style, save_plot


def _load_array(value: Any) -> np.ndarray | None:
    """The case array, squeezed; ``None`` when the case does not carry it.

    Arrays only: opening files is the data layer's job (non-negotiable 7), and
    the case loader already materialises every recorded array. A path raises
    rather than degrading to a missing panel.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.squeeze()
    raise TypeError(
        f"failure_gallery takes arrays, got {type(value).__name__}; load files through "
        "spectramr.data or the report case loader"
    )


def _prediction(case: dict) -> Any:
    """The recorder writes ``prediction``; older callers pass ``predicted``."""
    for key in ("predicted", "prediction", "pred"):
        if case.get(key) is not None:
            return case[key]
    return None


def _sort_key(c: dict) -> float:
    v = c.get("residual", 0.0)
    if isinstance(v, np.ndarray):
        return float(np.abs(v).mean())
    if isinstance(v, (int, float)):
        return float(v)
    pred = _load_array(_prediction(c))
    targ = _load_array(c.get("target"))
    if pred is not None and targ is not None and pred.shape == targ.shape:
        return l1(pred, targ)
    return 0.0


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: Sequence[dict] | None = None,
    n_cases: int = 8,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the failure-case gallery.

    Args:
        cases: list of dicts ``{name, image|input, predicted, target, residual}``.
        n_cases: Number of failure cases to show (sorted by L1 descending).
    """
    if cases is None or len(cases) == 0:
        return None
    out_path = Path(out_path)

    sorted_cases = sorted(cases, key=_sort_key, reverse=True)[:n_cases]

    use_default_style()
    n = len(sorted_cases)
    cols_per_case = 2  # image-with-overlay + residual histogram
    # Cases per row — capped at 4 but never wider than the case count, so a
    # 1-3 case gallery stays compact instead of a mostly-empty ultra-wide canvas
    # (the figure-level provenance stamp otherwise pins bbox_inches to full width).
    grid_cols = min(4, max(1, n))
    n_rows = (n + grid_cols - 1) // grid_cols
    n_cols = grid_cols * cols_per_case

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.4, n_rows * 1.6),
        squeeze=False,
        gridspec_kw={"wspace": 0.06, "hspace": 0.30},
    )

    for idx, case in enumerate(sorted_cases):
        r = idx // grid_cols
        c0 = (idx % grid_cols) * cols_per_case
        c1 = c0 + 1
        ax_img = axes[r][c0]
        ax_hist = axes[r][c1]

        pred = _load_array(_prediction(case))
        targ = _load_array(case.get("target"))
        name = case.get("name") or case.get("case_id", f"case_{idx}")
        rank_str = f"#{idx + 1}  {name}"

        if pred is None and targ is None:
            ax_img.set_axis_off()
            ax_hist.set_axis_off()
            continue

        if pred is None:
            add_image_panel(ax_img, targ, title=rank_str)
            ax_hist.set_axis_off()
            continue

        if targ is None:
            add_image_panel(ax_img, pred, title=rank_str)
            ax_hist.set_axis_off()
            continue

        # Image panel — show target with the |residual| as a hot overlay
        residual = pred - targ
        diff_mag = np.abs(residual)
        ax_img.imshow(targ, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # Overlay the residual hot-spots, scaled to [0, 1] of its own range
        if diff_mag.max() > 1e-12:
            ax_img.imshow(
                diff_mag / diff_mag.max(), cmap="hot", alpha=0.45, interpolation="nearest"
            )
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(rank_str, fontsize=7)
        ax_img.text(
            0.02,
            0.98,
            _format_metric_str(pred, targ),
            transform=ax_img.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="white",
            bbox=dict(
                facecolor="black", edgecolor="none", alpha=0.55, pad=1.2, boxstyle="round,pad=0.2"
            ),
        )

        # Residual histogram with L1, bias, σ
        add_residual_histogram(
            ax_hist,
            residual,
            title="residual hist" if idx < grid_cols else None,
        )

    # Hide any leftover empty axes
    for empty in range(n, n_rows * grid_cols):
        r = empty // grid_cols
        c0 = (empty % grid_cols) * cols_per_case
        axes[r][c0].set_axis_off()
        axes[r][c0 + 1].set_axis_off()

    if metadata is not None:
        stamp_figure(fig, metadata)
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
