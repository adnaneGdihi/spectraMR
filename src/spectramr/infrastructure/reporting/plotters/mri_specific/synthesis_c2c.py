r"""Figure A.2 — Contrast-to-contrast synthesis.

Per-row layout (6 columns; 5 when the cases carry no input):

    input | synthesised | real target | residual + colorbar | residual histogram | joint histogram (with MI)

Annotations:
* Synthesised image: PSNR / SSIM / L1 versus real target.
* Residual map: symmetric RdBu_r with colorbar; L1 annotation.
* Residual histogram: L1 mean, signed bias, σ.
* Joint histogram: mutual information annotated.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._comparison_helpers import (
    _format_metric_str,
    add_image_panel,
    add_residual_histogram,
    add_residual_panel,
)
from spectramr.infrastructure.reporting.style import use_default_style, save_plot


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    h, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins)
    p = h / h.sum().clip(min=1)
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nz, p / (px @ py).clip(min=1e-12), 1.0)
        return float((p[nz] * np.log(ratio[nz])).sum())


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: Sequence[dict] | None = None,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the c2c synthesis figure.

    Args:
        cases: dicts with arrays ``pred``, ``target`` (real target) and, when
            recorded, ``input``; the input column is drawn only if every case has
            one. Optional ``name`` for the row label.
    """
    if cases is None or len(cases) == 0:
        return None
    out_path = Path(out_path)

    use_default_style()
    # Cases from saved PNG pairs carry no input, and the loader refuses to invent one.
    has_input = all("input" in case for case in cases)
    col_titles = (["input contrast"] if has_input else []) + [
        "synthesised",
        "real target",
        "residual (synth \u2212 real)",
        "residual histogram",
        "joint histogram",
    ]
    off = len(col_titles) - 5
    n_rows = len(cases)
    n_cols = len(col_titles)
    col_w, row_h = 1.65, 1.6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * col_w, n_rows * row_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},
    )

    for r, case in enumerate(cases):
        pred = np.asarray(case["pred"]).squeeze().astype(float)
        target = np.asarray(case["target"]).squeeze().astype(float)
        row_label = case.get("name")

        if has_input:
            add_image_panel(
                axes[r][0],
                np.asarray(case["input"]).squeeze().astype(float),
                title=col_titles[0] if r == 0 else None,
                row_label=row_label,
            )
        add_image_panel(
            axes[r][off],
            pred,
            title=col_titles[off] if r == 0 else None,
            metric_str=_format_metric_str(pred, target),
            row_label=None if has_input else row_label,
        )
        add_image_panel(
            axes[r][off + 1],
            target,
            title=col_titles[off + 1] if r == 0 else None,
        )
        abs_max, residual = add_residual_panel(
            axes[r][off + 2],
            pred,
            target,
            title=col_titles[off + 2] if r == 0 else None,
            fig=fig,
        )
        add_residual_histogram(
            axes[r][off + 3],
            residual,
            abs_max=abs_max,
            title=col_titles[off + 3] if r == 0 else None,
        )
        # Joint histogram of (target, pred) intensities with MI annotation
        ax = axes[r][off + 4]
        h, _, _ = np.histogram2d(
            target.ravel(),
            pred.ravel(),
            bins=64,
            range=[[0, 1], [0, 1]],
        )
        ax.imshow(
            np.log1p(h.T),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="viridis",
            aspect="equal",
            interpolation="nearest",
        )
        ax.plot([0, 1], [0, 1], "--", color="white", linewidth=0.8, alpha=0.7)
        mi = _mutual_information(target, pred)
        ax.text(
            0.04,
            0.95,
            f"MI={mi:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(
                facecolor="black", edgecolor="none", alpha=0.55, pad=1.5, boxstyle="round,pad=0.2"
            ),
        )
        ax.set_xlabel("real intensity", fontsize=7)
        ax.set_ylabel("synth intensity", fontsize=7)
        ax.tick_params(labelsize=6)
        if r == 0:
            ax.set_title(col_titles[off + 4], fontsize=8)

    if metadata is not None:
        stamp_figure(fig, metadata)
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
