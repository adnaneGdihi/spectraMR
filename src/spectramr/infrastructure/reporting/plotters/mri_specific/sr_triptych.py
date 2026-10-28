r"""Figure A.1 — Super-resolution triptych grid.

Per-row layout (5 columns by default, plus an optional 6th radial-spectrum panel):

    LR (bicubic-up)  |  predicted  |  HR  |  residual + colorbar  |  residual histogram  [|  radial spectrum]

Per-image annotations: PSNR / SSIM / L1 (white-on-black, top-left),
making magnitude differences immediately readable.
Residual column uses a *symmetric* RdBu_r colormap with the absolute
range printed via the colorbar; the histogram column shares the same
x-range, plus shows L1 mean, signed bias, and residual σ.
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


def _radial_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.fft.fftshift(np.fft.fft2(image))
    p = np.abs(f) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(p.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    radial = np.bincount(r.ravel(), p.ravel()) / np.bincount(r.ravel()).clip(min=1)
    return np.arange(len(radial)), radial


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: Sequence[dict] | None = None,
    show_spectrum: bool = True,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the SR triptych with residual + histogram + (optional) spectrum.

    Args:
        cases: list of dicts with arrays ``lr``, ``pred``, ``hr`` (all
            same H×W). Length ≥ 3 recommended.
        show_spectrum: Append a 6th column with the radial power spectrum.
    """
    if cases is None or len(cases) == 0:
        return None
    out_path = Path(out_path)

    use_default_style()
    n_rows = len(cases)
    n_cols = 5 + (1 if show_spectrum else 0)
    col_w = 1.7  # inches per column — give colorbar room
    row_h = 1.6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * col_w, n_rows * row_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},
    )
    col_titles = [
        "LR (bicubic-up)",
        "predicted",
        "HR ground-truth",
        "residual (pred − HR)",
        "residual histogram",
    ]
    if show_spectrum:
        col_titles.append("radial spectrum")

    for r, case in enumerate(cases):
        lr = np.asarray(case["lr"]).squeeze().astype(float)
        pred = np.asarray(case["pred"]).squeeze().astype(float)
        hr = np.asarray(case["hr"]).squeeze().astype(float)
        row_label = case.get("name") if r >= 0 else None

        # 1. LR (no metric — it's the input, not a prediction)
        add_image_panel(
            axes[r][0],
            lr,
            title=col_titles[0] if r == 0 else None,
            row_label=row_label if r >= 0 else None,
        )
        # 2. Predicted (annotate vs HR)
        add_image_panel(
            axes[r][1],
            pred,
            title=col_titles[1] if r == 0 else None,
            metric_str=_format_metric_str(pred, hr),
        )
        # 3. HR ground truth
        add_image_panel(
            axes[r][2],
            hr,
            title=col_titles[2] if r == 0 else None,
        )
        # 4. Residual map with colorbar
        abs_max, residual = add_residual_panel(
            axes[r][3],
            pred,
            hr,
            title=col_titles[3] if r == 0 else None,
            fig=fig,
        )
        # 5. Residual histogram with shared x-range
        add_residual_histogram(
            axes[r][4],
            residual,
            abs_max=abs_max,
            title=col_titles[4] if r == 0 else None,
        )

        if show_spectrum:
            ax = axes[r][5]
            for img, label, colour in (
                (lr, "LR", "#888"),
                (pred, "pred", "#0072B2"),
                (hr, "HR", "#000"),
            ):
                k, p = _radial_power_spectrum(img)
                ax.semilogy(k[1:], p[1:], label=label, color=colour, linewidth=0.9)
            if r == 0:
                ax.set_title(col_titles[5], fontsize=8)
                ax.legend(frameon=False, fontsize=6, loc="upper right")
            ax.set_xlabel("|k|", fontsize=7)
            ax.set_ylabel(r"|F|$^2$", fontsize=7)
            ax.tick_params(labelsize=6)

    if metadata is not None:
        stamp_figure(fig, metadata)
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
