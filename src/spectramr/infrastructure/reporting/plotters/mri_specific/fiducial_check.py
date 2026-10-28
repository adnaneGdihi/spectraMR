r"""Figure A.12 — Virtual-Fiducial preservation check.

For experiments under ``experiments/inprogress/vf/`` (and any other
paradigm that injects a :class:`VirtualFiducial` grid into the training
pipeline), this figure verifies that the model's output still carries
the fiducial pattern at the expected grid locations.

Per row layout (3 panels per case):

    predicted | target | |pred − target|

Each panel has the canonical ``VirtualFiducial`` peak locations outlined
in cyan contour, so the eye immediately sees whether the markers landed
where they were placed at acquisition time. Two PSNR numbers are
annotated per case — full-image PSNR vs ROI-PSNR within the fiducial
mask — so the absolute / relative preservation jumps out without
zooming.

The mask itself is regenerated from the canonical ``VirtualFiducial``
at ``im_size = pred.shape[-2:]``; the pipeline does **not** need to
plumb an extra ``fiducial_mask`` field through ``cases``.

Caveat: this figure makes sense only for VF-paradigm experiments. The
plotter dispatcher soft-fails (logs and returns ``None``) for any
exception — including pre-fiducial paradigms whose cases lack the
canonical 2-D layout — so wiring it into a non-VF YAML's
``reporting.figures`` list won't crash the report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._comparison_helpers import (
    add_image_panel,
    add_residual_panel,
)
from spectramr.infrastructure.reporting.style import use_default_style, save_plot


def _build_fiducial_mask(
    im_size: tuple[int, int],
    grid_spacing: int = 16,
    sigma: float = 2.0,
    mask_threshold: float = 0.25,
) -> np.ndarray:
    """Regenerate the canonical fiducial grid and return a boolean mask.

    Imports :class:`VirtualFiducial` lazily so this plotter has no
    runtime dependency on the physics package unless actually rendered.
    """
    from spectramr.infrastructure.physics.virtual_fiducial import VirtualFiducial

    fiducial = VirtualFiducial(
        im_size=im_size, grid_spacing=grid_spacing, sigma=sigma, learnable=False
    )
    grid = fiducial().detach()
    magnitude = grid.abs().squeeze().cpu().numpy()
    return magnitude > mask_threshold


def _overlay_mask_contour(ax, mask: np.ndarray) -> None:
    """Outline ``True`` regions of ``mask`` in cyan on ``ax``."""
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors=["#00BFFF"],
        linewidths=0.4,
        alpha=0.9,
    )


def _psnr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """PSNR with peak = max(|target|) to match the reporting convention."""
    mse = float(np.mean((pred - target) ** 2))
    if mse < eps:
        return float("inf")
    peak = max(float(np.abs(target).max()), eps)
    return 10.0 * float(np.log10(peak * peak / mse))


def _roi_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """PSNR computed over ``mask == True`` pixels only."""
    if not mask.any():
        return float("nan")
    return _psnr(pred[mask], target[mask])


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: Sequence[dict] | None = None,
    grid_spacing: int = 16,
    sigma: float = 2.0,
    mask_threshold: float = 0.25,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Render the per-case fiducial-preservation grid.

    Args:
        cases: list of dicts with arrays ``pred`` and ``target`` of the
            same ``H × W`` shape. Optional ``name`` field is used as a
            row label.
        grid_spacing: Pixel spacing between fiducial peaks, must match
            the value used during training.
        sigma: Gaussian width of each peak, must match training.
        mask_threshold: Magnitude cutoff used to turn the continuous
            Gaussian grid into a boolean ROI mask.
    """
    if cases is None or len(cases) == 0:
        return None
    out_path = Path(out_path)

    use_default_style()
    n_rows = len(cases)
    n_cols = 3
    col_w = 1.7
    row_h = 1.6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * col_w, n_rows * row_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},
    )
    col_titles = [
        "predicted (cyan = fiducial ROI)",
        "target (cyan = fiducial ROI)",
        "residual (pred − target)",
    ]

    for r, case in enumerate(cases):
        pred = np.asarray(case["pred"]).squeeze().astype(float)
        target = np.asarray(case["target"]).squeeze().astype(float)
        if pred.shape != target.shape or pred.ndim != 2:
            raise ValueError(
                f"fiducial_check expects 2-D pred/target of equal shape; "
                f"got pred={pred.shape}, target={target.shape}"
            )

        mask = _build_fiducial_mask(
            im_size=pred.shape,
            grid_spacing=grid_spacing,
            sigma=sigma,
            mask_threshold=mask_threshold,
        )
        full_psnr = _psnr(pred, target)
        roi_psnr = _roi_psnr(pred, target, mask)
        metric_str = f"PSNR: {full_psnr:.2f} dB | ROI: {roi_psnr:.2f} dB"
        row_label = case.get("name")

        add_image_panel(
            axes[r][0],
            pred,
            title=col_titles[0] if r == 0 else None,
            row_label=row_label,
            metric_str=metric_str,
        )
        _overlay_mask_contour(axes[r][0], mask)

        add_image_panel(
            axes[r][1],
            target,
            title=col_titles[1] if r == 0 else None,
        )
        _overlay_mask_contour(axes[r][1], mask)

        add_residual_panel(
            axes[r][2],
            pred,
            target,
            title=col_titles[2] if r == 0 else None,
            fig=fig,
        )

    if metadata is not None:
        stamp_figure(fig, metadata)
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
