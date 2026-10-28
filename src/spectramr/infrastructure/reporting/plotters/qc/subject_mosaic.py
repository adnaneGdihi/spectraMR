r"""Per-subject QC mosaic.

One row per recorded case; columns:

- **Prediction** with a foreground-mask contour overlay (the mask reportlet
  analog — the contour lets you eyeball segmentation/support errors).
- **Target** reference.
- **Error** ``|prediction - target|`` (magma).
- **Background** — the prediction stretched to a low percentile so air-region
  noise, ghosting and wrap show up (the background/air reportlet analog).

Consumes the ``cases`` payload (``input``/``prediction``/``target`` 2-D magnitude
arrays + ``metrics``/``rank``/``case_id``). Returns None when no cases (soft-skip).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    column_width,
    save_figure,
    use_default_style,
)

_MAX_CASES = 6


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _foreground_mask(img: np.ndarray) -> np.ndarray:
    """Cheap foreground mask: above a small fraction of the dynamic range."""
    n = _norm(img)
    thr = max(0.08, float(np.percentile(n, 60)) * 0.25)
    return (n > thr).astype(float)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not cases:
        return None
    cases = list(cases)[:_MAX_CASES]
    use_default_style()
    n = len(cases)
    col_titles = ["Prediction + mask", "Target", "Error", "Background"]
    fig, axes = plt.subplots(n, 4, figsize=(column_width("double"), 1.15 * n + 0.4), squeeze=False)
    for r, case in enumerate(cases):
        pred = np.asarray(case.get("prediction"), dtype=float)
        tgt = np.asarray(case.get("target", pred), dtype=float)
        pred_n, tgt_n = _norm(pred), _norm(tgt)
        err = np.abs(pred - tgt)
        err_n = err / (err.max() + 1e-8)
        # background reportlet: stretch to a low percentile to expose air noise
        bg_hi = max(float(np.percentile(pred_n, 25)), 1e-3)
        # Prediction + foreground contour
        ax0 = axes[r][0]
        ax0.imshow(pred_n, cmap="gray", vmin=0, vmax=1)
        with contextlib.suppress(Exception):  # degenerate mask -> skip contour
            ax0.contour(_foreground_mask(pred), levels=[0.5], colors="#E69F00", linewidths=0.5)
        axes[r][1].imshow(tgt_n, cmap="gray", vmin=0, vmax=1)
        axes[r][2].imshow(err_n, cmap="magma", vmin=0, vmax=1)
        axes[r][3].imshow(pred_n, cmap="magma", vmin=0, vmax=bg_hi)
        for c in range(4):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=6)
        axes[r][0].set_ylabel(str(case.get("rank", case.get("case_id", f"case {r}"))), fontsize=6)
        m = case.get("metrics", {}) or {}
        txt = "  ".join(
            f"{k.upper()} {v:.3g}" for k, v in m.items() if k in ("ssim", "psnr", "nmse")
        )
        if txt:
            axes[r][2].set_xlabel(txt, fontsize=4)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.05, wspace=0.05, hspace=0.22)
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
