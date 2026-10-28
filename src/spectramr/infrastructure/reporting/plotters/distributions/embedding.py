"""Figure 3.7 — embedding of the per-slice metric vectors.

Each slice is a point in z-scored metric space (PSNR, SSIM, NMSE, ...) projected with t-SNE
(``scikit-learn``). Colour is the arm (the contrast for a single run). t-SNE keeps neighbours
together and says nothing about the distance between clusters, so the axes carry no ticks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

logger = logging.getLogger(__name__)

try:
    from sklearn.manifold import TSNE

    HAVE_SKLEARN = True
except ImportError:
    TSNE = None
    HAVE_SKLEARN = False
    logger.info("scikit-learn is not installed; the t-SNE embedding figures are unavailable")

_MAX_POINTS = 4000
_MIN_METRICS = 3


def pca_2d(z: np.ndarray) -> np.ndarray:
    """Project centred ``z`` onto its first two principal axes."""
    u, s, _ = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    return u[:, :2] * s[:2]


def tsne_2d(z: np.ndarray, seed: int = 0) -> np.ndarray:
    """2-D t-SNE of the rows of ``z`` (PCA-initialised, perplexity bounded by the sample size).

    Raises:
        RuntimeError: ``scikit-learn`` is not installed.
        ValueError: fewer than four rows (the perplexity would be undefined).
    """
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for the t-SNE embedding")
    if len(z) < 4:
        raise ValueError("t-SNE needs at least four points")
    perplexity = float(min(30.0, (len(z) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(z)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    slice_df: pd.DataFrame | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    table = _frame.wide(slice_df)
    if table is None or not HAVE_SKLEARN:
        return None
    cols = _frame.embedding_columns(table)
    if len(cols) < _MIN_METRICS:
        return None
    data = table[cols].dropna()
    if len(data) < 10:
        return None
    data = data.sample(min(len(data), _MAX_POINTS), random_state=0)
    z = ((data - data.mean()) / data.std(ddof=0).replace(0, 1)).to_numpy()
    xy = tsne_2d(z)
    methods = _frame.methods_of(data)
    if len(methods) > 1:
        keys = data.index.get_level_values("method").astype(str).to_numpy()
    else:
        keys = np.array([_frame.contrast_of(c) for c in data.index.get_level_values("case_id")])
    labels = list(pd.unique(keys))
    colours = arm_palette(labels)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, width * 0.9))
    for lab in labels:
        sel = keys == lab
        ax.scatter(
            xy[sel, 0],
            xy[sel, 1],
            s=3,
            color=colours[lab],
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if len(labels) > 1:
        _frame.swatch_legend(fig, labels, colours)
    return _frame.finish(fig, out_path, formats, dpi, metadata, legend_rows=len(labels) > 1)
