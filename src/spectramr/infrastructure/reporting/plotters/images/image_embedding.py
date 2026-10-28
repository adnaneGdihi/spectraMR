"""Figure 4.4 — prediction and target images in a spectral-descriptor space.

Each recorded case contributes two points, its target and its prediction, joined by a line.
A descriptor is the radially averaged log power spectrum (24 bins, from the k-space of the
image through ``fft2c``) plus an 8-bin gradient-magnitude histogram: it sees sharpness and
noise texture, which PSNR does not, and it needs no pretrained network. Projected with t-SNE
(needs at least eight cases, i.e. sixteen points).

The recorded cases come from ``report_cases/`` (written by validation); a run without them
soft-skips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from spectramr.infrastructure.reporting.legend import arm_palette, shared_legend
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.plotters.distributions.embedding import (
    HAVE_SKLEARN,
    tsne_2d,
)
from spectramr.infrastructure.reporting.style import column_width, use_default_style

_SPECTRUM_BINS = 24
_GRADIENT_BINS = 8
_MIN_CASES = 8


def descriptor(image: np.ndarray) -> np.ndarray:
    """Radial log-power spectrum plus a gradient-magnitude histogram of a 2-D image."""
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c

    img = np.asarray(image, dtype=np.float32)
    top = float(np.percentile(np.abs(img), 99.5)) or 1.0
    img = img / top
    power = torch.abs(fft2c(torch.from_numpy(img)[None, None]))[0, 0].numpy() ** 2
    h, w = power.shape
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    edges = np.linspace(0, 1, _SPECTRUM_BINS + 1)
    which = np.clip(np.digitize(radius, edges) - 1, 0, _SPECTRUM_BINS - 1)
    inside = radius <= 1
    sums = np.bincount(which[inside], power[inside], minlength=_SPECTRUM_BINS)
    counts = np.bincount(which[inside], minlength=_SPECTRUM_BINS)
    spectrum = np.log10(sums / np.maximum(counts, 1) + 1e-12)
    gy, gx = np.gradient(img)
    hist, _ = np.histogram(np.hypot(gx, gy), bins=_GRADIENT_BINS, range=(0, 0.5), density=True)
    return np.concatenate([spectrum, hist]).astype(np.float64)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: list[dict] | None = None,
    arm_cases: dict[str, list[dict]] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not HAVE_SKLEARN:
        return None
    groups = arm_cases if arm_cases else ({"": cases} if cases else {})
    rows = []
    for arm, items in groups.items():
        for case in items or []:
            pred, target = case.get("prediction"), case.get("target")
            if (
                pred is None
                or target is None
                or np.ndim(pred) != 2
                or np.shape(pred) != np.shape(target)
            ):
                continue
            rows.append((arm, descriptor(target), descriptor(pred)))
    if len(rows) < _MIN_CASES:
        return None
    matrix = np.stack([d for _, t, p in rows for d in (t, p)])
    spread = matrix.std(axis=0)
    z = (matrix - matrix.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    xy = tsne_2d(z)
    arms = list(dict.fromkeys(a for a, *_ in rows))
    colours = arm_palette([a or "cases" for a in arms])
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, ax = plt.subplots(figsize=(width, width * 0.9))
    for i, (arm, *_) in enumerate(rows):
        t, p = xy[2 * i], xy[2 * i + 1]
        colour = colours[arm or "cases"]
        ax.plot([t[0], p[0]], [t[1], p[1]], color=colour, lw=0.5, alpha=0.6, zorder=1)
        ax.scatter(t[0], t[1], s=14, facecolor="white", edgecolor=colour, linewidths=0.8, zorder=2)
        ax.scatter(p[0], p[1], s=14, color=colour, zorder=3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", ms=5),
        Line2D([], [], marker="o", ls="", color="#555555", ms=5),
    ]
    names = ["target", "prediction"]
    if len(arms) > 1:
        handles = [
            Line2D([], [], marker="s", ls="", color=colours[a or "cases"], ms=5) for a in arms
        ] + handles
        names = [*arms, *names]
    shared_legend(fig, handles, names)
    return _frame.finish(fig, out_path, formats, dpi, metadata)
