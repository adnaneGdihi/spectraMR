"""Figure 5.1 — spectral recovery: validation error per k-space band against training step.

One panel per band (low, mid, high), log y, one line per arm. A network that only learns
the low band shows a falling low-band curve beside flat mid and high ones: the
high-frequency recovery problem, visible while training rather than after it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _series

BANDS = ("low", "mid", "high")
# First family the run logged wins: corrected MSE is the noise-floor-corrected error.
FAMILIES = (("band_corrected_mse", "corrected MSE"), ("band_bias_squared", "bias$^2$"))


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    family = next(
        (
            (stem, label)
            for stem, label in FAMILIES
            if all(_series.has(df, "val", f"val_{stem}_{b}") for b in BANDS)
        ),
        None,
    )
    if family is None:
        return None
    stem, label = family
    frames = {b: _series.series(df, "val", f"val_{stem}_{b}") for b in BANDS}
    frames = {b: f[f["value"] > 0] for b, f in frames.items()}
    names = _series.methods(*frames.values())
    colours = arm_palette(names)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.32))
    for ax, band in zip(axes, BANDS, strict=True):
        _series.lines(ax, frames[band], colours, marker="o", ms=2)
        ax.set_yscale("log")
        ax.set_title(f"{band} band", fontsize=8)
        ax.set_xlabel("training step")
        sparse_ticks(ax, axes="x")
    axes[0].set_ylabel(f"validation {label}")
    return _series.finish(fig, out_path, formats, dpi, metadata, names)
