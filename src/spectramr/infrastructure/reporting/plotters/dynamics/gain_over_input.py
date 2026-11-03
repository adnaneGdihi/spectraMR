"""Figure 5.2 — gain over the input: validation metric minus the input's, against step.

The input is what the network receives (zero-filled for undersampling, the noisy single
excitation for n2n), scored by the same metric. Above zero the network adds something;
flat at zero is an identity map; below zero it is worse than doing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _series

PAIRS = (("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("sense_psnr", "SENSE PSNR (dB)"))


def gains(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = _series.series(df, "val", f"val_{metric}")
    inp = _series.series(df, "val", f"val_zf_{metric}")
    if out.empty or inp.empty:
        return pd.DataFrame(columns=["method", "step", "value"])
    both = out.merge(inp, on=["method", "step"], suffixes=("", "_in"))
    return both.assign(value=both["value"] - both["value_in"])[["method", "step", "value"]]


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    panels = [(label, g) for metric, label in PAIRS if not (g := gains(df, metric)).empty][:2]
    if not panels:
        return None
    names = _series.methods(*(g for _, g in panels))
    colours = arm_palette(names)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double" if len(panels) > 1 else "single")
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, width * (0.36 if len(panels) > 1 else 0.7)), squeeze=False
    )
    for ax, (label, g) in zip(axes[0], panels, strict=True):
        ax.axhline(0, color="#6A7382", lw=0.8, ls="--")
        _series.lines(ax, g, colours, marker="o", ms=2)
        ax.set_ylabel(f"Δ {label}, output - input")
        ax.set_xlabel("training step")
        sparse_ticks(ax, axes="x")
    return _series.finish(fig, out_path, formats, dpi, metadata, names)
