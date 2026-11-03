"""Figure 5.3 — training health: learning rate, loss with spikes, non-finite substitutions.

Stacked panels on a shared step axis. A loss spike is a step above the rolling median by
more than five rolling MADs and by half the median again; the line is the rolling median
and spikes are ringed on it. Panels whose
series the run did not log are left out, and with no loss the figure soft-skips.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.legend import arm_palette, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.style import column_width, use_default_style

from . import _series

LOSSES = ("loss", "g_total_loss", "total_loss")
NONFINITE = "loss_nonfinite_substitutions"


def learning_rate(df: pd.DataFrame) -> pd.DataFrame:
    names = sorted(
        m
        for m in df.get("metric", pd.Series(dtype=str)).astype(str).unique()
        if m == "lr" or m.startswith("lr_")
    )
    return (
        _series.series(df, "train", names[0])
        if names
        else pd.DataFrame(columns=["method", "step", "value"])
    )


def smoothed(frame: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    return frame.assign(
        value=frame.groupby("method")["value"].transform(
            lambda v: v.rolling(window, center=True, min_periods=1).median()
        )
    )


def spikes(part: pd.DataFrame, window: int = 25, k: float = 5.0) -> pd.DataFrame:
    roll = part["value"].rolling(window, center=True, min_periods=5)
    med = roll.median()
    mad = (part["value"] - med).abs().rolling(window, center=True, min_periods=5).median()
    # A spike must clear both the robust band and half again the local level: on a smooth
    # curve the MAD is ~0 and the band alone flags ordinary wiggles.
    return part[((part["value"] - med) > k * mad.clip(lower=1e-12)) & (part["value"] > 1.5 * med)]


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    loss = next((s for m in LOSSES if not (s := _series.series(df, "train", m)).empty), None)
    if loss is None:
        return None
    panels = [
        ("learning rate", learning_rate(df), False),
        ("training loss", loss, True),
        ("non-finite substitutions", _series.series(df, "train", NONFINITE), False),
    ]
    # An all-zero substitution count says nothing a caption cannot; drop the empty panel.
    panels = [p for p in panels if not p[1].empty and (p[2] or p[1]["value"].abs().sum() > 0)]
    names = _series.methods(*(p[1] for p in panels))
    colours = arm_palette(names)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single")
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(width, 0.9 * width * len(panels) / 2 + 0.4),
        squeeze=False,
    )
    for ax, (label, frame, is_loss) in zip(axes[:, 0], panels, strict=True):
        # Per-step loss is noisy; draw its rolling median and ring the raw spikes.
        drawn = smoothed(frame) if is_loss else frame
        _series.lines(ax, drawn, colours)
        ax.set_ylabel(label, fontsize=7)
        if is_loss:
            ax.set_yscale("log")
            for m, part in frame.groupby("method", sort=False):
                hit = spikes(part.reset_index(drop=True))
                ax.scatter(
                    hit["step"],
                    hit["value"],
                    s=14,
                    facecolors="none",
                    edgecolors=colours[str(m)],
                    lw=0.8,
                )
    axes[-1, 0].set_xlabel("training step")
    sparse_ticks(axes[-1, 0], axes="x")
    return _series.finish(fig, out_path, formats, dpi, metadata, names)
