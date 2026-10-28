"""Figure 4.1 — the optimisation trajectory in weight space.

Each checkpoint is a point (a fixed subset of its parameters, projected onto the first two
principal axes of all the checkpoints drawn); the points of a run are joined in step order
and ``best`` is the star. One run is coloured by step, several runs by arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from spectramr.infrastructure.reporting.legend import arm_palette, shared_legend
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.plotters.distributions.embedding import pca_2d
from spectramr.infrastructure.reporting.style import column_width, use_default_style
from spectramr.infrastructure.reporting.weights import (
    HAVE_SAFETENSORS,
    checkpoint_files,
    sample_weights,
)

_WIDE_AFTER = 4


def read_series(weight_runs: dict[str, Path]) -> dict[str, list[tuple[int | None, np.ndarray]]]:
    """``{arm: [(step, sampled weights)]}`` for the arms sharing the most common vector length."""
    series = {}
    for arm, run in weight_runs.items():
        files = checkpoint_files(run)
        if len(files) >= 2:
            series[str(arm)] = [(step, sample_weights(path).values) for step, path in files]
    lengths = [len(v[0][1]) for v in series.values()]
    if not lengths:
        return {}
    common = max(set(lengths), key=lengths.count)
    return {arm: s for arm, s in series.items() if len(s[0][1]) == common}


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    weight_runs: dict[str, Path] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not weight_runs or not HAVE_SAFETENSORS:
        return None
    series = read_series(weight_runs)
    points = [(arm, step, vec) for arm, s in series.items() for step, vec in s]
    if len(points) < 3:
        return None
    xy = pca_2d(np.stack([vec for *_, vec in points]))
    use_default_style()
    import matplotlib.pyplot as plt

    arms = list(series)
    wide = len(arms) > _WIDE_AFTER
    width = column_width("double" if wide else "single")
    fig, ax = plt.subplots(figsize=(width, width * (0.45 if wide else 0.9)))
    colours = arm_palette(arms)
    cursor = 0
    mappable = None
    for arm in arms:
        n = len(series[arm])
        pts = xy[cursor : cursor + n]
        steps = [s for s, _ in series[arm]]
        cursor += n
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=colours[arm] if len(arms) > 1 else "#999999",
            lw=0.8,
            zorder=1,
        )
        periodic = [i for i, s in enumerate(steps) if s is not None]
        if len(arms) > 1:
            ax.scatter(pts[periodic, 0], pts[periodic, 1], s=9, color=colours[arm], zorder=2)
        else:
            mappable = ax.scatter(
                pts[periodic, 0],
                pts[periodic, 1],
                s=14,
                c=[steps[i] for i in periodic],
                cmap="viridis",
                zorder=2,
            )
        for i, s in enumerate(steps):
            if s is None:
                ax.scatter(
                    pts[i, 0],
                    pts[i, 1],
                    marker="*",
                    s=60,
                    color=colours[arm],
                    edgecolor="black",
                    linewidths=0.4,
                    zorder=3,
                )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, shrink=0.8).set_label("training step")
    handles = [Line2D([], [], marker="*", ls="", color="#555555", markeredgecolor="black", ms=7)]
    if len(arms) > 1:
        _frame.swatch_legend(fig, arms, colours)
    else:
        shared_legend(fig, handles, ["best checkpoint"])
    return _frame.finish(fig, out_path, formats, dpi, metadata)
