"""Figure 4.2 — how far each layer group moved from its first checkpoint.

Relative drift ``||w - w0|| / ||w0||`` per layer group over the sampled parameters. One run:
one column per later checkpoint. Several runs: one column per arm (first checkpoint to
``best``, else the last). Colour is the drift; rows are labelled sparsely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style
from spectramr.infrastructure.reporting.weights import (
    HAVE_SAFETENSORS,
    WeightSample,
    checkpoint_files,
    sample_weights,
)

_MAX_ROW_LABELS = 12
_MAX_COLUMNS = 8
_MAX_COLUMN_LABELS = 16


def group_drift(first: WeightSample, later: WeightSample) -> np.ndarray:
    """Relative drift per layer group; NaN for a group with no sampled weight norm."""
    out = np.full(len(first.group_names), np.nan)
    for g in range(len(first.group_names)):
        sel = first.group == g
        base = float(np.linalg.norm(first.values[sel]))
        if base > 0:
            out[g] = float(np.linalg.norm(later.values[sel] - first.values[sel])) / base
    return out


def _columns(weight_runs: dict[str, Path]) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    names: tuple[str, ...] = ()
    cols: dict[str, np.ndarray] = {}
    for arm, run in weight_runs.items():
        files = checkpoint_files(run)
        if len(files) < 2:
            continue
        first = sample_weights(files[0][1])
        if not names:
            names = first.group_names
        if first.group_names != names:
            continue
        later = files[1:] if len(weight_runs) == 1 else files[-1:]
        if len(weight_runs) == 1 and len(later) > _MAX_COLUMNS:
            later = [
                later[i]
                for i in np.unique(np.linspace(0, len(later) - 1, _MAX_COLUMNS).round().astype(int))
            ]
        for step, path in later:
            label = "best" if step is None else str(step)
            cols[arm if len(weight_runs) > 1 else label] = group_drift(first, sample_weights(path))
    return names, cols


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
    names, cols = _columns(weight_runs)
    if not cols or not names:
        return None
    grid = np.stack(list(cols.values()), axis=1)
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("single" if len(cols) <= 4 else "double")
    fig, ax = plt.subplots(figsize=(width, width * min(1.1, max(0.5, 0.045 * len(names) + 0.3))))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest")
    step = max(1, -(-len(names) // _MAX_ROW_LABELS))
    ax.set_yticks(range(0, len(names), step))
    ax.set_yticklabels([names[i] for i in range(0, len(names), step)])
    labels = list(cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels if len(labels) <= _MAX_COLUMN_LABELS else [""] * len(labels),
        rotation=60,
        ha="right",
    )
    ax.set_xlabel("arm" if len(weight_runs) > 1 else "checkpoint, in step order")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("relative drift")
    return _frame.finish(fig, out_path, formats, dpi, metadata)
