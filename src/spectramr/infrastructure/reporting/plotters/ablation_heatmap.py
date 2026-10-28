r"""Two-axis ablation heatmap (knob × knob → metric)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    attach_colorbar,
    column_width,
    save_figure,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    x="axis_x",
    y="axis_y",
    value="value",
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if df is None or df.empty or not {x, y, value} <= set(df.columns):
        return None
    pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single") * 0.9))
    arr = pivot.to_numpy()
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    attach_colorbar(im, ax, value)
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
