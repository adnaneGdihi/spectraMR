r"""Calibration / coverage reliability diagram.

Empirical coverage vs nominal confidence with the ideal y = x diagonal.
Consumes a frame with columns ``nominal``, ``empirical`` (and optional
``method``); when those are absent it returns None (advisory figure).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    column_width,
    save_figure,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if df is None or df.empty or not {"nominal", "empirical"} <= set(df.columns):
        return None
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single")))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls="--", label="ideal")
    methods = df["method"].unique() if "method" in df.columns else [None]
    for i, m in enumerate(methods):
        sub = df if m is None else df[df["method"] == m]
        ax.plot(
            sub["nominal"],
            sub["empirical"],
            marker="o",
            color=colour_for(str(m) if m else "ours", i),
            label=str(m) if m else "ours",
        )
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=5)
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
