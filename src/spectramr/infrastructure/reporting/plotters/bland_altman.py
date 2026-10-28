r"""Bland–Altman agreement plot.

Mean of (measurement, reference) on x; their difference on y; bias line +
±1.96σ limits of agreement. Used for quantitative-map agreement against a
reference measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    metric=None,
    predictions_df=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    frame = predictions_df if predictions_df is not None else df
    if frame is None or frame.empty or "reference" not in frame.columns:
        return None
    if metric is not None and "metric" in frame.columns:
        frame = frame[frame["metric"] == metric]
    meas = frame["value"].to_numpy(dtype=float)
    ref = frame["reference"].to_numpy(dtype=float)
    mask = ~(np.isnan(meas) | np.isnan(ref))
    meas, ref = meas[mask], ref[mask]
    if meas.size < 2:
        return None
    mean = (meas + ref) / 2.0
    diff = meas - ref
    bias = diff.mean()
    sd = diff.std(ddof=1)
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single") * 0.85))
    ax.scatter(mean, diff, s=6, color=colour_for("ours", 1), alpha=0.7)
    ax.axhline(bias, color="#333333", lw=0.8, label=f"bias {bias:.3g}")
    ax.axhline(
        bias + 1.96 * sd, color="#D55E00", lw=0.7, ls="--", label=f"+1.96σ {bias + 1.96 * sd:.3g}"
    )
    ax.axhline(bias - 1.96 * sd, color="#D55E00", lw=0.7, ls="--")
    ax.set_xlabel("mean of measurement & reference")
    ax.set_ylabel("measurement − reference")
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
