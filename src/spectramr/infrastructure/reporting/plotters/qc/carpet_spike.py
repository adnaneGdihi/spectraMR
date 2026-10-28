r"""QC carpet + spike diagnostics.

the carpet plot exposes structured (temporal) artefacts as a voxel-by-time
heatmap. For static single-slice reconstruction the analog is a **cases-by-
image-column** carpet of the column-mean reconstruction error: a vertical band
that is bright across many cases flags a systematic spatial artefact (a bad coil
column, residual aliasing, a wrap edge). Alongside it, a **spike strip** gives
each case a scalar spike/ghost score (95th-percentile error over median error)
so outlier cases stand out.

Consumes the ``cases`` payload. Falls back to prediction-intensity columns when a
case has no target. Returns None when no cases (soft-skip).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    column_width,
    save_figure,
    use_default_style,
)

_N_COLS = 96  # resample every case's column profile onto a common width


def _column_profile(case: dict) -> tuple[np.ndarray, float]:
    """Return (column-error profile resampled to _N_COLS, spike score)."""
    pred = np.asarray(case.get("prediction"), dtype=float)
    tgt = case.get("target")
    signal = np.abs(pred - np.asarray(tgt, dtype=float)) if tgt is not None else np.abs(pred)
    col = signal.mean(axis=0)  # mean over rows -> per-column profile
    med = np.median(signal) + 1e-8
    spike = float(np.percentile(signal, 95) / med)
    # resample the per-column profile onto a common grid
    if col.size != _N_COLS:
        x_old = np.linspace(0, 1, col.size)
        x_new = np.linspace(0, 1, _N_COLS)
        col = np.interp(x_new, x_old, col)
    hi = col.max() + 1e-8
    return col / hi, spike


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not cases:
        return None
    cases = list(cases)
    profiles, spikes, labels = [], [], []
    for i, case in enumerate(cases):
        if case.get("prediction") is None:
            continue
        prof, spk = _column_profile(case)
        profiles.append(prof)
        spikes.append(spk)
        labels.append(str(case.get("case_id", case.get("rank", f"case {i}"))))
    if not profiles:
        return None
    carpet = np.vstack(profiles)
    spikes_arr = np.asarray(spikes)

    use_default_style()
    fig, (ax_c, ax_s) = plt.subplots(
        1,
        2,
        figsize=(column_width("double"), 0.35 * len(profiles) + 1.0),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    im = ax_c.imshow(carpet, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax_c.set_title("Column-error carpet", fontsize=7)
    ax_c.set_xlabel("image column (resampled)", fontsize=6)
    ax_c.set_yticks(range(len(labels)))
    ax_c.set_yticklabels(labels, fontsize=4)
    fig.colorbar(im, ax=ax_c, shrink=0.7, label="norm. |error|")

    y = np.arange(len(spikes_arr))
    # flag cases whose spike score is a Tukey outlier
    if spikes_arr.size >= 4:
        q1, q3 = np.percentile(spikes_arr, [25, 75])
        thr = q3 + 1.5 * (q3 - q1)
    else:
        thr = np.inf
    colours = ["#D55E00" if s > thr else "#0072B2" for s in spikes_arr]
    ax_s.barh(y, spikes_arr, color=colours, height=0.7)
    ax_s.set_title("Spike / ghost score", fontsize=7)
    ax_s.set_xlabel("p95 / median error", fontsize=6)
    ax_s.set_yticks(y)
    ax_s.set_yticklabels([])
    ax_s.invert_yaxis()
    ax_c.invert_yaxis()

    fig.suptitle("Carpet + spike diagnostics", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
