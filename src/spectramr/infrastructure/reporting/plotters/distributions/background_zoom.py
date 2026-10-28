"""Figure 3.9 — the background, stretched to the noise (MRIQC's zoomed-background reportlet).

Prediction and target windowed to a few background standard deviations, so ghosts, ringing
and any structure leaking into the background become visible. The image-quality numbers
(EFC, FBER, SNR, background sigma) go to ``<figure>.iqm.csv``, not onto the canvas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.metrics.iqm import foreground_mask, iqms
from spectramr.infrastructure.reporting.plotters.distributions import _frame
from spectramr.infrastructure.reporting.style import column_width, use_default_style

_STRETCH_SIGMA = 4.0


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: list[dict] | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not cases:
        return None
    case = cases[0]
    images = {k: np.asarray(case.get(k), dtype=np.float64) for k in ("prediction", "target")}
    if any(v.ndim != 2 for v in images.values()):
        return None
    mask = foreground_mask(images["target"])
    if mask.all() or not mask.any():
        return None
    rows = [{"image": k, **iqms(v, mask)} for k, v in images.items()]
    sigma = rows[1]["bg_sigma"]
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    use_default_style()
    import matplotlib.pyplot as plt

    width = column_width("double")
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.5), squeeze=False)
    for ax, (name, img) in zip(axes[0], images.items(), strict=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=_STRETCH_SIGMA * sigma, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    primary = _frame.finish(fig, out_path, formats, dpi, metadata)
    if primary is not None:
        pd.DataFrame(rows).to_csv(Path(primary).with_suffix(".iqm.csv"), index=False)
    return primary
