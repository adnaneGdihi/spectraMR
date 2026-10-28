r"""Contact sheet — a thumbnail montage of every emitted figure (the index)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import column_width, save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    figure_dir=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    out_path = Path(out_path)
    fig_dir = Path(figure_dir) if figure_dir is not None else out_path.parent
    pngs = sorted(p for p in fig_dir.glob("*.png") if p.stem != out_path.stem)
    if not pngs:
        return None
    use_default_style()
    cols = min(3, len(pngs))
    rows = (len(pngs) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(column_width("double"), 2.0 * rows), squeeze=False
    )
    for idx, png in enumerate(pngs):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(mpimg.imread(png))
        ax.set_title(png.stem, fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
    for empty in range(len(pngs), rows * cols):
        axes[empty // cols][empty % cols].set_axis_off()
    fig.suptitle("Figure index", fontsize=8)
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
