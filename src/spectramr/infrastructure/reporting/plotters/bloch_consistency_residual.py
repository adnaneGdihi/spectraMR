"""Figure B3 — Bloch-equivariance residual per tissue compartment (Idea 3).

Visualises the per-tissue Bloch-equivariance residual

    ‖ B_θ ( F^B_ULF(q) ) − F^B_HF(q) ‖

stratified by tissue class (white matter, grey matter, CSF, fat,
muscle) so the reader can see where the translator is faithful to the
forward model and where it drifts. Input is a dataframe with one row
per (tissue_class, training_step) pair containing the residual
magnitude.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import (
    RunMetadata,
    stamp_figure,
    write_sidecar,
)
from spectramr.infrastructure.reporting.style import colour_for, use_default_style, save_plot

logger = logging.getLogger(__name__)


_TISSUE_ORDER = ["WM", "GM", "CSF", "fat", "muscle"]


def make(
    df: pd.DataFrame | None,
    out_path: str | Path,
    *,
    tissue_column: str = "tissue",
    residual_column: str = "bloch_eq_residual",
    step_column: str = "step",
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Render Bloch-equivariance residual stratified by tissue class.

    Args:
        df: Long-form dataframe with one row per (tissue, step) and a
            scalar residual. The plotter is permissive — missing tissues
            simply do not appear.
        out_path: Destination PDF.
        tissue_column / residual_column / step_column: Column names so
            callers can plug in alternate schemas without renaming.
        metadata: Provenance stamp.

    Returns:
        Output path on success, ``None`` on missing data.
    """
    out_path = Path(out_path)
    if df is None or df.empty:
        logger.info("fig_b3: empty dataframe — skipping")
        return None
    required = {tissue_column, residual_column, step_column}
    if not required <= set(df.columns):
        logger.warning(
            "fig_b3: missing required columns %s; got %s",
            required - set(df.columns),
            list(df.columns),
        )
        return None

    use_default_style()
    fig, (ax_line, ax_box) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left panel: residual vs. step, one line per tissue.
    sub = df[[tissue_column, residual_column, step_column]].dropna()
    tissues_present = [t for t in _TISSUE_ORDER if t in sub[tissue_column].unique()]
    tissues_present += sorted(set(sub[tissue_column].unique()) - set(tissues_present))
    for tissue in tissues_present:
        per_tissue = sub[sub[tissue_column] == tissue].sort_values(step_column)
        if per_tissue.empty:
            continue
        ax_line.plot(
            per_tissue[step_column].to_numpy(),
            per_tissue[residual_column].to_numpy(),
            label=str(tissue),
            color=colour_for(str(tissue)),
            linewidth=1.4,
        )
    ax_line.set_xlabel("Training step")
    ax_line.set_ylabel("‖B(F_ULF(q)) − F_HF(q)‖")
    ax_line.set_title("Residual vs. step")
    ax_line.set_yscale("log")
    ax_line.legend(loc="upper right", fontsize=7, frameon=True)
    ax_line.grid(True, alpha=0.3)

    # Right panel: final-epoch boxplot by tissue.
    final_step = sub[step_column].max()
    final = sub[sub[step_column] == final_step]
    if not final.empty:
        data_box = [
            final[final[tissue_column] == t][residual_column].to_numpy() for t in tissues_present
        ]
        # Matplotlib 3.9+ renamed ``labels`` to ``tick_labels``; we set
        # tick labels manually for forward compatibility with both.
        ax_box.boxplot(
            data_box,
            showmeans=True,
            meanline=True,
        )
        ax_box.set_xticks(range(1, len(tissues_present) + 1))
        ax_box.set_xticklabels(tissues_present)
        ax_box.set_yscale("log")
        ax_box.set_ylabel("Final residual")
        ax_box.set_title(f"Per-tissue distribution at step {int(final_step)}")
        ax_box.grid(True, alpha=0.3, axis="y")
    else:
        ax_box.set_axis_off()

    if metadata is not None:
        stamp_figure(fig, metadata)
    fig.tight_layout()
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)


__all__ = ["make"]
