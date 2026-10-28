r"""Figure 1.18 — Train / validation generalization gap.

For every metric logged on both splits (``ssim`` / ``train_ssim`` matched to
``val_ssim`` by base name), overlay the two curves — train solid, validation
dashed — and shade the band between them. A widening band is the overfitting
tell; a validation curve *below* its training twin from step one usually
means the splits are not exchangeable (leakage or site shift).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from spectramr.infrastructure.reporting.legend import reserve_legend_space, shared_legend, sparse_ticks
from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)

# preferred panel order — headline quality metrics first, losses last
_PREFERRED = ("psnr", "ssim", "nmse", "mae", "hfen", "loss")


def _series_by_base(sub: pd.DataFrame, drop_prefix: str, skip_prefix: str) -> dict[str, pd.Series]:
    """Collapse a split's rows into ``{base_metric: step-indexed series}``."""
    out: dict[str, pd.Series] = {}
    for metric in sub["metric"].unique():
        low = str(metric).lower()
        if skip_prefix and low.startswith(skip_prefix):
            continue  # e.g. val_* columns duplicated into the training CSV
        base = low.removeprefix(drop_prefix)
        series = (
            sub[sub["metric"] == metric]
            .dropna(subset=["value"])
            .groupby("step")["value"]
            .mean()
            .sort_index()
        )
        if len(series) >= 2 and base not in out:
            out[base] = series
    return out


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    max_panels: int = 4,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Render the gap figure. ``None`` when no metric exists on both splits."""
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns or "split" not in df.columns:
        return None
    pos = df[df["step"].fillna(-1) >= 0]
    train = _series_by_base(pos[pos["split"] == "train"], "train_", "val_")
    val = _series_by_base(pos[pos["split"] == "val"], "val_", "train_")
    common = set(train) & set(val)
    if not common:
        return None
    ordered = [m for m in _PREFERRED if m in common]
    ordered += sorted(common - set(ordered))
    ordered = ordered[:max_panels]

    use_default_style()
    n = len(ordered)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.0), squeeze=False)
    for i, base in enumerate(ordered):
        ax = axes[0][i]
        t, v = train[base], val[base]
        colour = colour_for(base, fallback_index=i)
        ax.plot(t.index, t.values, color=colour, linewidth=1.0, label="train")
        ax.plot(
            v.index,
            v.values,
            color=colour,
            linewidth=1.0,
            linestyle="--",
            marker="o",
            markersize=2,
            label="validation",
        )
        # shade the gap on the overlapping step range (linear interpolation
        # aligns the sparser validation grid onto the union of steps)
        union = t.index.union(v.index)
        ti = t.reindex(union).interpolate(method="index", limit_area="inside")
        vi = v.reindex(union).interpolate(method="index", limit_area="inside")
        both = ti.notna() & vi.notna()
        if both.sum() >= 2:
            ax.fill_between(union[both], ti[both], vi[both], color=colour, alpha=0.15, linewidth=0)
        ax.set_title(pretty_label(base))
        ax.set_xlabel(pretty_label("step"))
        if n > 1:
            panel_label(ax, chr(ord("a") + i))
        sparse_ticks(ax)
    # colour names the metric; line style names the split
    shared_legend(
        fig,
        [Line2D([], [], color="#555555", lw=1.0), Line2D([], [], color="#555555", lw=1.0, ls="--")],
        ["train", "validation"],
    )
    fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 1))
    if metadata is not None:
        stamp_figure(fig, metadata)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
