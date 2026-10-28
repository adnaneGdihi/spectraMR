r"""Does the ranking of the arms depend on the metric?

Arms are ranked on each metric (1 = best, by the metric's declared direction) and
joined by a line across metrics, so a crossing is a disagreement. A metric whose
direction the SSOT does not declare is left out and named in the note: ranking it
would assume a direction. Ranks of tied values share the smaller rank.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..metadata import RunMetadata, stamp_figure, write_sidecar
from ..style import colour_for, column_width, pretty_label, save_figure, use_default_style
from .metric_selection import _ordered

MAX_METRICS = 8


def rank_table(table: pd.DataFrame, phase: str, metrics: list[str] | None = None) -> pd.DataFrame:
    """Arms by metrics, the rank of each arm (1 = best) on every metric with a direction."""
    one = table[(table["phase"] == phase) & table["mean"].notna()]
    wide = one.pivot(index="arm", columns="metric", values="mean")
    direction = one.drop_duplicates("metric").set_index("metric")["direction"]
    keep = [m for m in (metrics or _ordered(set(wide.columns))) if m in wide.columns]
    ranks = {}
    for metric in keep:
        word = direction.get(metric, "")
        values = wide[metric].dropna()
        if word and len(values) >= 2:
            ranks[metric] = values.rank(ascending=(word == "lower"), method="min")
    return pd.DataFrame(ranks)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    phase_df: pd.DataFrame | None = None,
    arms: list[str] | None = None,
    phase: str = "val",
    metrics: list[str] | None = None,
    max_metrics: int = MAX_METRICS,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Bump chart of ``phase_df`` at ``phase``; ``None`` below two ranked metrics."""
    if phase_df is None or phase_df.empty or not arms or len(arms) < 2:
        return None
    ranks = rank_table(phase_df, phase, metrics)
    ranks = ranks.iloc[:, :max_metrics]
    if ranks.shape[1] < 2:
        return None
    unranked = [
        m
        for m in dict.fromkeys(phase_df.loc[phase_df["phase"] == phase, "metric"])
        if m not in ranks.columns
    ]
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("double") * 0.7, 0.45 * len(arms) + 1.4))
    for i, arm in enumerate(arms):
        if arm not in ranks.index:
            continue
        ax.plot(range(ranks.shape[1]), ranks.loc[arm], "-o", ms=4, lw=1.2, color=colour_for(arm, i))
        ax.annotate(
            arm,
            (ranks.shape[1] - 1, ranks.loc[arm].iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=6,
            va="center",
        )
    ax.set_xticks(range(ranks.shape[1]))
    ax.set_xticklabels([pretty_label(m) for m in ranks.columns], rotation=30, ha="right")
    ax.set_yticks(range(1, len(arms) + 1))
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    note = f"{phase}: ranks by each metric's declared direction"
    if unranked:
        note += "; no declared direction, not ranked: " + ", ".join(unranked[:6])
    fig.suptitle(note, fontsize="x-small")
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    plt.close(fig)
    path = written.get(formats[0])
    if metadata is not None and path is not None:
        write_sidecar(path, metadata)
    return path


__all__ = ["MAX_METRICS", "make", "rank_table"]
