"""Step-wise series for the training-dynamics figures.

Reads the tidy frame every run and comparison already carries (``split``, ``step``,
``metric``, ``value``, ``method``), so a figure draws one run or many arms unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.legend import (
    arm_palette,
    reserve_legend_space,
    shared_legend,
)
from spectramr.infrastructure.reporting.style import save_plot

_NEEDED = {"split", "metric", "value"}


def series(df: pd.DataFrame | None, split: str, metric: str) -> pd.DataFrame:
    """``method, step, value`` for ``metric`` in ``split``, averaged over repeated rows."""
    if df is None or df.empty or not set(df.columns) >= _NEEDED:
        return pd.DataFrame(columns=["method", "step", "value"])
    rows = df[(df["split"] == split) & (df["metric"] == metric)]
    if rows.empty:
        return pd.DataFrame(columns=["method", "step", "value"])
    step = "step" if "step" in rows and rows["step"].notna().any() else "iteration"
    method = rows["method"] if "method" in rows else pd.Series("run", index=rows.index)
    out = pd.DataFrame(
        {
            "method": method.astype(str),
            "step": pd.to_numeric(rows[step], errors="coerce"),
            "value": pd.to_numeric(rows["value"], errors="coerce"),
        }
    ).dropna()
    return out.groupby(["method", "step"], as_index=False)["value"].mean().sort_values("step")


def has(df: pd.DataFrame | None, split: str, metric: str) -> bool:
    return not series(df, split, metric).empty


def methods(*frames: pd.DataFrame) -> list[str]:
    seen: dict[str, None] = {}
    for f in frames:
        for m in f["method"]:
            seen.setdefault(str(m), None)
    return list(seen)


def lines(ax, frame: pd.DataFrame, colours: dict[str, str], **kw) -> None:
    for m, part in frame.groupby("method", sort=False):
        ax.plot(part["step"], part["value"], color=colours[str(m)], lw=1.2, label=str(m), **kw)


def finish(fig, out_path: str | Path, formats, dpi, metadata, names: list[str]) -> Path | None:
    if len(names) > 1:
        colours = arm_palette(names)
        handles = [plt.Line2D([], [], color=colours[n], lw=1.5) for n in names]
        shared_legend(fig, handles, names)
    fig.tight_layout(rect=(0, reserve_legend_space(fig) if len(names) > 1 else 0, 1, 1))
    primary = save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
    plt.close(fig)
    return primary
