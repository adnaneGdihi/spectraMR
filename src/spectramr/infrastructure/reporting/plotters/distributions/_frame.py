"""Per-slice frames for the distribution, bivariate and embedding figures.

``slice_df`` is the long per-case table (``method``, ``case_id``, ``base_metric``,
``value``, ``acceleration``): ``comparison.slice_cases`` builds it from a run's
``validation_batches.csv`` and ``Arms.per_case`` carries the same schema for several
arms. A ``case_id`` is ``subject|contrast|slice|repetition_group``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from spectramr.infrastructure.reporting.legend import reserve_legend_space, shared_legend
from spectramr.infrastructure.reporting.style import save_plot

PREFERRED: tuple[str, ...] = ("psnr", "ssim", "nmse", "hfen", "mae", "lpips")
_NEEDED = {"method", "case_id", "base_metric", "value"}


def wide(slice_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """``(method, case_id)`` x ``base_metric`` of unaccelerated values; ``None`` without rows."""
    if slice_df is None or slice_df.empty or not set(slice_df.columns) >= _NEEDED:
        return None
    rows = slice_df
    if "acceleration" in rows.columns:
        rows = rows[rows["acceleration"].isna()]
    if rows.empty:
        return None
    return rows.pivot_table(
        index=["method", "case_id"], columns="base_metric", values="value", aggfunc="mean"
    )


def pick_metrics(table: pd.DataFrame, wanted: list[str] | None = None, k: int = 3) -> list[str]:
    """The ``k`` most useful metrics present: ``wanted``, else the preferred order."""
    present = [c for c in table.columns if table[c].notna().any()]
    chosen = [m for m in (wanted or PREFERRED) if m in present]
    return (chosen or present)[:k]


def embedding_columns(table: pd.DataFrame, limit: int = 10) -> list[str]:
    """Metrics worth embedding: well populated and not constant, preferred ones first."""
    ok = [c for c in table.columns if table[c].notna().mean() >= 0.9 and table[c].std() > 0]
    first = [c for c in PREFERRED if c in ok]
    return (first + [c for c in ok if c not in first])[:limit]


def methods_of(table: pd.DataFrame) -> list[str]:
    return [str(m) for m in pd.unique(table.index.get_level_values("method"))]


def contrast_of(case_id: str) -> str:
    parts = str(case_id).split("|")
    return parts[1] if len(parts) > 1 else "all"


def subject_of(case_id: str) -> str:
    return str(case_id).split("|")[0]


def swatch_legend(fig, labels: list[str], colours: dict[str, str]) -> None:
    """One legend of colour swatches, capped by :func:`shared_legend`."""
    shared_legend(fig, [Patch(facecolor=colours[lab], edgecolor="none") for lab in labels], labels)


def finish(fig, out_path, formats, dpi, metadata, *, legend_rows: bool = False) -> Path | None:
    fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 1))
    primary = save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
    plt.close(fig)
    return primary
