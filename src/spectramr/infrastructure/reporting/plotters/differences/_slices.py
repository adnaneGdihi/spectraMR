"""Per-slice tables for the "where arms differ" figures.

``slice_df`` is the long per-slice frame (``method``, ``case_id``, ``base_metric``,
``value``) with ``case_id = subject|contrast|slice|repetition`` (``slice_case_rows``), read
at each arm's selected validation pass. Figures that need subjects or slice positions
soft-skip on rows without that identity.
"""

from __future__ import annotations

import pandas as pd

from spectramr.core.metrics.metric_directions import resolve_direction
from spectramr.infrastructure.reporting.plotters.distributions import _frame

PRIMARY = ("psnr", "ssim", "nmse", "mse")
#: every figure in this package reads the selected validation pass
PHASE_NOTE = "validation, selected pass"


def table(slice_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """``(method, case_id)`` x ``base_metric``; ``None`` without rows."""
    return _frame.wide(slice_df)


def primary(tab: pd.DataFrame, wanted: str | None = None) -> tuple[str, bool] | None:
    """The metric to rank on and whether higher is better; ``None`` when none has a direction."""
    for m in ([wanted] if wanted else []) + [*PRIMARY, *tab.columns]:
        if m in tab.columns and tab[m].notna().any() and (hib := resolve_direction(m)) is not None:
            return m, hib
    return None


def has_identity(tab: pd.DataFrame) -> bool:
    ids = pd.Series(tab.index.get_level_values("case_id").astype(str))
    return bool((ids.str.count(r"\|") >= 2).all())


def subject_of(case_id: str) -> str:
    return str(case_id).split("|")[0]


def slice_of(case_id: str) -> float:
    parts = str(case_id).split("|")
    return pd.to_numeric(parts[2], errors="coerce") if len(parts) > 2 else float("nan")


def subject_means(tab: pd.DataFrame, metric: str) -> pd.DataFrame:
    """subjects x methods of per-subject mean ``metric``, only subjects every arm has."""
    s = tab[metric].dropna().reset_index()
    s["subject"] = s["case_id"].map(subject_of)
    wide = s.groupby(["subject", "method"])[metric].mean().unstack("method")
    return wide.dropna()
