r"""Two arms' inference results paired on case, for the test phase.

``final_eval.json`` holds ``{metric: {case_id: value}}`` (the case is the input
file's stem). Arms evaluated on the same files pair on that id; each case is one
unit, so the result is reported over ``cases`` where the validation pairing is
over ``subjects``. Reciprocal metrics are compared on ``1/N_eq`` as everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spectramr.core.metrics.reciprocal import is_reciprocal

from ..per_subject_stats import PairedSubjects
from .metric_selection import frame_of


def _values(rows: pd.DataFrame, metric: str) -> tuple[pd.Series, int]:
    one = rows[rows["base_metric"] == metric]
    if "subject_id" not in one.columns or one["subject_id"].isna().all():
        return pd.Series(dtype=float), 0
    values = pd.to_numeric(one.set_index("subject_id")["value"], errors="coerce").astype(float)
    values = values[~values.index.duplicated(keep="last")]
    if is_reciprocal(metric):
        with np.errstate(divide="ignore"):
            values = 1.0 / values
    finite = np.isfinite(values.to_numpy())
    return values[finite], int((~finite).sum())


def pair_on_cases(arms, variant: str, baseline: str, metric: str) -> PairedSubjects | None:
    """Pair ``variant`` with ``baseline`` on shared case ids; ``None`` if either has no cases."""
    test = frame_of(arms, "test")
    va, dropped_a = _values(test[test["method"] == variant], metric)
    vb, dropped_b = _values(test[test["method"] == baseline], metric)
    if va.empty or vb.empty:
        return None
    joined = pd.concat({"a": va, "b": vb}, axis=1, join="inner")
    return PairedSubjects(
        table=joined,
        n_slices=len(joined),
        nonfinite_dropped={"a": dropped_a, "b": dropped_b},
        unpaired_slices={"a": len(va) - len(joined), "b": len(vb) - len(joined)},
        scale="1/N_eq (lower is better)" if is_reciprocal(metric) else "metric",
    )


__all__ = ["pair_on_cases"]
