"""Paired tests of each method against a baseline, per metric, Holm-corrected."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from spectramr.core.metrics.statistical_tests import StatisticalTests

logger = logging.getLogger(__name__)


def _holm(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    return list(
        StatisticalTests.holm_bonferroni_correction(pvals, alpha=0.05)["corrected_p_values"]
    )


def paired_significance(rows, methods, metrics, baseline_method) -> dict:
    """``{(method, metric): p < 0.05}`` for methods with values paired by subject.

    Rows carry ``(mean, std, values)`` per metric, values keyed by subject. A test
    that cannot run is left out rather than recorded as p = 1.
    """
    sig: dict[tuple[str, str], bool] = {}
    if baseline_method is not None and baseline_method in methods:
        base = next(r for r in rows if r["method"] == baseline_method)
        for metric in metrics:
            base_arr = base[metric][2]
            if not isinstance(base_arr, pd.Series) or base_arr.size < 2:
                continue
            pvals: list[float] = []
            others: list[str] = []
            for r in rows:
                if r["method"] == baseline_method:
                    continue
                arr = r[metric][2]
                if not isinstance(arr, pd.Series) or arr.size < 2:
                    continue
                # one value per subject: repeated rows of a subject are averaged, or
                # the two arrays differ in length and the test cannot pair them
                a_sub = arr.groupby(level=0).mean() if arr.index.has_duplicates else arr
                b_sub = (
                    base_arr.groupby(level=0).mean() if base_arr.index.has_duplicates else base_arr
                )
                shared = a_sub.index.intersection(b_sub.index)
                if len(shared) < 2:
                    continue  # nothing to pair: no test, and the caption says so
                diff = a_sub[shared].to_numpy() - b_sub[shared].to_numpy()
                if np.ptp(diff) == 0:
                    # every subject moves by the same amount: certain unless it is zero
                    pvals.append(0.0 if diff[0] != 0 else 1.0)
                    others.append(r["method"])
                    continue
                try:
                    res = stats.ttest_rel(a_sub[shared].to_numpy(), b_sub[shared].to_numpy())
                    pvals.append(float(res.pvalue))
                    others.append(r["method"])
                except (ValueError, FloatingPointError) as exc:
                    # an untestable pair is left untested, never recorded as p = 1
                    logger.warning(
                        "tab_2_1: no paired test for %s on %s (%s)", r["method"], metric, exc
                    )
            adj = _holm(pvals)
            for i, m in enumerate(others):
                sig[(m, metric)] = adj[i] < 0.05
    return sig
