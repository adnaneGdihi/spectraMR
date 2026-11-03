"""Friedman test and Nemenyi critical difference over a blocks x methods score matrix.

Demšar (2006): rank the methods within each block (subject, physics axis, dataset), test
whether the mean ranks differ (Friedman), and call two methods distinguishable at level
``alpha`` only when their mean ranks differ by more than the Nemenyi critical difference.
The reporting CD figure and Sim2Rank's metric CD diagram both read these functions.
"""

from __future__ import annotations

import numpy as np


def rank_rows(scores: np.ndarray, *, higher_is_better: bool = True) -> np.ndarray:
    """Per-row ranks, 1 = best, ties share their mid-rank; NaN ranks worst.

    A NaN is a method that failed on that block: it is penalised, not dropped, so every
    method keeps the same number of blocks.
    """
    from scipy.stats import rankdata

    signed = np.asarray(scores, dtype=float) * (-1.0 if higher_is_better else 1.0)
    filled = np.where(np.isnan(signed), np.inf, signed)
    return np.vstack([rankdata(row, method="average") for row in filled])


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference ``q_alpha * sqrt(k (k + 1) / (6 n))`` for ``k`` methods, ``n`` blocks."""
    from scipy.stats import studentized_range

    q_alpha = float(studentized_range.ppf(1.0 - alpha, k, np.inf)) / float(np.sqrt(2.0))
    return q_alpha * float(np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_p(ranks: np.ndarray) -> float:
    """Tie-corrected Friedman p-value from a blocks x methods rank matrix (NaN below 3 methods)."""
    from scipy.stats import friedmanchisquare

    n, k = ranks.shape
    if k < 3 or n < 2 or np.allclose(ranks, ranks[:, :1]):
        return float("nan")
    return float(friedmanchisquare(*ranks.T).pvalue)


def cliques(mean_rank: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal runs of consecutive methods (sorted by mean rank) within ``cd`` of each other."""
    k = len(mean_rank)
    runs: list[tuple[int, int]] = []
    for j in range(k):
        hi = j
        while hi + 1 < k and (mean_rank[hi + 1] - mean_rank[j]) <= cd:
            hi += 1
        if hi > j:
            runs.append((j, hi))
    kept: list[tuple[int, int]] = []
    for lo, hi in runs:
        if not any(lo >= a and hi <= b for a, b in kept):
            kept.append((lo, hi))
    return kept


__all__ = ["cliques", "friedman_p", "nemenyi_cd", "rank_rows"]
