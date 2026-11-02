"""Sign e-values from the betting martingale, and e-BH false-discovery control.

The e-values come from :func:`betting.betting_wealth`, so this module adds no second
e-value source. Increments are signs in ``{-1, 0, 1}``, whose bounds are fixed in
advance, as betting requires. The null being tested is therefore a majority statement,
``E sign(d) <= 0`` (``d > 0`` is no more likely than ``d < 0``), not ``E d <= 0``. An
average of e-values over a fixed grid of bets is again an e-value.

e-BH (Wang and Ramdas, JRSS-B 84(3), 2022) rejects the ``k*`` largest e-values, where
``k* = max{k : e_(k) >= m / (q k)}``. It controls FDR at ``q`` under arbitrary dependence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .betting import betting_wealth

#: Fixed bet grid. Each wealth uses one bet throughout, so its factors stay nonnegative.
DEFAULT_BETS = (0.25, 0.5, 0.75)


def sign_evalue(diffs: Sequence[float] | np.ndarray, bets: Sequence[float] = DEFAULT_BETS) -> float:
    """e-value against ``E sign(d) <= 0`` from independent differences ``d`` (NaNs dropped)."""
    if not bets or any(not 0.0 < b < 1.0 for b in bets):
        raise ValueError(f"bets must lie in (0, 1), got {tuple(bets)}")
    d = np.asarray(diffs, dtype=np.float64)
    s = np.sign(d[np.isfinite(d)])
    return float(np.mean([betting_wealth(s.tolist(), 0.0, b) for b in bets]))


def e_bh(evalues: Sequence[float] | np.ndarray, q: float) -> np.ndarray:
    """Boolean rejection mask of e-BH at level ``q``."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    e = np.asarray(evalues, dtype=np.float64)
    if np.any(e < 0) or np.any(np.isnan(e)):
        raise ValueError("e-values must be nonnegative and not NaN")
    m = e.size
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    order = np.argsort(-e, kind="stable")
    ranks = np.arange(1, m + 1)
    passing = np.nonzero(e[order] >= m / (q * ranks))[0]
    if passing.size:
        reject[order[: passing.max() + 1]] = True
    return reject


__all__ = ["DEFAULT_BETS", "e_bh", "sign_evalue"]
