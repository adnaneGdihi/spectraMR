"""Kendall τ-b pooled over strata: concordance counted within a stratum, summed across them.

A pair of samples is compared only inside its stratum (one content, or one content in one
cell), so contents never trade pairs with each other. Summing the pair counts weights each
stratum by how many pairs it has, instead of averaging per-stratum τ values (Fisher-z is
undefined at the ubiquitous τ = ±1, and a 3-level τ is as noisy as it is coarse).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = [
    "CIRCULARITY_THRESHOLD",
    "PooledTau",
    "measured_circular",
    "tau_b_rows",
    "tau_parts",
    "within_content_tau",
]

#: |pooled within-content τ| with the yardstick at or above which a metric is circular.
CIRCULARITY_THRESHOLD: float = 0.98

#: Upper bound on rows x pairs held in one sign matrix (float64: about 128 MB).
_CHUNK_ELEMS = 16_000_000
#: Above this many pairs per row, the O(n log n) sort path beats the O(n^2) sign matrix.
_SORT_PATH_PAIRS = 2_000


def _untied_pairs(v: np.ndarray) -> float:
    """Pairs of ``v`` whose two values differ."""
    _, counts = np.unique(v, return_counts=True)
    n = v.size
    return float(n * (n - 1) // 2 - int((counts * (counts - 1) // 2).sum()))


def _tau_parts_sorted(x: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy.stats import kendalltau

    nd = _untied_pairs(d)
    nx = np.array([_untied_pairs(row) for row in x])
    if nd == 0:
        return np.zeros(x.shape[0]), nx, nd
    tau = np.array(
        [kendalltau(row, d).statistic if n > 0 else 0.0 for row, n in zip(x, nx, strict=True)]
    )
    s = np.rint(np.nan_to_num(tau) * np.sqrt(nx * nd))
    return s, nx, nd


def tau_parts(x: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``(S[M], nx[M], nd)`` of every row of ``x`` [M, n] against ``d`` [n].

    ``S = Σ sign(Δx)·sign(Δd)`` over the ``n(n-1)/2`` pairs, ``nx`` and ``nd`` the pairs
    untied in ``x`` and in ``d``; τ-b is ``S / sqrt(nx·nd)``. Short rows use one
    vectorised sign matrix (chunked over rows); long rows recover ``S`` from scipy's
    O(n log n) τ-b, which is exact once rounded since ``S`` is an integer.
    """
    if d.size * (d.size - 1) // 2 > _SORT_PATH_PAIRS:
        return _tau_parts_sorted(x, d)
    i, j = np.triu_indices(d.size, 1)
    sd = np.sign(d[i] - d[j])
    m = x.shape[0]
    s, nx = np.zeros(m), np.zeros(m)
    step = max(1, _CHUNK_ELEMS // max(1, i.size))
    for lo in range(0, m, step):
        sx = np.sign(x[lo : lo + step, i] - x[lo : lo + step, j])
        s[lo : lo + step] = sx @ sd
        nx[lo : lo + step] = np.count_nonzero(sx, axis=1)
    return s, nx, float(np.count_nonzero(sd))


def tau_b_rows(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Kendall τ-b of every row of ``x`` [M, n] against ``d`` [n]; a constant row gives 0.0."""
    s, nx, nd = tau_parts(x, d)
    den = np.sqrt(nx * nd)
    return np.divide(s, den, out=np.zeros(x.shape[0]), where=den > 0)


def _distinct(d: np.ndarray) -> int:
    return int(np.unique(d).size)


class PooledTau:
    """Accumulates pooled τ-b for ``M`` metric rows, one stratum at a time."""

    def __init__(self, m: int) -> None:
        self.s = np.zeros(m)
        self.nx = np.zeros(m)
        self.nd = np.zeros(m)
        self.voted = np.zeros(m, dtype=bool)

    def add(self, x: np.ndarray, d: np.ndarray, min_levels: int) -> None:
        """Add one stratum: ``x`` [M, n] against ``d`` [n].

        A row with non-finite samples is scored on its finite ones only. The stratum votes
        for a row only when the damage it keeps has ``min_levels`` distinct values, so a
        saturated (tied-damage) stratum is uncovered rather than scored 0.
        """
        finite = np.isfinite(x)
        full = finite.all(axis=1)
        if full.any() and _distinct(d) >= min_levels:
            s, nx, nd = tau_parts(x[full], d)
            self._update(np.flatnonzero(full), s, nx, nd)
        for r in np.flatnonzero(~full):
            ok = finite[r]
            if _distinct(d[ok]) >= min_levels:
                s, nx, nd = tau_parts(x[r, ok][None], d[ok])
                self._update(np.array([r]), s, nx, nd)

    def _update(self, rows: np.ndarray, s: np.ndarray, nx: np.ndarray, nd: float) -> None:
        self.s[rows] += s
        self.nx[rows] += nx
        self.nd[rows] += nd
        self.voted[rows] = True

    def tau(self) -> np.ndarray:
        """``[M]`` pooled τ-b; 0.0 where a row is constant or never voted."""
        den = np.sqrt(self.nx * self.nd)
        return np.divide(self.s, den, out=np.zeros(self.s.shape), where=den > 0)


def within_content_tau(
    values: Mapping[str, np.ndarray],
    damage: np.ndarray,
    content: np.ndarray,
    min_levels: int = 2,
) -> dict[str, float]:
    """Per metric, τ-b with ``damage`` pooled within each content over every other sample.

    Every axis of a content is one stratum, so a metric scores ±1 only if it orders the
    damage of *all* that content's degradations, not one degradation at a time.
    """
    names = list(values)
    if not names:
        return {}
    damage = np.asarray(damage, dtype=np.float64)
    content = np.asarray(content)
    stacked = np.stack([np.asarray(values[m], dtype=np.float64) for m in names])
    acc = PooledTau(len(names))
    for c in np.unique(content):
        sel = np.flatnonzero(content == c)
        acc.add(stacked[:, sel], damage[sel], min_levels)
    tau = acc.tau()
    return {m: float(tau[i]) for i, m in enumerate(names) if acc.voted[i]}


def measured_circular(
    values: Mapping[str, np.ndarray],
    damage: np.ndarray,
    axis: np.ndarray,
    content: np.ndarray,
    threshold: float = CIRCULARITY_THRESHOLD,
) -> dict[str, float]:
    """``{metric: pooled within-content τ}`` for metrics with |τ| >= ``threshold``.

    A good metric is monotone within each degradation; a circular one stays monotone in
    the yardstick across all of them. With fewer than two axes the two cannot be told
    apart, so nothing is flagged.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"circularity threshold must be in (0, 1], got {threshold!r}")
    if np.unique(np.asarray(axis)).size < 2:
        return {}
    taus = within_content_tau(values, damage, content)
    return {m: t for m, t in taus.items() if abs(t) >= threshold}
