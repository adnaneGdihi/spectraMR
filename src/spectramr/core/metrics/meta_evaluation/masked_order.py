"""Order statistics over masked last axes: medians, weighted medians, distinct levels.

The banded scores keep per-content statistics and reduce them under weights (unit weights
for the nominal scores, resampled counts for a bootstrap). These kernels do that without a
Python loop over metrics, cells or contents.
"""

from __future__ import annotations

import numpy as np

__all__ = ["block_end", "distinct_levels", "masked_median", "weighted_median"]

_CHUNK_ELEMS = 8_000_000


def _take(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.take_along_axis(x, np.broadcast_to(idx, (*x.shape[:-1], idx.shape[-1])), -1)


def masked_median(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median over the last axis of the ``valid`` entries; NaN where there are none."""
    s = np.sort(np.where(valid, x, np.nan), axis=-1)
    k = valid.sum(-1)
    lo, hi = np.maximum((k - 1) // 2, 0)[..., None], np.minimum(k // 2, x.shape[-1] - 1)[..., None]
    med = 0.5 * (np.take_along_axis(s, lo, -1) + np.take_along_axis(s, hi, -1))[..., 0]
    return np.where(k > 0, med, np.nan)


def block_end(ds: np.ndarray) -> np.ndarray:
    """``(N, T)``: for each sorted level, the last index of its block of equal values."""
    t = ds.shape[-1]
    last = np.ones(ds.shape, bool)
    last[:, :-1] = ds[:, 1:] != ds[:, :-1]
    idx = np.where(last, np.arange(t), t)
    return np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]


def distinct_levels(mask: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Distinct values among the ``mask``ed sorted levels; ``ends`` is :func:`block_end`."""
    c = np.cumsum(mask, axis=-1)
    later = _take(c, ends) - c
    return (mask & (later == 0)).sum(-1)


def weighted_median(x: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(B, ...)`` median over the last axis of ``x``, each entry counted ``weights[b]`` times.

    Integer weights reproduce ``np.median`` of the repeated sample (the mean of the two
    middle values on an even count); NaN where no valid entry has weight.
    """
    n = x.shape[-1]
    order = np.argsort(np.where(valid, x, np.inf), axis=-1)
    xs = np.take_along_axis(x, order, -1)
    # An invalid entry gathers the appended zero weight; counts stay exact in float32.
    gather = np.where(np.take_along_axis(valid, order, -1), order, n)
    w = np.asarray(weights, np.float32)
    w = np.concatenate([w, np.zeros((w.shape[0], 1), np.float32)], axis=1)
    out = np.empty((w.shape[0], *x.shape[:-1]))
    step = max(1, _CHUNK_ELEMS // max(1, x.size))
    for lo in range(0, w.shape[0], step):
        c = np.cumsum(w[lo : lo + step][:, gather], axis=-1)
        tot = c[..., -1:]
        i1 = np.minimum((c <= np.floor((tot - 1) / 2)).sum(-1), n - 1)[..., None]
        i2 = np.minimum((c <= np.floor(tot / 2)).sum(-1), n - 1)[..., None]
        xb = np.broadcast_to(xs, c.shape)
        med = 0.5 * (np.take_along_axis(xb, i1, -1) + np.take_along_axis(xb, i2, -1))[..., 0]
        out[lo : lo + step] = np.where(tot[..., 0] > 0, med, np.nan)
    return out
