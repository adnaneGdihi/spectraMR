"""Masked order statistics shared by the banded scores (masked_order)."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.masked_order import (
    block_end,
    distinct_levels,
    masked_median,
    weighted_median,
)


def test_weighted_median_matches_numpy_for_unit_and_repeated_weights() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 7))
    valid = np.ones_like(x, bool)
    valid[0, 3] = False
    got = weighted_median(x, valid, np.ones((1, 7)))[0]
    assert got[0] == pytest.approx(np.median(np.delete(x[0], 3)))
    assert got[1:] == pytest.approx(np.median(x[1:], axis=1))
    w = np.array([[2, 0, 1, 1, 3, 0, 1]], float)
    rep = weighted_median(x, np.ones_like(valid), w)[0]
    assert rep == pytest.approx([np.median(np.repeat(r, w[0].astype(int))) for r in x])


def test_weighted_median_is_nan_without_weight_and_skips_invalid_entries() -> None:
    x = np.array([[5.0, 1.0, 3.0, 100.0]])
    valid = np.array([[True, True, True, False]])
    out = weighted_median(x, valid, np.array([[0.0, 0.0, 0.0, 4.0], [1.0, 1.0, 1.0, 9.0]]))
    assert np.isnan(out[0, 0]) and out[1, 0] == 3.0


def test_masked_median_and_distinct_levels() -> None:
    x = np.array([[3.0, 1.0, 2.0, 9.0]])
    assert masked_median(x, np.array([[True, True, True, False]]))[0] == 2.0
    assert np.isnan(masked_median(x, np.zeros_like(x, bool))[0])
    ds = np.array([[0.1, 0.1, 0.2, 0.3]])
    ends = block_end(ds)
    assert ends.tolist() == [[1, 1, 2, 3]]
    assert distinct_levels(np.array([[True, True, True, False]]), ends)[0] == 2
