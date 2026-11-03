"""The shared y-limit and flat-response rules used by every severity figure."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.axis_limits import FLAT_RTOL, is_flat, padded_limits


def test_padded_limits_pads_by_span_fraction() -> None:
    assert padded_limits([0.0, 1.0], 0.05) == pytest.approx((-0.05, 1.05))


def test_padded_limits_ignores_nonfinite_and_none_when_empty() -> None:
    assert padded_limits([np.nan, 2.0, np.inf, 4.0], 0.0) == pytest.approx((2.0, 4.0))
    assert padded_limits([np.nan, np.inf]) is None


def test_padded_limits_constant_has_width() -> None:
    lo, hi = padded_limits([3.0, 3.0])
    assert hi - lo == pytest.approx(0.15)


def test_is_flat_floor_and_rtol() -> None:
    assert is_flat(np.array([1.0, 1.0003]), floor=0.001)
    assert not is_flat(np.array([1.0, 1.1]), floor=0.001)
    assert is_flat(np.array([1.0, 1.0 + 1e-5]), floor=0.0, ylim_span=100.0)


def test_default_rtol_is_the_shared_constant() -> None:
    # banded_scores takes its no-floor scale from the same constant.
    assert is_flat(np.array([0.0, 0.9 * FLAT_RTOL]), None, ylim_span=1.0)
    assert not is_flat(np.array([0.0, 1.1 * FLAT_RTOL]), None, ylim_span=1.0)
