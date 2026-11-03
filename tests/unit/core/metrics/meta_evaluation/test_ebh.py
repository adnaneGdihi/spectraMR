"""Sign e-values and e-BH."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.betting import betting_wealth
from spectramr.core.metrics.meta_evaluation.ebh import DEFAULT_BETS, e_bh, sign_evalue

pytestmark = pytest.mark.unit


def test_sign_evalue_is_the_mean_betting_wealth_over_the_bet_grid() -> None:
    d = np.array([0.3, -0.1, 2.0, 0.0, np.nan, 1.0])
    s = np.sign(d[np.isfinite(d)])
    expect = np.mean([betting_wealth(s, 0.0, b) for b in DEFAULT_BETS])
    assert sign_evalue(d) == pytest.approx(expect)


def test_sign_evalue_grows_on_consistent_signs_and_shrinks_against_them() -> None:
    assert sign_evalue(np.ones(10)) > 20.0
    assert sign_evalue(-np.ones(10)) < 0.1
    assert sign_evalue([]) == pytest.approx(1.0)


def test_sign_evalue_rejects_bets_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        sign_evalue([1.0], bets=(1.0,))


def test_e_bh_step_up_rule() -> None:
    # m=4, q=0.5: thresholds m/(q k) = 8, 4, 2.67, 2
    assert e_bh([9.0, 0.1, 0.1, 0.1], 0.5).tolist() == [True, False, False, False]
    assert e_bh([3.0, 3.0, 3.0, 3.0], 0.5).all()
    assert not e_bh([1.0, 1.0, 1.0, 1.0], 0.5).any()
    assert e_bh([], 0.1).size == 0


def test_e_bh_validates_inputs() -> None:
    with pytest.raises(ValueError):
        e_bh([1.0], 0.0)
    with pytest.raises(ValueError):
        e_bh([-1.0], 0.1)


def test_e_bh_controls_fdr_on_null_signs() -> None:
    rng = np.random.default_rng(0)
    false = [
        e_bh([sign_evalue(rng.choice([-1.0, 1.0], 12)) for _ in range(50)], 0.1).any()
        for _ in range(200)
    ]
    assert np.mean(false) <= 0.1  # all nulls: FDR equals the any-rejection rate
