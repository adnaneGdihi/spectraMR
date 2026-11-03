"""Gate statistics drop NaNs and never turn missing data into a clean rate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.gate import (
    certifiable,
    concordance,
    equivalence_violations,
    probe_violations,
    violations,
)

pytestmark = pytest.mark.unit
NAN = float("nan")


def test_violations_count_negative_means_and_skip_empty_relations() -> None:
    diffs = [np.array([1.0, 2.0]), np.array([-1.0, -0.5]), np.array([NAN, NAN])]
    point, certified = violations(diffs, 0.1)
    assert point == pytest.approx(0.5) and certified == 0


def test_all_nan_inputs_return_none_not_zero() -> None:
    assert violations([np.array([NAN])], 0.1) == (None, None)
    assert probe_violations([np.array([NAN])], [np.array([1.0])], 0.1) == (None, None)
    assert np.isnan(concordance(torch.full((2, 3, 2), NAN), ((0, 1),)))


def test_violations_certify_with_enough_consistent_contents() -> None:
    point, certified = violations([-np.ones(20)], 0.1)
    assert point == 1.0 and certified == 1


def test_probe_violation_needs_both_members_preferred() -> None:
    a = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    b = [np.array([1.0, 2.0]), np.array([-1.0, -1.0])]
    point, _ = probe_violations(a, b, 0.1)
    assert point == pytest.approx(0.5)


def test_concordance_counts_matched_draws_and_halves_ties() -> None:
    v = torch.tensor([[[2.0, 1.0], [1.0, 1.0]]])  # (C=1, E=2, R=2)
    assert concordance(v, ((0, 1),)) == pytest.approx(0.75)
    assert concordance(v, ((1, 0),)) == pytest.approx(0.25)


def test_certifiable_reflects_the_e_bh_threshold() -> None:
    assert not certifiable(3, 115, 0.1)
    assert certifiable(40, 115, 0.1)
    assert not certifiable(40, 0, 0.1)


def test_equivalence_violations_flag_a_consistent_preference_either_way() -> None:
    steady = np.array([0.5, 0.6, 0.55, 0.52])
    noisy = np.array([0.5, -0.4, 0.1, -0.2])
    point, certified = equivalence_violations([steady, -steady, noisy], 0.1)
    assert point == pytest.approx(2 / 3) and certified == 0
    assert equivalence_violations([np.array([NAN, 1.0])], 0.1) == (None, None)
