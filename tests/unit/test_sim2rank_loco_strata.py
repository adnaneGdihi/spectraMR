"""LOCO validates the per-contrast composite orders, and only when two contrasts exist."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scripts.sim2rank")

from scripts.sim2rank.loco_strata import (
    LOCO_BASIS,
    NOT_APPLICABLE,
    contrast_loco,
    ranking_validated,
    should_fail_strict_validation,
)
from spectramr.core.metrics.meta_evaluation.banded_composite import parse_composite_weights

W = parse_composite_weights(None)


def _scores(orders: list[list[float]]) -> np.ndarray:
    return np.asarray(orders, float)


def test_one_contrast_is_not_applicable() -> None:
    tau = _scores([[1.0, 0.5, 0.2]])
    res, kind = contrast_loco(("T1",), tau, tau, tau, weights=W, circular=np.zeros(3, bool))
    assert res is None and kind == NOT_APPLICABLE
    assert ranking_validated(None, kind) == NOT_APPLICABLE
    assert should_fail_strict_validation(True, None, kind) is True
    assert should_fail_strict_validation(False, None, kind) is False


def test_agreeing_contrasts_pass_and_opposite_ones_fail() -> None:
    same = _scores([[1.0, 0.6, 0.3, 0.1], [0.9, 0.5, 0.2, 0.0], [0.8, 0.7, 0.1, 0.05]])
    res, kind = contrast_loco(("T1", "T2", "FLAIR"), same, same, same, weights=W,
                              circular=np.zeros(4, bool))  # fmt: skip
    assert kind == "per_contrast" and res.passed and res.kendall_w == pytest.approx(1.0)
    assert res.n_folds == 3 and res.n_metrics == 4 and res.axiom_results == {}
    opposite = _scores([[1.0, 0.6, 0.3, 0.1], [0.1, 0.3, 0.6, 1.0]])
    res, _ = contrast_loco(("T1", "T2"), opposite, opposite, opposite, weights=W,
                           circular=np.zeros(4, bool))  # fmt: skip
    assert not res.passed and res.kendall_w == pytest.approx(0.0)


def test_circular_metrics_are_not_folded() -> None:
    tau = _scores([[1.0, 0.6, 0.3], [0.9, 0.5, 0.2]])
    res, _ = contrast_loco(("T1", "T2"), tau, tau, tau, weights=W,
                           circular=np.array([True, False, False]))  # fmt: skip
    assert res.n_metrics == 2
    assert all(np.isnan(r[0]) for r in res.per_fold_rankings.values())


def test_strict_gate_keeps_its_old_cases() -> None:
    assert should_fail_strict_validation(True, SimpleNamespace(passed=False)) is True
    assert should_fail_strict_validation(True, SimpleNamespace(passed=True)) is False
    assert should_fail_strict_validation(True, None) is False  # --no-validate-ranking


def test_the_basis_is_the_per_contrast_composite_and_the_old_basis_is_gone() -> None:
    from scripts.sim2rank import loco_strata
    from scripts.sim2rank import sim2rank as mod

    assert LOCO_BASIS == "per_contrast_composite"
    assert not hasattr(loco_strata, "build_loco_strata")
    src = inspect.getsource(mod.run_sim2rank)
    assert "contrast_loco(" in src and "run_loco_validation" not in src
    assert '"loco_basis": LOCO_BASIS' in src
    assert "ranking_validated(loco_result, loco_strata)" in src


def test_the_score_polarity_loco_and_its_controls_are_deleted() -> None:
    # run_loco_validation stored higher-is-better scores where the plot reads ranks;
    # contrast_loco is the one LOCO owner (rule 17).
    from scripts.sim2rank import axiomatic_validation

    for gone in ("run_loco_validation", "inject_control_metrics", "validate_axioms", "ORACLE"):
        assert not hasattr(axiomatic_validation, gone), gone


def test_run_sim2rank_keeps_no_dead_jacobian_sparsity() -> None:
    from scripts.sim2rank import sim2rank as mod

    assert "sparsity_scores" not in inspect.getsource(mod.run_sim2rank)
