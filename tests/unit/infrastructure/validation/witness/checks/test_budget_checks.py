"""Budget witnesses (cohort review 2026-09-02, T0.9). Planted violations first."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness.checks.budget_checks import (
    training_budget_is_positive,
    warmup_shorter_than_budget,
)
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _subject(
    training: dict | None = None,
    scheduler: dict | None = None,
    optimization: dict | None = None,
) -> WitnessSubject:
    raw = {
        "training": training or {},
        "optimization": {"scheduler": scheduler or {}, **(optimization or {})},
    }
    return WitnessSubject.for_ci(None, raw)


# --- warmup_shorter_than_budget ---------------------------------------------------


@pytest.mark.parametrize(("warmup", "budget"), [(2000, 1), (2000, 200), (1000, 1000)])
def test_warmup_covering_the_budget_is_an_error(warmup: int, budget: int) -> None:
    """The planted violations: vf_22 / vf_m2 (2000 vs 1), vf_tto (2000 vs 200)."""
    verdict = warmup_shorter_than_budget(
        _subject({"max_iterations": budget}, {"warmup_steps": warmup})
    )
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert str(warmup) in verdict.message and str(budget) in verdict.message


def test_warmup_below_the_budget_passes() -> None:
    assert warmup_shorter_than_budget(
        _subject({"max_iterations": 70000}, {"warmup_steps": 3500})
    ).passed


@pytest.mark.parametrize(
    "training",
    [{"max_iterations": -1}, {}, {"max_iterations": None}],
)
def test_unbounded_budget_has_nothing_to_compare(training: dict) -> None:
    """``max_iterations: -1`` (experiment_32b) is unbounded, not a one-step run."""
    assert warmup_shorter_than_budget(_subject(training, {"warmup_steps": 1000})).passed


def test_no_warmup_declared_passes() -> None:
    assert warmup_shorter_than_budget(_subject({"max_iterations": 5}, {})).passed


@pytest.mark.parametrize(("warmup", "max_iterations"), [(2000, 4000), (1000, 4000)])
def test_the_budget_is_counted_in_optimizer_steps(warmup: int, max_iterations: int) -> None:
    """Planted violation: 4000 micro-batches at accumulation 4 is 1000 scheduler steps."""
    verdict = warmup_shorter_than_budget(
        _subject(
            {"max_iterations": max_iterations},
            {"warmup_steps": warmup},
            {"gradient": {"accumulation_steps": 4}},
        )
    )
    assert verdict.passed is False and "1000 optimizer steps" in verdict.message


def test_one_step_inside_the_optimizer_budget_passes() -> None:
    assert warmup_shorter_than_budget(
        _subject({"max_iterations": 4000}, {"warmup_steps": 999}, {"gradient": {"accumulation_steps": 4}})
    ).passed


def test_the_flat_warmup_spelling_is_read() -> None:
    """``optimization.warmup_steps`` is the other spelling the scheduler resolver reads."""
    verdict = warmup_shorter_than_budget(
        _subject({"max_iterations": 1000}, {}, {"warmup_steps": 1000})
    )
    assert verdict.passed is False


def test_declared_epochs_do_not_shorten_a_max_iterations_run() -> None:
    """exp_ulf_01 shape: the loop reads epochs only when max_iterations <= 0
    (pipelines/training_loop.py, the ``max_iterations <= 0`` branch), so 500000 // 4 =
    125000 optimizer steps run and a 2000-step warm-up ends inside them."""
    verdict = warmup_shorter_than_budget(
        _subject(
            {"max_iterations": 500000, "epochs": 150},
            {"warmup_steps": 2000},
            {"gradient": {"accumulation_steps": 4}},
        )
    )
    assert verdict.passed is True and "125000" in verdict.message


# --- training_budget_is_positive ----------------------------------------------------


def test_zero_epochs_without_iterations_is_an_error() -> None:
    """The planted violation: seven arms declare ``epochs: 0``."""
    verdict = training_budget_is_positive(_subject({"epochs": 0}))
    assert verdict.passed is False and "epochs=0" in verdict.message


def test_zero_epochs_with_a_positive_iteration_budget_passes() -> None:
    assert training_budget_is_positive(_subject({"epochs": 0, "max_iterations": 1000})).passed


@pytest.mark.parametrize("training", [{"epochs": 1}, {}, {"epochs": None}])
def test_positive_or_undeclared_epochs_pass(training: dict) -> None:
    assert training_budget_is_positive(_subject(training)).passed


def test_both_witnesses_are_registered_after_discovery() -> None:
    import spectramr.infrastructure.validation.witness  # noqa: F401

    registry = get_witness_registry()
    assert registry.get("warmup_shorter_than_budget") is not None
    assert registry.get("training_budget_is_positive") is not None


def test_calibration_mode_may_declare_a_zero_budget() -> None:
    """Split-conformal calibration optimises nothing; ``epochs: 0`` is its contract."""
    verdict = training_budget_is_positive(
        _subject({"epochs": 0, "max_iterations": 0, "training_mode": "calibration"})
    )
    assert verdict.passed is True and "optimises nothing" in verdict.message
