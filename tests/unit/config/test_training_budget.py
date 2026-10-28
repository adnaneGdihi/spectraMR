"""``zero_budget_defect``: the one budget rule the registry and the witness share."""

from __future__ import annotations

from spectramr.config.training_budget import NO_TRAINING_MODES, zero_budget_defect


def test_zero_epochs_with_no_iterations_is_the_finding() -> None:
    """Planted violation."""
    message = zero_budget_defect({"epochs": 0})
    assert message is not None and "training.epochs=0" in message


def test_a_positive_iteration_budget_rescues_zero_epochs() -> None:
    assert zero_budget_defect({"epochs": 0, "max_iterations": 100000}) is None


def test_calibration_modes_optimise_nothing() -> None:
    assert {
        "calibration",
        "phys_residual_conformal",
        "equivariance_conformal",
        "validation_badge",
    } == NO_TRAINING_MODES
    for mode in NO_TRAINING_MODES:
        assert zero_budget_defect({"epochs": 0, "max_iterations": 0, "training_mode": mode}) is None


def test_a_missing_epochs_key_is_not_a_zero_budget() -> None:
    assert zero_budget_defect({}) is None
    assert zero_budget_defect(None) is None


def test_a_boolean_or_prose_value_is_not_a_number() -> None:
    assert zero_budget_defect({"epochs": True}) is None
    assert zero_budget_defect({"epochs": "many"}) is None


def test_twin_dps_still_needs_a_budget() -> None:
    """Its strategy inherits the diffusion train step; a zero budget would train nothing and say so."""
    assert "twin_dps" not in NO_TRAINING_MODES
    assert zero_budget_defect({"epochs": 0, "training_mode": "twin_dps"}) is not None


def test_a_bare_strategy_class_alias_names_the_mode_too() -> None:
    """``strategy_class: calibration`` (mrixfields b17) is the same certify-only mode."""
    assert (
        zero_budget_defect({"epochs": 0, "max_iterations": 0, "strategy_class": "calibration"})
        is None
    )
    dotted = "spectramr.infrastructure.training.strategies.calibration.Calibration"
    assert zero_budget_defect({"epochs": 0, "strategy_class": dotted}) is not None
