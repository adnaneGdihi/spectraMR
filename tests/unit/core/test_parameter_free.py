"""The parameter-free declaration: one owner, and both disagreements raise."""

from __future__ import annotations

import pytest
import torch.nn as nn

from spectramr.core.parameter_free import declares_parameter_free, require_parameter_free_run


class _Solver(nn.Module):
    parameter_free = True


class _SolverWithWeights(nn.Module):
    parameter_free = True

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)


class _NetworkThatLostItsLayers(nn.Module):
    """Zero parameters, no declaration: what a failed layer registration looks like."""


def test_declared_solver_is_parameter_free_and_a_network_is_not() -> None:
    assert declares_parameter_free(_Solver())
    assert not declares_parameter_free(nn.Conv2d(2, 2, 1))
    assert not declares_parameter_free(object())


def test_declared_model_holding_parameters_raises() -> None:
    with pytest.raises(ValueError, match="declares parameter_free but holds 6 parameters"):
        declares_parameter_free(_SolverWithWeights())


def test_zero_parameters_without_the_declaration_raises() -> None:
    """Never inferred from the count: an empty network must not be graded weightless."""
    with pytest.raises(ValueError, match="does not declare parameter_free"):
        declares_parameter_free(_NetworkThatLostItsLayers())


def test_run_guard_admits_only_validate_only_without_resume() -> None:
    assert require_parameter_free_run(_Solver(), validate_only=True, resume=None)
    assert not require_parameter_free_run(nn.Linear(2, 2), validate_only=False, resume="auto")
    assert not require_parameter_free_run(None, validate_only=False, resume=None)
    with pytest.raises(ValueError, match="nothing to train"):
        require_parameter_free_run(_Solver(), validate_only=False, resume=None)
    with pytest.raises(ValueError, match="no checkpoint to resume"):
        require_parameter_free_run(_Solver(), validate_only=True, resume="auto")


def test_a_mock_or_an_instance_attribute_is_not_a_declaration() -> None:
    """A MagicMock answers every attribute truthily; only the class's literal True counts."""
    from unittest.mock import MagicMock

    assert not declares_parameter_free(MagicMock())
    assert not require_parameter_free_run(MagicMock(), validate_only=False, resume=None)
    network = nn.Linear(2, 2)
    network.parameter_free = True  # type: ignore[attr-defined]
    assert not declares_parameter_free(network)
