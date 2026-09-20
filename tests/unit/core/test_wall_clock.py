"""Unit tests for :mod:`spectramr.core.wall_clock`.

Every malformed-budget case here is a planted violation: each one produces a run
that trains normally and then dies at the wall with nothing saved, which is
indistinguishable from a crash unless the resolver refuses it up front
(non-negotiable 15).
"""

from __future__ import annotations

import time

import pytest

from spectramr.core.env_names import (
    SPECTRAMR_WALL_CLOCK_DEADLINE,
    SPECTRAMR_WALL_CLOCK_MARGIN_S,
)
from spectramr.core.wall_clock import (
    DEFAULT_MARGIN_S,
    WallClockBudget,
    WallClockBudgetError,
    resolve_wall_clock_budget,
)


@pytest.fixture(autouse=True)
def _clear_budget_env(monkeypatch):
    monkeypatch.delenv(SPECTRAMR_WALL_CLOCK_DEADLINE, raising=False)
    monkeypatch.delenv(SPECTRAMR_WALL_CLOCK_MARGIN_S, raising=False)


@pytest.mark.unit
def test_no_declaration_means_no_budget():
    """The normal case: an interactive run trains to max_iterations."""
    assert resolve_wall_clock_budget() is None


@pytest.mark.unit
def test_blank_declaration_is_treated_as_absent(monkeypatch):
    """An sbatch that could not derive a deadline exports an empty string."""
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, "   ")
    assert resolve_wall_clock_budget() is None


@pytest.mark.unit
def test_declared_deadline_yields_a_margin_early(monkeypatch):
    deadline = time.time() + 10_000
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, str(deadline))

    budget = resolve_wall_clock_budget()

    assert budget is not None
    assert budget.margin_s == DEFAULT_MARGIN_S
    assert budget.yield_at == pytest.approx(deadline - DEFAULT_MARGIN_S)
    assert not budget.expired()


@pytest.mark.unit
def test_declared_margin_overrides_the_default(monkeypatch):
    deadline = time.time() + 10_000
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, str(deadline))
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARGIN_S, "1800")

    assert resolve_wall_clock_budget().margin_s == 1800.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["soon", "12:00:00", "1e", "--5"])
def test_unparseable_deadline_raises(monkeypatch, bad):
    """A deadline that degraded to None is a job that dies at the wall."""
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, bad)
    with pytest.raises(WallClockBudgetError, match="not a number"):
        resolve_wall_clock_budget()


@pytest.mark.unit
def test_unparseable_margin_raises(monkeypatch):
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, str(time.time() + 10_000))
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARGIN_S, "15m")
    with pytest.raises(WallClockBudgetError, match="not a number"):
        resolve_wall_clock_budget()


@pytest.mark.unit
@pytest.mark.parametrize("margin", ["0", "-60"])
def test_non_positive_margin_raises(monkeypatch, margin):
    """A zero margin means the job is killed mid-save, holding a partial file."""
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, str(time.time() + 10_000))
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARGIN_S, margin)
    with pytest.raises(WallClockBudgetError, match="must be positive"):
        resolve_wall_clock_budget()


@pytest.mark.unit
def test_allocation_already_inside_the_margin_raises(monkeypatch):
    """The infinite-requeue shape: yield at step 1, requeue, repeat forever."""
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, str(time.time() + 60))
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARGIN_S, "900")
    with pytest.raises(WallClockBudgetError, match="requeue forever"):
        resolve_wall_clock_budget()


@pytest.mark.unit
def test_a_deadline_in_the_past_raises(monkeypatch):
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_DEADLINE, str(time.time() - 5))
    with pytest.raises(WallClockBudgetError, match="requeue forever"):
        resolve_wall_clock_budget()


@pytest.mark.unit
def test_expiry_is_evaluated_against_the_supplied_clock():
    """`now` is injectable so the loop's check is testable without sleeping."""
    budget = WallClockBudget(deadline=1_000_000.0, margin_s=900.0)

    assert not budget.expired(now=999_000.0)
    assert budget.expired(now=999_100.0)
    assert budget.seconds_remaining(now=999_000.0) == pytest.approx(100.0)
