"""Tests for ``PlateauMonitor``.

Targets ``spectramr.infrastructure.training.plateau_monitor`` — the shared
plateau-detection primitive extracted from ``EarlyStoppingService`` and reused
by ``LossScheduleController``. It owns the strict MIN/MAX comparator, the
``min_delta`` threshold, the ``patience`` wait counter, and a post-fire
``cooldown`` window for repeatable triggering.

Categories:

- ``is_improvement`` uses a STRICT comparator (a flat metric never improves)
- ``observe`` resets wait on improvement, increments otherwise
- ``min_delta`` blocks trivial improvements
- ``is_plateaued`` fires at ``wait_count >= patience``
- ``fire_if_plateaued`` fires once per plateau and respects ``cooldown``
- ``reset`` returns to ±inf initial state
"""

from __future__ import annotations

from spectramr.config.schemas.enums import MetricMode
from spectramr.infrastructure.training.plateau_monitor import PlateauMonitor

# ---------------------------------------------------------------------------
# Construction / reset
# ---------------------------------------------------------------------------


def test_min_mode_initialises_best_to_pos_inf() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=3)
    assert m.best_value == float("inf")
    assert m.wait_count == 0


def test_max_mode_initialises_best_to_neg_inf() -> None:
    m = PlateauMonitor(mode=MetricMode.MAX, patience=3)
    assert m.best_value == float("-inf")


# ---------------------------------------------------------------------------
# Strict comparator
# ---------------------------------------------------------------------------


def test_min_mode_strict_flat_metric_is_not_improvement() -> None:
    """A bit-identical flat metric must NOT read as improvement (identity-collapse guard)."""
    m = PlateauMonitor(mode=MetricMode.MIN, patience=10)
    assert m.observe(1.0, iteration=10) is True  # first finite value beats +inf
    assert m.is_improvement(1.0) is False  # equal is not strictly better
    assert m.observe(1.0, iteration=20) is False
    assert m.wait_count == 1


def test_max_mode_higher_is_improvement() -> None:
    m = PlateauMonitor(mode=MetricMode.MAX, patience=10)
    assert m.observe(20.0, iteration=10) is True
    assert m.observe(25.0, iteration=20) is True
    assert m.best_value == 25.0
    assert m.wait_count == 0


# ---------------------------------------------------------------------------
# observe / wait counting
# ---------------------------------------------------------------------------


def test_observe_increments_wait_on_no_improvement() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=10)
    m.observe(1.0, iteration=10)
    m.observe(2.0, iteration=20)
    m.observe(3.0, iteration=30)
    assert m.wait_count == 2


def test_observe_records_best_iteration() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=10)
    m.observe(1.0, iteration=10)
    m.observe(0.5, iteration=25)
    assert m.best_iteration == 25


# ---------------------------------------------------------------------------
# min_delta
# ---------------------------------------------------------------------------


def test_min_delta_blocks_trivial_improvement() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=10, min_delta=0.5)
    m.observe(1.0, iteration=10)
    assert m.observe(0.9, iteration=20) is False  # only 0.1 better
    assert m.best_value == 1.0
    assert m.wait_count == 1


# ---------------------------------------------------------------------------
# is_plateaued
# ---------------------------------------------------------------------------


def test_is_plateaued_fires_at_patience() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=2)
    m.observe(1.0, iteration=10)  # best
    m.observe(2.0, iteration=20)  # wait=1
    assert m.is_plateaued() is False
    m.observe(3.0, iteration=30)  # wait=2
    assert m.is_plateaued() is True


# ---------------------------------------------------------------------------
# fire_if_plateaued + cooldown
# ---------------------------------------------------------------------------


def test_fire_if_plateaued_fires_once_then_re_arms() -> None:
    """Fires exactly when patience is crossed; wait resets so it re-arms."""
    m = PlateauMonitor(mode=MetricMode.MIN, patience=2, cooldown=0)
    assert m.fire_if_plateaued(1.0, iteration=10) is False  # improvement
    assert m.fire_if_plateaued(2.0, iteration=20) is False  # wait=1
    assert m.fire_if_plateaued(3.0, iteration=30) is True  # wait=2 -> FIRE, wait reset
    assert m.wait_count == 0
    assert m.fire_if_plateaued(4.0, iteration=40) is False  # wait=1 again


def test_cooldown_suppresses_immediate_refire() -> None:
    """With cooldown>0, a fired plateau cannot re-fire for ``cooldown`` checks."""
    m = PlateauMonitor(mode=MetricMode.MAX, patience=1, cooldown=2)
    assert m.fire_if_plateaued(10.0, iteration=10) is False  # improvement (beats -inf)
    assert m.fire_if_plateaued(9.0, iteration=20) is True  # wait=1 -> FIRE, cooldown=2
    # next two checks are suppressed even though metric keeps stagnating
    assert m.fire_if_plateaued(9.0, iteration=30) is False  # cooldown
    assert m.fire_if_plateaued(9.0, iteration=40) is False  # cooldown
    assert m.fire_if_plateaued(9.0, iteration=50) is True  # cooldown elapsed -> FIRE again


def test_a_sustained_plateau_fires_once_per_cooldown_plus_patience() -> None:
    """Bad checks inside the cooldown counted toward the next plateau, so it
    re-fired every ``cooldown + 1`` checks: [6, 12, 18, ...] here."""
    m = PlateauMonitor(mode=MetricMode.MIN, patience=5, cooldown=5)
    fired = [check for check in range(1, 40) if m.fire_if_plateaued(1.0, iteration=check)]
    assert fired == [6, 16, 26, 36]


def test_no_fire_on_a_nan_right_after_the_cooldown() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=2, cooldown=3)
    for check, value in enumerate([1.0, 2.0, 2.0], start=1):  # fires on check 3
        m.fire_if_plateaued(value, iteration=check)
    for check in (4, 5, 6):  # cooldown, non-improving
        assert m.fire_if_plateaued(2.0, iteration=check) is False
    assert m.fire_if_plateaued(float("nan"), iteration=7) is False


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_returns_to_initial_state() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=1, cooldown=3)
    m.fire_if_plateaued(1.0, iteration=10)
    m.fire_if_plateaued(2.0, iteration=20)
    m.reset()
    assert m.best_value == float("inf")
    assert m.wait_count == 0
    assert m.cooldown_remaining == 0


# ---------------------------------------------------------------------------
# Non-finite guard (M7): NaN/Inf is ignored, never read as a plateau
# ---------------------------------------------------------------------------


def test_nan_value_is_ignored_not_counted() -> None:
    import math

    m = PlateauMonitor(mode=MetricMode.MAX, patience=2)
    assert m.fire_if_plateaued(0.5, iteration=1) is False  # improvement
    # Three NaN checks must NOT accrue wait or fire a plateau.
    for it in (2, 3, 4):
        assert m.fire_if_plateaued(float("nan"), iteration=it) is False
    assert m.wait_count == 0
    assert math.isclose(m.best_value, 0.5)


def test_inf_value_is_ignored_by_observe() -> None:
    m = PlateauMonitor(mode=MetricMode.MIN, patience=1)
    m.observe(1.0, iteration=1)  # improvement
    assert m.observe(float("inf"), iteration=2) is False
    assert m.wait_count == 0  # inf did not count as a non-improving check
