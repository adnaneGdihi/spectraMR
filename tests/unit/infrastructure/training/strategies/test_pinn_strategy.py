"""``ConcretePINNSensitivityStrategy``: the epoch-end sensitivity-map save seam.

Paired with ``infrastructure/training/strategies/pinn_strategy.py`` (non-negotiable 10).

``on_epoch_end`` stamps each saved coil-sensitivity map with the global step. It read
``getattr(self.env, "step", 0)``, and ``TrainingEnvironment`` is ``frozen=True`` with no
``step`` field -- so every map from every epoch was labelled step ``0`` and successive
saves wrote indistinguishable names (pitfall #16, #1937). The read is now
``resolve_loop_iteration(self)``, off the ``LoopState`` the training loop writes at
``training_loop.py`` each step.

The assertions below **observe the value arrive** at ``_save_sensitivity_maps`` rather
than inspecting the call site's source (non-negotiable 16), and probe with a step that
is neither ``0`` nor any schema default, so the pre-fix behaviour cannot satisfy them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from spectramr.infrastructure.training.loop_state import LoopState
from spectramr.infrastructure.training.strategies.pinn_strategy import (
    ConcretePINNSensitivityStrategy,
)

#: Neither the frozen read's 0 nor any schema default -- a coincident value would
#: make the assertion vacuous.
_LIVE_STEP = 1337


def _pinn(iteration: int | None, save_interval: int = 5) -> tuple[Any, list[dict[str, int]]]:
    """A strategy stitched with only what ``on_epoch_end`` touches, plus a spy.

    ``__new__`` skips the heavy SIREN/optimizer construction; the base
    ``on_epoch_end`` this override calls through to is a ``pass``.
    """
    s = ConcretePINNSensitivityStrategy.__new__(ConcretePINNSensitivityStrategy)
    s._csm_save_interval = save_interval
    # A real ``TrainingEnvironment`` stand-in: frozen, and carrying no ``step``.
    # Stitched deliberately so reverting this hook to the old
    # ``getattr(self.env, "step", 0)`` fails these tests on the ASSERTION
    # (0 != 1337) rather than on a missing attribute.
    s.env = SimpleNamespace(generator=None, losses={})
    if iteration is not None:
        s.loop_state = LoopState(iteration=iteration, epoch=7)
    calls: list[dict[str, int]] = []
    s._save_sensitivity_maps = lambda **kw: calls.append(dict(kw))
    return s, calls


@pytest.mark.unit
def test_epoch_end_stamps_the_live_iteration_not_a_frozen_zero() -> None:
    strategy, calls = _pinn(_LIVE_STEP)
    strategy.on_epoch_end(epoch=0, metrics={})
    assert calls == [{"epoch": 0, "step": _LIVE_STEP}], (
        "the saved sensitivity map must carry the live loop iteration; the frozen "
        'getattr(self.env, "step", 0) stamped every map step 0 (pitfall #16)'
    )


@pytest.mark.unit
def test_successive_saves_carry_distinguishable_steps() -> None:
    """The defect's user-visible symptom: two epochs, one filename stem.

    Both epochs are multiples of the interval, so both save; the *step* must differ.
    """
    strategy, calls = _pinn(100, save_interval=5)
    strategy.on_epoch_end(epoch=5, metrics={})
    strategy.loop_state.iteration = 900
    strategy.on_epoch_end(epoch=10, metrics={})
    steps = [c["step"] for c in calls]
    assert steps == [100, 900], f"steps must track the loop, got {steps}"
    assert len(set(steps)) == 2, "pre-fix both were 0 and the two saves collided"


@pytest.mark.unit
def test_no_save_off_the_interval() -> None:
    """The guard the step read sits behind still gates: epoch 3 of interval 5."""
    strategy, calls = _pinn(_LIVE_STEP, save_interval=5)
    strategy.on_epoch_end(epoch=3, metrics={})
    assert calls == []


@pytest.mark.unit
def test_a_strategy_without_loop_state_degrades_to_zero_rather_than_raising() -> None:
    """``resolve_loop_iteration`` is total: an unattached strategy still saves.

    Pinned so a future owner cannot make the seam raise on the lifecycle hook, which
    runs outside the step loop.
    """
    strategy, calls = _pinn(None)
    strategy.on_epoch_end(epoch=0, metrics={})
    assert calls == [{"epoch": 0, "step": 0}]
