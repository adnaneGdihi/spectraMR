"""``SliceToVolumeStrategy``: the loss-computer iteration seam.

Paired with ``infrastructure/training/strategies/slice_to_volume_strategy.py``
(non-negotiable 10).

``_compute_losses_impl`` threads an ``iteration`` into
``UnifiedReconstructionLossComputer.compute``, which is what advances every
iteration-dependent schedule the computer owns -- the warm-up gate above all, where
``resolve_loss_weight`` returns ``0.0`` while ``iteration < warmup_iterations``. It
read ``int(kwargs.get("step", 0))``, but the training loop passes ``iteration=`` and
never ``step=``, so the value was a constant ``0``: the gate stayed shut for the whole
run and every warm-up-gated loss was **absent from ``components``**, not scaled to
zero (pitfall #16, #1937).

No ``experiments/inprogress/`` arm resolves to this strategy today, so the defect was
latent here -- it is live on ``vf_admm_strategy.py``, which carried the identical read.
That is precisely why it needs a test: a latent defect has no run to reveal it.

The spy below **observes the value arrive** at ``compute`` (non-negotiable 16) and
probes with an iteration that is neither ``0`` nor a schema default.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectramr.infrastructure.training.loop_state import LoopState
from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
    SliceToVolumeStrategy,
)

#: Past the schema-default ``warmup_iterations`` of 1000, and not 0.
_LIVE_ITERATION = 1337


class _SpyComputer:
    """Records the kwargs ``_compute_losses_impl`` hands the real computer."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compute(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            total=torch.zeros((), requires_grad=True),
            components={"l1": torch.zeros(())},
        )


def _s2v(iteration: int) -> tuple[Any, _SpyComputer]:
    """A strategy stitched with only what ``_compute_losses_impl`` touches."""
    s = SliceToVolumeStrategy.__new__(SliceToVolumeStrategy)
    spy = _SpyComputer()
    s.loss_computer = spy
    s.loop_state = LoopState(iteration=iteration, epoch=3)
    s.env = SimpleNamespace(generator=lambda x: x, losses={"l1": {"weight": 1.0}})
    s.lambda_through_plane = 0.1
    s.lambda_ortho = 0.05
    return s, spy


#: (B, C, D, H, W) -- both helpers this hook calls reject anything but 5-D.
def _vol() -> torch.Tensor:
    return torch.rand(1, 1, 4, 8, 8)


@pytest.mark.unit
def test_loss_computer_receives_the_live_iteration_not_a_frozen_zero() -> None:
    strategy, spy = _s2v(_LIVE_ITERATION)
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=3)
    assert len(spy.calls) == 1
    assert spy.calls[0]["iteration"] == _LIVE_ITERATION, (
        "the loss computer must receive the live loop iteration; frozen at 0 the "
        "warm-up gate never opens and a gated loss never enters components "
        "(pitfall #16, #1937)"
    )


@pytest.mark.unit
def test_a_stale_step_kwarg_does_not_win_over_the_loop_state() -> None:
    """The loop passes ``iteration=``; a caller passing ``step=`` must not steer it.

    This is the shape the fix retires: had the read stayed on ``kwargs``, a caller
    supplying ``step=0`` would still freeze the schedule.
    """
    strategy, spy = _s2v(_LIVE_ITERATION)
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=3, step=0)
    assert spy.calls[0]["iteration"] == _LIVE_ITERATION


@pytest.mark.unit
def test_the_iteration_advances_with_the_loop() -> None:
    """Two steps, two values -- a constant would satisfy a single-point assertion."""
    strategy, spy = _s2v(0)
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=0)
    strategy.loop_state.iteration = _LIVE_ITERATION
    strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=0)
    assert [c["iteration"] for c in spy.calls] == [0, _LIVE_ITERATION]


@pytest.mark.unit
def test_the_hook_still_returns_its_own_anchor_terms() -> None:
    """Guard the edit's blast radius: the two bespoke terms survive the change."""
    strategy, _ = _s2v(_LIVE_ITERATION)
    out = strategy._compute_losses_impl(input_batch=_vol(), target_batch=_vol(), epoch=3)
    assert {"g_total_loss", "g_through_plane_consistency", "g_isotropy_consistency"} <= set(out)
