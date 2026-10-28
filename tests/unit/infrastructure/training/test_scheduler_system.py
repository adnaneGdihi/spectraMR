"""Scheduler factories and the warmup wrapper.

Covers the factories added for issue #533 (``linear``, ``constant`` — both named
in the corpus' ``lr_scheduler_strategy`` but previously registered nowhere) and
the ``WarmupScheduler`` fix that made the warmup LR apply at construction instead
of one optimizer step later.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR

from spectramr.infrastructure.training.scheduler_system import (
    SCHEDULER_REGISTRY,
    WarmupScheduler,
)


@pytest.fixture
def optimizer() -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)


class TestRegistryCoverage:
    @pytest.mark.parametrize(
        "name",
        [
            "cosine_annealing",
            "cosine_annealing_warm_restarts",
            "step_lr",
            "exponential",
            "linear",
            "constant",
        ],
    )
    def test_family_is_registered(self, name):
        assert name in SCHEDULER_REGISTRY

    def test_linear_and_constant_were_the_missing_two(self, optimizer):
        """``linear_decay`` / ``constant`` appear in the corpus; both now build."""
        assert isinstance(SCHEDULER_REGISTRY["linear"](optimizer, {}), LinearLR)
        assert isinstance(SCHEDULER_REGISTRY["constant"](optimizer, {}), ConstantLR)

    def test_constant_default_really_holds_the_lr(self, optimizer):
        """torch's ConstantLR default is 1/3 for 5 epochs; 'constant' must not be."""
        sched = SCHEDULER_REGISTRY["constant"](optimizer, {})
        start = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(start)

    def test_linear_decays(self, optimizer):
        sched = SCHEDULER_REGISTRY["linear"](
            optimizer, {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 10}
        )
        start = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        assert optimizer.param_groups[0]["lr"] < start


class TestWarmupScheduler:
    def test_warmup_lr_is_applied_at_construction(self, optimizer):
        """Without this the FIRST optimizer step ran at the full base LR.

        ``step()`` only runs after an optimizer step, so the update warmup exists
        to protect was the one update that never saw it (issue #533).
        """
        main = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        WarmupScheduler(optimizer, main, warmup_steps=100, warmup_start_lr=1e-7)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-7)

    def test_zero_warmup_leaves_the_lr_untouched(self, optimizer):
        main = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        before = optimizer.param_groups[0]["lr"]
        WarmupScheduler(optimizer, main, warmup_steps=0, warmup_start_lr=1e-7)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(before)

    def test_warmup_ramps_then_hands_over_to_the_main_scheduler(self, optimizer):
        main = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        sched = WarmupScheduler(optimizer, main, warmup_steps=10, warmup_start_lr=1e-7)
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            sched.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        # monotone ramp through warmup, ending at the base LR
        assert all(b >= a for a, b in pairwise(lrs))
        assert lrs[-1] == pytest.approx(5e-5, rel=1e-3)
        # then the main scheduler drives it back down
        for _ in range(200):
            sched.step()
        assert optimizer.param_groups[0]["lr"] < 5e-5

    def test_state_dict_roundtrip(self, optimizer):
        main = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        sched = WarmupScheduler(optimizer, main, warmup_steps=5, warmup_start_lr=1e-7)
        for _ in range(3):
            sched.step()
        state = sched.state_dict()

        opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
        main2 = CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
        sched2 = WarmupScheduler(opt2, main2, warmup_steps=5, warmup_start_lr=1e-7)
        sched2.load_state_dict(state)
        assert sched2.current_step == sched.current_step


def test_every_recommended_scheduler_is_one_the_factory_builds():
    """A recommendation naming a removed scheduler (the plateau family, #2301)
    turned ``create_scheduler(..., model_type=...)`` into a raise."""
    import torch

    from spectramr.infrastructure.training.scheduler_system import SchedulerFactory

    recs = SchedulerFactory.get_model_specific_scheduler_recommendations()
    for model_type in recs:
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        SchedulerFactory.create_scheduler(opt, "cosine", None, model_type=model_type)


# --- Resuming across a scheduler-family change (#2748) --------------------------------
# A warm-restart state loaded into CosineAnnealingLR keeps last_epoch past T_max and the LR
# climbs back up; the loads must refuse it, on the bare and the warmup-wrapped path alike.


def _opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)


def _warm_restarts(opt):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=70000, T_mult=2)


def _cosine(opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=31500, eta_min=1e-6)


def test_a_warm_restart_state_is_refused_by_a_cosine_scheduler() -> None:
    from spectramr.infrastructure.training.scheduler_system import (
        SchedulerStateMismatchError,
        load_scheduler_state,
    )

    old = _warm_restarts(_opt())
    for _ in range(5):
        old.step()
    with pytest.raises(SchedulerStateMismatchError, match="different scheduler family"):
        load_scheduler_state(_cosine(_opt()), old.state_dict())


def test_the_warmup_wrapped_path_refuses_it_too() -> None:
    """DeepSpeed restores through WarmupScheduler.load_state_dict, not the helper."""
    from spectramr.infrastructure.training.scheduler_system import (
        SchedulerStateMismatchError,
        WarmupScheduler,
    )

    old = WarmupScheduler(_o := _opt(), main_scheduler=_warm_restarts(_o), warmup_steps=10)
    new = WarmupScheduler(_n := _opt(), main_scheduler=_cosine(_n), warmup_steps=10)
    with pytest.raises(SchedulerStateMismatchError):
        new.load_state_dict(old.state_dict())


def test_a_same_family_state_still_resumes() -> None:
    from spectramr.infrastructure.training.scheduler_system import (
        WarmupScheduler,
        load_scheduler_state,
    )

    old = WarmupScheduler(_o := _opt(), main_scheduler=_cosine(_o), warmup_steps=10)
    for _ in range(25):
        old.step()
    new = WarmupScheduler(_n := _opt(), main_scheduler=_cosine(_n), warmup_steps=10)
    load_scheduler_state(new, old.state_dict())
    assert new.current_step == 25
    assert new.main_scheduler.last_epoch == old.main_scheduler.last_epoch


def test_cosine_annealing_key_builds_the_monotone_schedule() -> None:
    """The registry's cosine never re-ascends; the periodic torch class did."""
    from spectramr.infrastructure.training.schedulers.monotone_cosine import MonotoneCosineLR

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = SCHEDULER_REGISTRY["cosine_annealing"](opt, {"T_max": 5, "eta_min": 0.0})
    assert isinstance(sched, MonotoneCosineLR)
    for _ in range(12):
        sched.step()
    assert sched.get_last_lr()[0] == 0.0
