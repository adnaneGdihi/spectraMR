"""``MonotoneCosineLR`` decays once and never climbs back.

The planted shapes are the two ways ``CosineAnnealingLR`` restarted: stepping
past ``T_max`` and resuming a checkpoint written under a shorter horizon.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.training.scheduler_system import (
    SCHEDULER_REGISTRY,
    WarmupScheduler,
    check_scheduler_state_compatible,
)
from spectramr.infrastructure.training.schedulers.monotone_cosine import MonotoneCosineLR


def _opt(lr: float = 5e-5) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def _trace(sched, n: int) -> list[float]:
    out = []
    for _ in range(n):
        sched.step()
        out.append(sched.get_last_lr()[0])
    return out


def test_closed_form_matches_the_cosine() -> None:
    sched = MonotoneCosineLR(_opt(1.0), T_max=10, eta_min=0.0)
    lrs = _trace(sched, 10)
    for t, lr in enumerate(lrs, start=1):
        assert lr == pytest.approx(0.5 * (1 + math.cos(math.pi * t / 10)))


def test_holds_eta_min_past_t_max() -> None:
    """Planted: CosineAnnealingLR rises again over the same steps."""
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(_opt(1.0), T_max=10, eta_min=0.1)
    ref_lrs = _trace(ref, 20)
    assert max(ref_lrs[10:]) > 0.5, "the planted periodic shape no longer rises"

    lrs = _trace(MonotoneCosineLR(_opt(1.0), T_max=10, eta_min=0.1), 20)
    assert all(b <= a + 1e-12 for a, b in pairwise(lrs))
    assert lrs[-1] == pytest.approx(0.1)


def test_resume_under_a_longer_budget_keeps_the_configured_horizon() -> None:
    """A 31500-step state loaded into a 101500-step run continues the new curve."""
    old = MonotoneCosineLR(_opt(), T_max=31500, eta_min=1e-6)
    for _ in range(20000):
        old.step()
    new = MonotoneCosineLR(_opt(), T_max=101500, eta_min=1e-6)
    new.load_state_dict(old.state_dict())
    assert new.T_max == 101500
    lrs = _trace(new, 90000)
    assert all(b <= a + 1e-15 for a, b in pairwise(lrs))


def test_old_cosine_checkpoints_are_accepted() -> None:
    torch_state = torch.optim.lr_scheduler.CosineAnnealingLR(_opt(), T_max=100).state_dict()
    check_scheduler_state_compatible(MonotoneCosineLR(_opt(), T_max=100), torch_state, "lr")


def test_the_registry_builds_it_and_warmup_ends_at_the_floor() -> None:
    opt = _opt(5e-5)
    main = SCHEDULER_REGISTRY["cosine_annealing"](opt, {"T_max": 900, "eta_min": 1e-6})
    assert isinstance(main, MonotoneCosineLR)
    sched = WarmupScheduler(opt, main, warmup_steps=100, warmup_start_lr=1e-6)
    lrs = _trace(sched, 1500)
    peak = lrs.index(max(lrs))
    assert peak == 99
    assert all(b <= a + 1e-15 for a, b in pairwise(lrs[peak:]))
    assert lrs[-1] == pytest.approx(1e-6)


def test_non_positive_horizon_raises() -> None:
    with pytest.raises(ValueError, match="T_max"):
        MonotoneCosineLR(_opt(), T_max=0)
