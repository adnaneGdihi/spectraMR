"""Regression: DiffusionScheduler silently coerced unknown beta schedules.

2026-06-11 infrastructure audit. An unsupported ``beta_schedule`` (typo, or a
``sqrt`` / ``quadratic`` value advertised by ``training.noise_schedule``) was
silently rewritten to ``linear`` and the ``else: raise`` was dead. It now raises
(pitfall #9).
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.training.schedulers.diffusion_scheduler import (
    DiffusionScheduler,
)


def test_unknown_beta_schedule_raises():
    with pytest.raises(ValueError, match="Unknown beta schedule"):
        DiffusionScheduler(num_timesteps=10, beta_schedule="quadratic")


def test_supported_schedules_still_build():
    for sched in ("linear", "cosine"):
        s = DiffusionScheduler(num_timesteps=10, beta_schedule=sched)
        assert s.betas.numel() == 10
        assert s.beta_schedule == sched  # not silently rewritten


@pytest.mark.parametrize("bad_t", [10, 999, -1])
def test_a_timestep_outside_the_schedule_raises(bad_t):
    """A clamp answered t >= T with the last entry, so a timestep taken from a
    different schedule produced plausible numbers (#2306)."""
    import torch

    s = DiffusionScheduler(num_timesteps=10, beta_schedule="linear")
    x = torch.zeros(2, 1, 4, 4)
    with pytest.raises((IndexError, RuntimeError)):
        s.q_sample(x, torch.tensor([0, bad_t]))


def test_the_last_timestep_is_still_in_range():
    import torch

    s = DiffusionScheduler(num_timesteps=10, beta_schedule="linear")
    out = s.q_sample(torch.ones(1, 1, 2, 2), torch.tensor([9]))
    assert torch.isfinite(out).all()
