"""Strided DDIM over the training schedule (#2305)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.inference.strided_sampling import ddim_step, strided_timesteps
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler


class TestStridedTimesteps:
    def test_starts_at_the_noisiest_index_and_descends(self) -> None:
        visits = strided_timesteps(1000, 100)
        assert visits[0] == 999
        assert visits[-1] == 9
        assert len(visits) == len(set(visits)) == 100
        assert visits == sorted(visits, reverse=True)

    def test_full_length_visits_every_index(self) -> None:
        assert strided_timesteps(28, 28) == list(range(27, -1, -1))

    @pytest.mark.parametrize("steps", [0, 1001])
    def test_a_length_outside_the_schedule_raises(self, steps: int) -> None:
        with pytest.raises(ValueError, match="training schedule"):
            strided_timesteps(1000, steps)


class TestDdimStepRecoversTheCleanSample:
    """With the true noise as the prediction, deterministic DDIM must land on x0
    at any stride. The sampler that stepped to t-1 instead of the previous
    visited index missed by 2.26 at 100 of 1000 steps."""

    @staticmethod
    def _sample(schedule: str, train_steps: int, steps: int, eta: float = 0.0) -> float:
        torch.manual_seed(0)
        sched = DiffusionScheduler(num_timesteps=train_steps, beta_schedule=schedule)
        # float64: at t = T-1 alpha_bar is ~1e-6 under this cosine, and x0 is
        # divided by its root, so float32 round-off alone reaches 2e-3.
        abar = sched.alphas_cumprod.double()
        x0 = torch.rand(1, 1, 8, 8, dtype=torch.float64) * 0.8 - 0.4
        eps = torch.randn(1, 1, 8, 8, dtype=torch.float64)
        visits = strided_timesteps(train_steps, steps)
        x = abar[visits[0]].sqrt() * x0 + (1 - abar[visits[0]]).sqrt() * eps
        for i, t in enumerate(visits):
            prev = abar[visits[i + 1]] if i + 1 < len(visits) else 1.0
            x = ddim_step(x, eps, abar[t], prev, eta)
        return (x - x0).abs().max().item()

    @pytest.mark.parametrize("steps", [1000, 100, 50, 7])
    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_any_stride_is_exact(self, schedule: str, steps: int) -> None:
        assert self._sample(schedule, 1000, steps) < 1e-4

    def test_the_last_step_returns_the_predicted_clean_sample(self) -> None:
        x0, eps = torch.ones(2), torch.full((2,), 0.5)
        abar = torch.tensor(0.64)
        x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
        assert torch.allclose(ddim_step(x_t, eps, abar, 1.0, eta=1.0), x0)
