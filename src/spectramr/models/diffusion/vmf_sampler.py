r"""Reverse sampler for von Mises-Fisher diffusion (``vmf_reverse``).

Starts from the uniform distribution on the channel sphere and walks a strided
subset of the ``T`` noise levels downwards. At each level the model predicts
the clean direction :math:`\hat z_0 = f_\theta(z_t, t)`; the next state is a
fresh draw :math:`z_{t'} \sim \mathrm{vMF}(\hat z_0, \kappa_{t'})` at the next
(lower) level, and the last prediction is returned. This is the
:math:`x_0`-prediction re-noising sampler expressed on the sphere.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.models.diffusion.samplers import register_sampler


@register_sampler(name="vmf_reverse")
class VMFReverseSampler:
    """Strided vMF reverse chain.

    Args:
        steps: Number of model evaluations (strided levels), ``1 <= steps <= T``.
    """

    def __init__(self, steps: int = 50) -> None:
        if int(steps) < 1:
            raise ValueError(f"vmf_reverse needs steps >= 1, got {steps}.")
        self.steps = int(steps)

    def levels(self, timesteps: int) -> list[int]:
        """Descending distinct levels from ``T - 1`` to ``0``."""
        if self.steps > timesteps:
            raise ValueError(
                f"vmf_reverse steps={self.steps} exceeds the schedule's {timesteps} levels."
            )
        grid = torch.linspace(timesteps - 1, 0, self.steps).round().long().tolist()
        return sorted(set(grid), reverse=True)

    @torch.no_grad()
    def sample(
        self,
        model: Any,
        shape: tuple[int, ...],
        cond: Tensor | None = None,
    ) -> Tensor:
        """Draw ``shape`` = ``(B, C, *spatial)`` unit-norm latents from ``model``.

        ``model`` must expose ``schedule`` (a ``VMFNoiseSchedule``) and be
        callable as ``model(z_t, t, cond)`` returning the predicted clean
        direction.
        """
        schedule = model.schedule
        device = schedule.kappas.device
        levels = self.levels(schedule.timesteps)
        batch = shape[0]
        z = schedule.prior_sample(shape, device=device)
        z0_hat = z
        for i, level in enumerate(levels):
            t = torch.full((batch,), level, device=device, dtype=torch.long)
            z0_hat = model(z, t, cond)
            if i + 1 < len(levels):
                t_next = torch.full((batch,), levels[i + 1], device=device, dtype=torch.long)
                z = schedule.q_sample(z0_hat, t_next)
        return z0_hat


__all__ = ["VMFReverseSampler"]
