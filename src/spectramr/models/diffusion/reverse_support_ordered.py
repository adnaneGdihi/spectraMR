"""Support-ordered commitment reverse loop (Algorithm D).

A freeze loop fixes each unmeasured line with the estimate of the step at which
the random ladder happens to reveal it, so a line far from any measured line can
be frozen from an input that holds only a few real lines. This loop orders the
commitment by SUPPORT instead: at step ``i`` of ``S`` it commits the unmeasured
lines whose phase-encode distance to the nearest measured line is at most

.. math::

    d_i = 1 + \\left\\lfloor (R - 1)\\, \\frac{i}{\\max(S - 2, 1)} \\right\\rfloor,
    \\qquad R = \\texttt{reverse\\_support\\_radius},

and the terminal step commits every line left, so lines beyond ``R`` take the
estimate of the call with the most context. Lines not yet committed are not
frozen: they hold the latest bounded estimate and are rewritten every step.

The model input stays on the training ladder, ``z = s \\odot (M_t \\cup O)``,
where the state ``s`` holds the data-consistent measured lines, the committed
lines and the latest estimates. Every value in ``s`` is one bounded estimate
(``reverse_hooks.bound_estimate``) and nothing is summed, so the state obeys
the ceiling at every step: the failure of plain ``additive`` has no path here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spectramr.models.diffusion.reverse_envelope import per_coefficient_support
from spectramr.models.diffusion.reverse_hooks import (
    bound_estimate,
    prepare_reverse,
    refine_estimate,
)
from spectramr.models.diffusion.reverse_integrity import check_coverage, check_step
from spectramr.models.diffusion.reverse_physics import SUPPORT_ORDERED_REVERSE_MODE

if TYPE_CHECKING:
    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion


def line_distance(obs: Tensor, like: Tensor, line_dim: int = -2) -> Tensor:
    """``[B, N]`` phase-encode distance of each line to the nearest measured line.

    A line is a row (``line_dim=-2``) or a column (``-1``), whichever axis the mask
    removes lines along (#2160).
    """
    support = per_coefficient_support(obs, like) > 0
    across = -1 if line_dim == -2 else -2
    lines = support.any(dim=across).any(dim=-2)
    count = lines.shape[-1]
    index = torch.arange(count, device=lines.device, dtype=torch.float32)
    gap = (index[:, None] - index[None, :]).abs()
    far = torch.full_like(gap, float("inf"))
    return torch.where(lines[:, None, :], gap[None], far[None]).amin(dim=-1)


def commit_radius(step: int, steps: int, radius: int) -> int:
    """``d_i`` of a non-terminal step: 1 at the first step, ``radius`` at the last one before the terminal."""
    return 1 + ((radius - 1) * step) // max(steps - 2, 1)


@torch.no_grad()
def sample_support_ordered(
    sampler: PhysicsInformedColdDiffusion,
    measurement: Tensor,
    mask: Tensor | None,
    timestep_schedule: list[int],
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Run the support-ordered commitment loop for ``sampler`` over ``timestep_schedule``."""
    device = measurement.device
    batch = measurement.shape[0]
    eps = 1e-8
    steps = len(timestep_schedule)
    radius = sampler.physics.support_radius
    assert radius is not None, "resolve_reverse_physics_kwargs requires it in this mode"

    obs = sampler._observed_support(measurement, mask, eps)
    ctx = prepare_reverse(sampler, measurement, obs, eps=eps)
    if getattr(sampler.process, "line_axis", None) == "x":
        distance = line_distance(obs, measurement, -1)[:, None, None, :]
    else:
        distance = line_distance(obs, measurement)[:, None, :, None]
    state = measurement.clone()
    committed = torch.zeros_like(obs)
    trajectory: list[torch.Tensor] = [state.clone()] if return_trajectory else []

    for i, t_idx in enumerate(timestep_schedule):
        t = torch.full((batch,), t_idx, device=device, dtype=torch.long)
        rung = sampler.process.cascade_batch_masks(
            t, tuple(state.shape[-2:]), device, acquired=sampler.measured_lines
        )
        visible = torch.clamp(rung.to(state.dtype) + obs, 0.0, 1.0)
        x0 = sampler.denoise(state * visible, t, visible)
        where = f"{SUPPORT_ORDERED_REVERSE_MODE} step {i} (t={t_idx})"
        x0 = refine_estimate(sampler, x0, measurement, obs, ctx, where=where)
        x0 = bound_estimate(sampler, x0, obs, ctx, eps)

        open_bins = (1.0 - committed) * (1.0 - obs)
        if i + 1 >= steps:
            reveal = open_bins
        else:
            near = (distance <= commit_radius(i, steps, radius)).to(obs.dtype)
            reveal = open_bins * near
        state = x0 * (1.0 - committed) + state * committed
        check_step(
            sampler,
            state,
            measurement,
            obs,
            step=i,
            t=t_idx,
            ceil=ctx.ceil,
            reveal=reveal,
            committed=committed,
            trust_radius=ctx.trust_radius,
        )
        committed = torch.clamp(committed + reveal, max=1.0)
        if return_trajectory:
            trajectory.append(state.clone())

    check_coverage(sampler, committed, obs)
    sampler.last_effective_steps = steps
    sampler.last_skipped_steps = 0
    sampler.last_terminal_step = timestep_schedule[-1] if timestep_schedule else None

    if return_trajectory:
        return state, trajectory
    return state


__all__ = ["commit_radius", "line_distance", "sample_support_ordered"]
