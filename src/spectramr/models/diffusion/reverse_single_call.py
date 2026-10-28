"""Single-call reconstruction for k-space cold diffusion.

The network is trained to map ``M_t * x_0`` to ``x_0`` in one forward pass. The
only reverse-loop input that matches that training distribution exactly is the
measurement itself at its own rung: every later state of a multi-step loop
contains the network's own predictions where training showed measured data.

``reverse_sampling_mode: single_call_dc`` therefore calls the model once, at
the head of the schedule (the rung the measurement is degraded at), applies the
configured data consistency on the observed support and the same magnitude
ceiling as the other loops, and writes every unobserved coefficient from that
one estimate. It is the baseline a multi-step loop has to beat.

Every coefficient is written by one call, so there is no per-step reveal
partition to attribute error to; the mode is absent from
``reveal_attribution.PARTITIONED_REVERSE_MODES``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spectramr.models.diffusion.reverse_hooks import (
    bound_estimate,
    prepare_reverse,
    refine_estimate,
)
from spectramr.models.diffusion.reverse_integrity import check_coverage, check_step

if TYPE_CHECKING:
    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion

#: The reverse mode this module implements.
SINGLE_CALL_REVERSE_MODE = "single_call_dc"


@torch.no_grad()
def sample_single_call(
    sampler: PhysicsInformedColdDiffusion,
    measurement: Tensor,
    mask: Tensor | None,
    timestep_schedule: list[int],
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """One model call at ``timestep_schedule[0]``, then data consistency."""
    if not timestep_schedule:
        raise ValueError("single_call_dc needs a non-empty timestep schedule.")
    eps = 1e-8
    head = int(timestep_schedule[0])
    obs = sampler._observed_support(measurement, mask, eps)
    ctx = prepare_reverse(sampler, measurement, obs, eps=eps)

    t = torch.full((measurement.shape[0],), head, device=measurement.device, dtype=torch.long)
    x0 = sampler.denoise(measurement.clone(), t, obs)
    where = f"{SINGLE_CALL_REVERSE_MODE} step 0 (t={head})"
    x0 = refine_estimate(sampler, x0, measurement, obs, ctx, where=where)
    x = bound_estimate(sampler, x0, obs, ctx, eps)
    check_step(
        sampler, x, measurement, obs, step=0, t=head, ceil=ctx.ceil, trust_radius=ctx.trust_radius
    )
    check_coverage(sampler, 1.0 - obs, obs)

    # The rest of the schedule is never evaluated; the stats say so.
    sampler.last_effective_steps = 1
    sampler.last_skipped_steps = len(timestep_schedule) - 1
    sampler.last_terminal_step = head

    if return_trajectory:
        return x, [measurement.clone(), x.clone()]
    return x


__all__ = ["SINGLE_CALL_REVERSE_MODE", "sample_single_call"]
