"""Reveal-then-refine reverse loop for k-space cold diffusion.

The freeze loops (``replace_freeze*``) write each unobserved coefficient once and
never revisit it, so the outermost band is committed from the estimate made with
the least context (the highest rung) and the terminal call cannot correct it.
This loop reveals on the same schedule but keeps every committed unobserved
coefficient live: each later model call moves it toward the newer estimate by
``refine_alpha``,

.. math::

    x \\leftarrow (1-\\alpha)\\,x + \\alpha\\,\\hat{x}_0 \\quad \\text{on committed bins},
    \\qquad x \\leftarrow \\hat{x}_0 \\quad \\text{on newly revealed bins},

with the observed support updated through the configured data consistency, as in
``replace_freeze_dc``. ``alpha = 1`` makes the terminal call the sole author of
every unobserved coefficient (Bansal Algorithm 1 restricted to the revealed
support); ``alpha < 1`` averages the estimates of successive rungs.

The reveal at each step is keyed to the next rung, so the call at ``t`` sees
exactly the support ``M(t)`` it is labelled with, and the terminal call runs at
``t = 0`` on a ladder with an identity rung and revisits the whole plane. Coefficients
are written more than once, so the mode does not partition the reveal and is absent
from ``reveal_attribution.PARTITIONED_REVERSE_MODES``.

Boundedness: ``x_0`` is clamped to the fixed per-sample ceiling before the blend,
and a convex combination of two coefficients inside the ceiling stays inside it,
so the output obeys the same bound as the freeze loops.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from spectramr.models.diffusion.reveal_plan import plan_reveals
from spectramr.models.diffusion.reverse_hooks import (
    bound_estimate,
    prepare_reverse,
    refine_estimate,
)
from spectramr.models.diffusion.reverse_integrity import check_coverage, check_step

if TYPE_CHECKING:
    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion

logger = logging.getLogger(__name__)

#: The reverse mode this module implements.
REFINE_REVERSE_MODE = "reveal_refine_dc"


def validate_refine_alpha(reverse_mode: str, refine_alpha: float | None) -> float | None:
    """Resolve ``refine_alpha`` against ``reverse_mode``; raise on any mismatch.

    The weight is required by the refine mode, since no value is a neutral
    default: ``1`` discards every earlier estimate and a small value keeps the
    first, highest-rung estimate. Declaring it under any other mode would be an
    unread knob.
    """
    if reverse_mode != REFINE_REVERSE_MODE:
        if refine_alpha is not None:
            raise ValueError(
                f"reverse_refine_alpha={refine_alpha!r} is only read by "
                f"reverse_sampling_mode={REFINE_REVERSE_MODE!r}, but the mode is "
                f"{reverse_mode!r}. Remove the key or change the mode."
            )
        return None
    if refine_alpha is None:
        raise ValueError(
            f"reverse_sampling_mode={REFINE_REVERSE_MODE!r} requires "
            "model.model_kwargs.reverse_refine_alpha in (0, 1]: the weight given "
            "to each newer estimate of an already-revealed coefficient."
        )
    alpha = float(refine_alpha)
    if not (math.isfinite(alpha) and 0.0 < alpha <= 1.0):
        raise ValueError(f"reverse_refine_alpha must lie in (0, 1], got {refine_alpha!r}.")
    return alpha


@torch.no_grad()
def sample_reveal_refine(
    sampler: PhysicsInformedColdDiffusion,
    measurement: Tensor,
    mask: Tensor | None,
    timestep_schedule: list[int],
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Run the reveal-then-refine loop for ``sampler`` over ``timestep_schedule``."""
    device = measurement.device
    batch = measurement.shape[0]
    eps = 1e-8
    alpha = float(sampler.refine_alpha)

    x = measurement.clone()
    obs = sampler._observed_support(measurement, mask, eps)
    ctx = prepare_reverse(sampler, measurement, obs, eps=eps)
    committed = torch.zeros_like(obs)
    trajectory = [x.clone()] if return_trajectory else None
    plan = plan_reveals(sampler, timestep_schedule, committed, obs, x, batch, device)
    first = plan.first_active()
    lead = len(timestep_schedule) if first is None else first

    skipped = 0
    terminal: int | None = None
    for i, t_idx in enumerate(timestep_schedule):
        reveal = plan.reveal(i, committed.dtype)
        # Under hard DC a call is a no-op only while nothing has been revealed
        # yet; once any bin is committed every later call refines it.
        if sampler.dc_method == "hard" and i < lead:
            skipped += 1
            if return_trajectory:
                trajectory.append(x.clone())
            continue

        t = torch.full((batch,), t_idx, device=device, dtype=torch.long)
        terminal = t_idx
        x0 = sampler.denoise(x, t, torch.clamp(obs + committed, max=1.0))
        where = f"{REFINE_REVERSE_MODE} step {i} (t={t_idx})"
        x0 = refine_estimate(sampler, x0, measurement, obs, ctx, where=where)
        x0 = bound_estimate(sampler, x0, obs, ctx, eps)

        refined = (1.0 - alpha) * x + alpha * x0
        own_new = torch.clamp(reveal + obs, 0.0, 1.0)
        x = x0 * own_new + refined * committed + x * (1.0 - own_new - committed)
        check_step(
            sampler, x, measurement, obs, step=i, t=t_idx, ceil=ctx.ceil,
            reveal=reveal, committed=committed, trust_radius=ctx.trust_radius,
        )
        committed = torch.clamp(committed + reveal, max=1.0)

        if return_trajectory:
            trajectory.append(x.clone())

    check_coverage(sampler, committed, obs)
    if skipped:
        logger.info(
            "cold reverse loop (%s): %d of %d scheduled steps preceded the first "
            "reveal and were skipped",
            REFINE_REVERSE_MODE,
            skipped,
            len(timestep_schedule),
        )
    sampler.last_effective_steps = len(timestep_schedule) - skipped
    sampler.last_skipped_steps = skipped
    sampler.last_terminal_step = terminal

    if return_trajectory:
        return x, trajectory
    return x


__all__ = ["REFINE_REVERSE_MODE", "sample_reveal_refine", "validate_refine_alpha"]
