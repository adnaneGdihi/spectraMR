"""Normalized additive reverse loop for k-space cold diffusion.

``additive`` (Bansal Algorithm 2) adds ``x0 * (M_{t-1} - M_t)`` to the state and
returns the terminal call's estimate unmodified. Nothing in it bounds a write:
its data consistency acts on ``x0`` and never re-pins the state, no magnitude
ceiling is applied, and the generator's ``output_kspace_clip_ratio`` is not
reached inside the sampler. The model reads its own writes at the next call, so
an estimate that overshoots by ``g`` produces an input that overshoots, and the
error compounds with the number of calls -- which is why the 8x and 32x rungs,
with 15 and 29 calls, wash out while the 2x rung does not.

``reverse_sampling_mode: additive_normalized`` keeps the additive structure --
every scheduled step calls the model, each step contributes the coefficients
between its rung and the next, and the terminal call contributes an estimate of
every unmeasured coefficient -- and normalizes what is accumulated:

1. The observed support is re-pinned by the configured data consistency after
   every call, so the next input carries the measurement exactly.
2. The coefficients a step contributes are divided per radial band by
   ``max(1, RMS_b(x0) / (rho * RMS_b(y)))`` (``reverse_envelope``), the energy
   envelope of the measured lines, with ``rho = reverse_clip_ratio``.
3. The per-coefficient magnitude clamp of the freeze loops is applied last,
   to the unmeasured coefficients only; the measured ones keep the DC output.
4. Each unmeasured coefficient holds the MEAN of every estimate written to it,
   ``sum / writes``. On a nested ladder a coefficient receives its reveal-step
   estimate and the terminal estimate, so the divisor is 2 and neither call
   authors it alone; on a ladder whose masks do not nest, a coefficient that
   re-enters the support is averaged where the plain loop sums it.

The contribution of step ``i`` keys off ``schedule[i + 1]`` rather than
``t - 1``, so a strided schedule tiles the support instead of skipping the rungs
between scheduled levels. Every coefficient can be written more than once, so
the mode is absent from ``reveal_attribution.PARTITIONED_REVERSE_MODES``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from spectramr.models.diffusion.reverse_envelope import (
    normalize_band_energy,
    observed_band_envelope,
)
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
NORMALIZED_ADDITIVE_REVERSE_MODE = "additive_normalized"


def additive_increment(
    sampler: PhysicsInformedColdDiffusion,
    i: int,
    timestep_schedule: list[int],
    obs: Tensor,
    x: Tensor,
) -> Tensor:
    """Unmeasured coefficients step ``i`` contributes: ``M(next) - M(current)``, or all at the end."""
    if i + 1 >= len(timestep_schedule):
        return 1.0 - obs
    batch, device = x.shape[0], x.device

    def _mask(t_key: int) -> Tensor:
        return sampler.process.cascade_batch_masks(
            torch.full((batch,), t_key, device=device, dtype=torch.long),
            tuple(x.shape[-2:]),
            device,
            acquired=sampler.measured_lines,
        ).float()

    grown = (_mask(timestep_schedule[i + 1]) - _mask(timestep_schedule[i])).clamp_min(0.0)
    return grown * (1.0 - obs)


@torch.no_grad()
def sample_additive_normalized(
    sampler: PhysicsInformedColdDiffusion,
    measurement: Tensor,
    mask: Tensor | None,
    timestep_schedule: list[int],
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Run the normalized additive loop for ``sampler`` over ``timestep_schedule``."""
    device = measurement.device
    batch = measurement.shape[0]
    eps = 1e-8

    x = measurement.clone()
    obs = sampler._observed_support(measurement, mask, eps)
    ctx = prepare_reverse(sampler, measurement, obs, eps=eps)
    envelope = observed_band_envelope(
        measurement, obs, log_scaled=sampler.kspace_log_scaled, eps=eps
    )
    total = torch.zeros_like(x)
    writes = torch.zeros_like(obs)
    trajectory: list[torch.Tensor] = [x.clone()] if return_trajectory else []
    max_divisor = torch.ones((), device=device)

    for i, t_idx in enumerate(timestep_schedule):
        t = torch.full((batch,), t_idx, device=device, dtype=torch.long)
        x0 = sampler.denoise(x, t, torch.clamp(obs + (writes > 0).to(obs.dtype), max=1.0))
        where = f"{NORMALIZED_ADDITIVE_REVERSE_MODE} step {i} (t={t_idx})"
        x0 = refine_estimate(sampler, x0, measurement, obs, ctx, where=where)

        increment = additive_increment(sampler, i, timestep_schedule, obs, x0)
        written_x0, divisor = normalize_band_energy(
            x0,
            envelope,
            increment,
            ratio=sampler.reverse_clip_ratio,
            log_scaled=sampler.kspace_log_scaled,
            eps=eps,
        )
        max_divisor = torch.maximum(max_divisor, divisor.amax())
        # ``write_ceil`` holds the acquired lines at or above their own modulus,
        # since the band-local ceiling can sit below an outer band's measured peak (#2310).
        written_x0 = bound_estimate(sampler, written_x0, obs, ctx, eps)

        total = total + written_x0 * increment
        writes = writes + increment
        written = (writes > 0).to(x.dtype)
        mean = total / writes.clamp_min(1.0)
        x = written_x0 * obs + (mean * written + x * (1.0 - written)) * (1.0 - obs)
        check_step(
            sampler, x, measurement, obs, step=i, t=t_idx, ceil=ctx.ceil,
            trust_radius=ctx.trust_radius,
        )

        if return_trajectory:
            trajectory.append(x.clone())

    check_coverage(sampler, (writes > 0).to(obs.dtype), obs)
    sampler.last_effective_steps = len(timestep_schedule)
    sampler.last_skipped_steps = 0
    sampler.last_terminal_step = timestep_schedule[-1] if timestep_schedule else None
    sampler.last_band_divisor_max = max_divisor

    if return_trajectory:
        return x, trajectory
    return x


__all__ = [
    "NORMALIZED_ADDITIVE_REVERSE_MODE",
    "additive_increment",
    "sample_additive_normalized",
]
