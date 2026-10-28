"""The fixed order in which every bounded reverse loop treats a model estimate.

Plain ``additive`` whitened the 8x and 32x rungs because nothing bounded what it
accumulated, and the model re-read its own writes. The loops that call these
hooks treat every estimate the same way, in one order, so no physics option can
reintroduce that failure:

1. **data consistency** on the measured support (``dc_method``; with
   ``reverse_dc_trust: fitted_decay`` the Wiener blend of ``reverse_trust``);
2. **projection** (``reverse_projection``): coil self-consistency or joint TV,
   which re-pins the measured bins to their step-1 values;
3. **reverse-step noise**, off the measured support;
4. **bound**: the decay-envelope row divisor, then one phase-preserving clamp
   to ``write_ceil``.

Step 4 is last, so whatever steps 1-3 do, the estimate a loop writes is under
the ceiling. Loops REPLACE or AVERAGE their writes and never sum them, so the
bound on one estimate is a bound on the state.

``write_ceil`` is ``kspace_process.measured_write_ceiling``: ``ceil`` on unmeasured
coefficients, ``max(ceil, |y|)`` on measured ones under a non-hard ``dc_method`` (an
unclamped blend would carry the model's overshoot into the next input), and unbounded
under hard DC, where the bin already equals ``y`` (#2310). Every loop uses it, so the
clamp is a projection onto a set that contains ``y``, as ``fitted_decay``'s ``3 sigma``
discrepancy guarantee needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spectramr.infrastructure.physics.data_consistency import NoiseAdaptiveDataConsistency
from spectramr.models.diffusion.reverse_decay import (
    DecayEnvelope,
    bound_bins_by_decay,
    fit_decay_envelope,
    tighten_ceiling,
)
from spectramr.models.diffusion.reverse_decay_axis import LINE_DECAY_AXES
from spectramr.models.diffusion.reverse_joint_tv import build_joint_tv
from spectramr.models.diffusion.reverse_physics import DECAY_ENVELOPE_SHAPES, LINE_ENVELOPE_AXIS
from spectramr.models.diffusion.reverse_spirit import build_spirit
from spectramr.models.diffusion.reverse_trust import (
    FittedTrust,
    apply_fitted_trust,
    build_fitted_trust,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion


@dataclass(frozen=True)
class ReverseContext:
    """What one :meth:`sample` call fixes before its first step."""

    ceil: Tensor
    write_ceil: Tensor
    decay: DecayEnvelope | None
    projector: Callable[..., Tensor] | None
    trust: FittedTrust | None

    @property
    def trust_radius(self) -> Tensor | None:
        """``c * sigma`` for the ``observed_discrepancy`` invariant, else None."""
        return None if self.trust is None else self.trust.radius


#: ``reverse_projection`` -> builder of the per-call projector (None: no projection).
_PROJECTORS: dict[str, Callable[..., Any] | None] = {
    "none": None,
    "spirit": build_spirit,
    "joint_tv": build_joint_tv,
}


def _lines_are_columns(sampler: PhysicsInformedColdDiffusion) -> bool:
    """Whether the process's mask removes columns (dim -1) rather than rows (#2160)."""
    return getattr(sampler.process, "line_axis", None) == "x"


def _on_transposed_plane(project: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Run a projector written for phase-encode rows on a plane whose lines are columns.

    SPIRiT calibrates on fully measured rows and joint TV differences along y; both
    are statements about the phase-encode direction, so on a column mask they run on
    the transposed plane, which is the same algorithm with the axes renamed.
    """

    def projected(x0: Tensor, obs_now: Tensor, *, where: str) -> Tensor:
        return project(x0.mT, obs_now.mT, where=where).mT

    return projected


def noise_estimator(sampler: PhysicsInformedColdDiffusion) -> NoiseAdaptiveDataConsistency:
    """The model's ``noise_adaptive`` layer when it has one, so ``sigma^2`` has one owner."""
    layer = getattr(sampler.model, "dc_layer", None)
    return (
        layer if isinstance(layer, NoiseAdaptiveDataConsistency) else NoiseAdaptiveDataConsistency()
    )


def prepare_reverse(
    sampler: PhysicsInformedColdDiffusion,
    measurement: Tensor,
    obs: Tensor,
    *,
    eps: float,
) -> ReverseContext:
    """Fit what the options need from ``measurement`` once, and build the clamp maps."""
    from spectramr.models.diffusion.kspace_process import measured_write_ceiling

    physics = sampler.physics
    log_scaled = sampler.kspace_log_scaled
    ceil = sampler._magnitude_ceiling(measurement, obs, eps)
    decay = None
    if physics.decay_envelope != "off":
        pooled, axis = DECAY_ENVELOPE_SHAPES[physics.decay_envelope]
        if axis == LINE_ENVELOPE_AXIS:
            axis = LINE_DECAY_AXES[getattr(sampler.process, "line_axis", None)]
        decay = fit_decay_envelope(
            measurement,
            obs,
            log_scaled=log_scaled,
            pooled=pooled,
            axis=axis,
            noise=noise_estimator(sampler),
            eps=eps,
        )
        ceil = tighten_ceiling(
            ceil, decay, obs, measurement, ratio=sampler.reverse_clip_ratio, log_scaled=log_scaled
        )
        # The mean slope of the envelopes applied, flat and noise-floor ones at 0, so the
        # column is defined on every batch; the unfitted count says how many those are.
        sampler.last_decay_slope = decay.slope.mean()
        sampler.last_decay_unfitted = decay.unfitted().sum()
    trust = None
    if physics.dc_trust == "fitted_decay":
        assert decay is not None, "resolve_reverse_physics_kwargs requires an envelope here"
        trust = build_fitted_trust(decay, measurement, obs, log_scaled=log_scaled)

    # An unconstrained control leaves raw estimates on the measured lines, so bound them too.
    writer = sampler.dc_method if physics.reverse_dc == "per_step" else "unconstrained"
    write_ceil = measured_write_ceiling(
        ceil, measurement, obs.amax(dim=-3, keepdim=True) > 0, writer
    )

    builder = _PROJECTORS[physics.projection]
    projector = None
    if builder is not None:
        projector = (
            _on_transposed_plane(builder(sampler, measurement.mT, obs.mT))
            if _lines_are_columns(sampler)
            else builder(sampler, measurement, obs)
        )
    return ReverseContext(
        ceil=ceil, write_ceil=write_ceil, decay=decay, projector=projector, trust=trust
    )


def observed_dc(
    sampler: PhysicsInformedColdDiffusion,
    x0: Tensor,
    measurement: Tensor,
    obs: Tensor,
    ctx: ReverseContext,
) -> Tensor:
    """Step 1: the configured data consistency, or the fitted-decay Wiener blend."""
    if ctx.trust is None:
        return sampler._apply_observed_dc(x0, measurement, obs)
    return apply_fitted_trust(sampler, ctx.trust, x0, measurement, obs)


def refine_estimate(
    sampler: PhysicsInformedColdDiffusion,
    x0: Tensor,
    measurement: Tensor,
    obs: Tensor,
    ctx: ReverseContext,
    *,
    where: str,
) -> Tensor:
    """Steps 1-3 on a raw model estimate."""
    from spectramr.models.diffusion.kspace_process import inject_reverse_step_noise

    x0 = observed_dc(sampler, x0, measurement, obs, ctx)
    if ctx.projector is not None:
        x0 = ctx.projector(x0, obs, where=where)
    if sampler.sampler_sigma > 0:
        x0 = inject_reverse_step_noise(
            x0, sampler.sampler_sigma, sampler._active_sampler_generator(), exclude_support=obs
        )
    return x0


def bound_estimate(
    sampler: PhysicsInformedColdDiffusion, x0: Tensor, obs: Tensor, ctx: ReverseContext, eps: float
) -> Tensor:
    """Step 4: the decay bin divisor, then the clamp to ``write_ceil``."""
    from spectramr.models.diffusion.kspace_process import clamp_to_magnitude_ceiling

    if ctx.decay is not None:
        x0, _ = bound_bins_by_decay(
            x0,
            ctx.decay,
            obs,
            ratio=sampler.reverse_clip_ratio,
            log_scaled=sampler.kspace_log_scaled,
            eps=eps,
        )
    return clamp_to_magnitude_ceiling(x0, ctx.write_ceil, eps)


__all__ = [
    "ReverseContext",
    "bound_estimate",
    "noise_estimator",
    "observed_dc",
    "prepare_reverse",
    "refine_estimate",
]
