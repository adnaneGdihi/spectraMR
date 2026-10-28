"""Integrity verifier for the cold-diffusion reverse loops.

Every reverse loop promises a small set of structural facts about the k-space
state it hands to the next model call. When one of them breaks, nothing
downstream fails: the loop keeps running, the output is still a tensor of the
right shape, and the defect shows up only as a washed-out or striped validation
image. This module states the facts once and checks them after every write,
under ``model.model_kwargs.reverse_integrity_check``:

``off``
    no check runs (the default, so the sampler's hot path is unchanged).
``raise``
    each write is verified and the first violation raises
    :class:`ReverseIntegrityError` naming the invariant, the step and the
    timestep.

The invariants, and the loops that promise them:

``shape``
    the state keeps the measurement's shape, and a real-stacked state keeps an
    even channel count (interleaved Re/Im pairs). All loops.
``finite``
    no NaN or Inf. All loops.
``observed_pinned``
    under hard data consistency the measured coefficients equal the
    measurement. All loops when ``dc_method == 'hard'``.
``magnitude_ceiling``
    no unmeasured coefficient exceeds the per-sample ceiling. Every loop except
    ``additive``, which has no bound; enabling the check there reports the
    first step that overshoots instead of letting it compound.
``partition``
    a step reveals only coefficients neither measured nor already committed.
    The loops that reveal through ``_reveal_support``.
``coverage``
    after the last step every coefficient is measured or written. The same
    loops, plus ``additive_normalized``.
``observed_discrepancy``
    with ``reverse_dc_trust: fitted_decay``, every measured coefficient lies
    within ``3 sigma`` of the measurement in linear k-space. Replaces
    ``observed_pinned``, which a Wiener trust does not promise.
``projection_residual``
    the objective a ``reverse_projection`` minimises -- the coil-consistency
    residual, or the joint-TV objective -- does not increase across its inner
    iterations.

The verifier never corrects the state. Rescaling is owned by the adjuster
(``reverse_envelope``) and the clamp; a checker that also repaired would hide
the violation it exists to report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spectramr.models.diffusion.reverse_envelope import per_coefficient_support

if TYPE_CHECKING:
    from torch import Tensor

    from spectramr.models.diffusion.kspace_process import PhysicsInformedColdDiffusion

#: Values of ``model.model_kwargs.reverse_integrity_check``.
VALID_INTEGRITY_CHECKS: frozenset[str] = frozenset({"off", "raise"})

#: Absolute slack on the observed pin and the ceiling, above float32 round-off
#: of the clamp's ``ceil / |z|`` rescale.
_ABS_TOL = 1e-5
_REL_TOL = 1e-4


class ReverseIntegrityError(RuntimeError):
    """A reverse step broke one of the invariants listed in this module."""

    def __init__(self, invariant: str, detail: str) -> None:
        super().__init__(f"reverse integrity [{invariant}]: {detail}")
        self.invariant = invariant


def validate_integrity_check(value: object) -> str:
    """Resolve ``reverse_integrity_check``; ``None`` means ``off``, anything unknown raises."""
    if value is None:
        return "off"
    if not isinstance(value, str) or value not in VALID_INTEGRITY_CHECKS:
        raise ValueError(
            f"reverse_integrity_check={value!r} is not one of {sorted(VALID_INTEGRITY_CHECKS)}."
        )
    return value


def verify_state(
    x: Tensor,
    measurement: Tensor,
    obs: Tensor,
    *,
    ceil: Tensor | None,
    dc_hard: bool,
    where: str,
) -> None:
    """Check ``shape``, ``finite``, ``observed_pinned`` and ``magnitude_ceiling`` on ``x``."""
    from spectramr.models.diffusion.kspace_process import paired_magnitude

    if x.shape != measurement.shape:
        raise ReverseIntegrityError(
            "shape", f"{where}: state {tuple(x.shape)} != measurement {tuple(measurement.shape)}"
        )
    if not torch.is_complex(x) and x.shape[-3] % 2 != 0:
        raise ReverseIntegrityError(
            "shape", f"{where}: odd channel count {x.shape[-3]} breaks the Re/Im pairing"
        )
    if not bool(torch.isfinite(x).all()):
        raise ReverseIntegrityError("finite", f"{where}: state holds NaN or Inf")
    if dc_hard:
        drift = ((x - measurement).abs() * obs).amax()
        if bool(drift > _ABS_TOL):
            raise ReverseIntegrityError(
                "observed_pinned", f"{where}: measured coefficient moved by {float(drift):.3e}"
            )
    if ceil is not None:
        unmeasured = 1.0 - (per_coefficient_support(obs, x) > 0).to(ceil.dtype)
        excess = (paired_magnitude(x) - ceil * (1.0 + _REL_TOL)) * unmeasured
        worst = excess.amax()
        if bool(worst > _ABS_TOL):
            raise ReverseIntegrityError(
                "magnitude_ceiling",
                f"{where}: a written coefficient exceeds the ceiling by {float(worst):.3e}",
            )


def verify_partition(reveal: Tensor, committed: Tensor, obs: Tensor, *, where: str) -> None:
    """``reveal`` must be disjoint from both the committed and the measured support."""
    if bool((reveal * committed).any()):
        raise ReverseIntegrityError("partition", f"{where}: revealed an already-committed bin")
    if bool((reveal * obs).any()):
        raise ReverseIntegrityError("partition", f"{where}: revealed a measured bin")


def verify_coverage(written: Tensor, obs: Tensor, *, where: str) -> None:
    """Every coefficient is measured or written once the loop ends."""
    missing = ((written + obs) <= 0).sum()
    if bool(missing > 0):
        raise ReverseIntegrityError(
            "coverage", f"{where}: {int(missing)} coefficient(s) never written or measured"
        )


def verify_observed_discrepancy(
    x: Tensor, measurement: Tensor, obs: Tensor, radius: Tensor, *, log_scaled: bool, where: str
) -> None:
    """Every measured coefficient of ``x`` lies within ``radius`` ``[B, 1, 1, 1]`` of ``y``."""
    from spectramr.models.diffusion.reverse_domain import linear_kspace

    lin_x = linear_kspace(x, log_scaled=log_scaled)
    lin_y = linear_kspace(measurement, log_scaled=log_scaled)
    measured = per_coefficient_support(obs, x) > 0
    slack = radius * (1.0 + _REL_TOL) + _REL_TOL * lin_y.abs() + _ABS_TOL
    excess = ((lin_x - lin_y).abs() - slack) * measured
    worst = excess.amax()
    if bool(worst > 0):
        raise ReverseIntegrityError(
            "observed_discrepancy",
            f"{where}: a measured coefficient sits {float(worst):.3e} beyond its trust radius",
        )


def verify_monotone(values: list[Tensor], *, invariant: str, where: str, rel_tol: float) -> None:
    """Each per-sample value in ``values`` is at most its predecessor, up to ``rel_tol``."""
    seq = torch.stack([v.detach().float() for v in values])
    rise = seq[1:] - seq[:-1] - rel_tol * seq[:-1].abs() - _ABS_TOL
    if bool((rise > 0).any()):
        at = int((rise > 0).any(dim=1).nonzero()[0])
        raise ReverseIntegrityError(
            invariant, f"{where}: rose from {seq[at].tolist()} to {seq[at + 1].tolist()} at iteration {at + 1}"
        )


def check_step(
    sampler: PhysicsInformedColdDiffusion,
    x: Tensor,
    measurement: Tensor,
    obs: Tensor,
    *,
    step: int,
    t: int,
    ceil: Tensor | None,
    reveal: Tensor | None = None,
    committed: Tensor | None = None,
    trust_radius: Tensor | None = None,
) -> None:
    """Verify one write of ``sampler``'s loop when its check is on; count it."""
    if sampler.integrity_check == "off":
        return
    where = f"{sampler.reverse_mode} step {step} (t={t})"
    if reveal is not None and committed is not None:
        verify_partition(reveal, committed, obs, where=where)
    pinned = sampler.dc_method == "hard" and sampler.physics.reverse_dc != "unconstrained"
    verify_state(x, measurement, obs, ceil=ceil, dc_hard=pinned, where=where)
    if trust_radius is not None:
        verify_observed_discrepancy(
            x, measurement, obs, trust_radius, log_scaled=sampler.kspace_log_scaled, where=where
        )
    sampler.last_integrity_steps_verified += 1


def check_monotone(
    sampler: PhysicsInformedColdDiffusion,
    values: list[Tensor],
    *,
    invariant: str,
    where: str,
    rel_tol: float = _REL_TOL,
) -> None:
    """Verify a non-increasing inner-iteration sequence when ``sampler``'s check is on."""
    if sampler.integrity_check == "off":
        return
    verify_monotone(values, invariant=invariant, where=where, rel_tol=rel_tol)


def check_coverage(sampler: PhysicsInformedColdDiffusion, written: Tensor, obs: Tensor) -> None:
    """Verify end-of-loop coverage for ``sampler`` when its check is on."""
    if sampler.integrity_check == "off":
        return
    # A never-acquired line is owned by the measurement's zero, not left unwritten.
    acquirable = sampler.acquirable_plane(tuple(written.shape[-2:]))
    if acquirable is not None:
        obs = torch.clamp(obs + (1.0 - acquirable.to(obs.device, obs.dtype)), max=1.0)
    verify_coverage(written, obs, where=f"{sampler.reverse_mode} after the last step")


__all__ = [
    "VALID_INTEGRITY_CHECKS",
    "ReverseIntegrityError",
    "check_coverage",
    "check_monotone",
    "check_step",
    "validate_integrity_check",
    "verify_coverage",
    "verify_monotone",
    "verify_observed_discrepancy",
    "verify_partition",
    "verify_state",
]
