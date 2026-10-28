"""The cold-diffusion rung for an acceleration ``R``: its timestep and its mask.

Validation and predict both need "degrade this fully sampled k-space to the rung
that realises ``R``, then sample from that rung". Validation's cascade answered it
inline in the training strategy and predict did not answer it at all: it inferred
the mask from the input's non-zeros and started the reverse loop at ``T - 1``
(#2883). This module is the one answer both call.

Two questions, kept apart because the callers react to their failures differently:

* :func:`rung_for_acceleration` maps ``R`` to a timestep through the schedule's own
  inverse and makes the inverse prove it (:func:`check_round_trip`). It neither logs
  nor skips; it returns a :class:`Rung` whose :class:`RungStatus` the caller turns
  into a skipped cascade level (validation) or an error (predict, where the user
  asked for that ``R`` and a skip would return nothing).
* :func:`rung_mask` builds that timestep's mask through the forward process's
  ``cascade_batch_masks``, counting ``R`` over the lines each row acquired (#2160).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from spectramr.core.cascade_round_trip import RoundTrip, check_round_trip
from spectramr.core.cascading_validation import IDENTITY_ACCELERATION


class RungStatus(StrEnum):
    """Why a requested acceleration does or does not name a usable rung."""

    OK = "ok"
    #: ``R`` is not in the explicit step ladder: a limitation of the arm's config.
    NOT_DECLARED = "not-declared"
    #: The inverse's timestep decodes back to a different ``R`` (saturation, re-pointing).
    ROUND_TRIP_MISMATCH = "round-trip-mismatch"
    #: The rung is the fully sampled identity mask, which measures nothing.
    IDENTITY = "identity"


@dataclass(frozen=True)
class Rung:
    """The schedule's answer for one requested acceleration.

    Attributes:
        status: Whether the rung is usable, and if not, why.
        requested: The acceleration asked for.
        timestep: The inverse's timestep; ``None`` only for ``NOT_DECLARED``.
        realized: What the schedule decodes ``timestep`` back to; ``None`` when
            ``NOT_DECLARED``.
        round_trip: The full verdict, for the caller's log line.
        error: The inverse's ``ValueError`` when ``NOT_DECLARED``.
    """

    status: RungStatus
    requested: float
    timestep: int | None = None
    realized: float | None = None
    round_trip: RoundTrip | None = None
    error: ValueError | None = None

    @property
    def ok(self) -> bool:
        """``True`` when the rung can be sampled from."""
        return self.status is RungStatus.OK


def level_is_declared(accelerator: Any, level: float) -> bool:
    """Is ``level`` a rung the configured step schedule explicitly declares?

    ``timestep_for_acceleration`` raises ``ValueError`` both when the rung is absent
    from ``acceleration_range`` (a limitation of the ladder: skip) and when it is
    declared but realised at no timestep (#1171, a config defect: raise). The two
    are told apart by the accelerator's state, never by the message text (#1303).

    ``ColdDiffusionAccelerator`` delegates to ``self.accelerator`` while the schedule
    state lives on the wrapped instance, so the range is read through that hop.
    Without an explicit range nothing is declared.

    Args:
        accelerator: The schedule-aware accelerator.
        level: The requested acceleration factor.

    Returns:
        True when ``level`` appears in the explicit ``acceleration_range``.
    """
    inner = getattr(accelerator, "accelerator", accelerator)
    if not getattr(inner, "_acceleration_range_explicit", False):
        return False
    declared = getattr(inner, "acceleration_range", None)
    if not isinstance(declared, list):
        return False
    # The schedule's own 1e-6 tolerance, so the two sides agree about membership.
    return any(abs(float(r) - float(level)) < 1e-6 for r in declared)


def rung_for_acceleration(accelerator: Any, acceleration: float, num_timesteps: int) -> Rung:
    """The timestep whose mask realises ``acceleration``, checked against the forward schedule.

    No clamp is applied to the inverse's answer: its contract is a legal timestep
    in ``[0, T - 1]``, and a post-clamp into a training band moved masks off their
    rung while the row kept the requested label (#1295). A continuous schedule
    inverts by clamping its progress, so a request above ``max_acceleration``
    returns ``T - 1``; the round trip is what catches that.

    Args:
        accelerator: Exposes ``timestep_for_acceleration`` and ``get_acceleration_factor``.
        acceleration: The requested acceleration factor.
        num_timesteps: The diffusion horizon ``T``.

    Returns:
        A :class:`Rung`; only ``status is RungStatus.OK`` may be sampled from.

    Raises:
        ValueError: ``acceleration`` is declared in the step ladder but realised at no
            timestep (#1171) -- a config defect the inverse exists to catch.
    """
    requested = float(acceleration)
    try:
        timestep = int(float(accelerator.timestep_for_acceleration(requested)))
    except ValueError as exc:
        if level_is_declared(accelerator, requested):
            raise
        return Rung(RungStatus.NOT_DECLARED, requested, error=exc)

    verdict = check_round_trip(
        requested=requested,
        timestep=timestep,
        forward=accelerator.get_acceleration_factor,
        num_timesteps=num_timesteps,
    )
    if not verdict.ok:
        status = RungStatus.ROUND_TRIP_MISMATCH
    elif verdict.realized <= IDENTITY_ACCELERATION + 1e-6:
        status = RungStatus.IDENTITY
    else:
        status = RungStatus.OK
    return Rung(status, requested, timestep, verdict.realized, verdict)


def schedule_accelerator(process: Any) -> Any:
    """The accelerator the forward process draws its masks from.

    The training strategy borrows this same mask generator from the generator's
    ``kspace_process`` (#2056), so validation and predict invert one schedule.
    """
    return process.mask_generator._get_accelerator(None)


def rung_mask(
    process: Any,
    timesteps: Tensor,
    kspace: Tensor,
    acquired: Tensor | None,
) -> Tensor:
    """The fixed-seed cascade mask at ``timesteps``, float, tiled to ``kspace``'s channels.

    Args:
        process: The generator's forward process (``cascade_batch_masks``).
        timesteps: ``[B]`` long timesteps, one per row of ``kspace``.
        kspace: ``[B, C, H, W]`` k-space the mask will be applied to.
        acquired: ``[B, N]`` bool acquired lines; each rung's ``R`` counts over them.
            ``None`` keeps the full-plane cascade.

    Returns:
        ``[B, C, H, W]`` float mask (an expanded view over the channel axis).
    """
    device = acquired.device if acquired is not None else kspace.device
    plane = (int(kspace.shape[-2]), int(kspace.shape[-1]))
    mask = process.cascade_batch_masks(timesteps, plane, device, acquired=acquired).float()
    if mask.shape[1] == 1 and kspace.shape[1] > 1:
        mask = mask.expand(-1, kspace.shape[1], -1, -1)
    return mask


def acceleration_over_acquired(sampled: Tensor, acquired: Tensor) -> Tensor:
    """``[B]`` acceleration of a mask, counted over each row's acquired lines.

    Args:
        sampled: ``[B, N]`` bool lines the mask keeps.
        acquired: ``[B, N]`` bool lines the scan acquired.

    Returns:
        ``acquired.sum / (sampled & acquired).sum`` per row; ``inf`` for an empty mask.
    """
    acquired = acquired.bool()
    kept = (sampled.bool() & acquired).sum(dim=-1).to(torch.float64)
    return acquired.sum(dim=-1).to(torch.float64) / kept


__all__ = [
    "Rung",
    "RungStatus",
    "acceleration_over_acquired",
    "level_is_declared",
    "rung_for_acceleration",
    "rung_mask",
    "schedule_accelerator",
]
