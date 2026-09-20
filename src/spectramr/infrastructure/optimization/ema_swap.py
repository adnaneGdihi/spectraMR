"""Hold EMA shadow weights in a live module for the duration of a block.

Validation does not forward the EMA object. It copies the shadow weights
**into the live generator in place**, forwards the generator, and copies the
originals back — so the swap mutates the model that training is about to
continue from, and every failure mode here is a training failure, not just a
metrics one.

Three of them were reachable (#2172):

* **The mutation outranged its own ``try``.** In ``_run_validation`` the swap
  ran ~66 lines above the ``with`` that restored it, with no enclosing ``try``
  in the function. Anything raising in between — a config read,
  ``resolve_axes_for``, an ``inspect.signature`` probe — left the generator
  holding shadow weights with no restore, and training resumed from them.
  Owning both halves in one context manager is what makes that unrepresentable.
* **A partial swap was reported as an EMA run.** ``load_state_dict(...,
  strict=False)`` applies only the keys it is handed and leaves the rest live,
  so the metrics grade a blend. Both halves have to be counted: a key whose
  shape clashes and a key the target does not have are equally unapplied, and
  only the first leaves any trace. Counting shape clashes alone made a
  1-of-100 overlap read as a clean swap, which is the #2172 shape one size
  down from the total miss the guard below already caught.
* **A partial restore was permanent and silent.** A tensor whose shape changed
  during the validation forward cannot accept its saved value, so it keeps the
  shadow weight — and ``strict=False`` accepts that without a word.

The shape filters themselves are deliberate, not defensive padding: a
``channel_adapter`` is legitimately rebuilt mid-run, which is exactly why a
*partial* result has to stay allowed while a *total* mismatch raises.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from typing import Any

import torch.nn as nn

from spectramr.infrastructure.optimization.ema import (
    EMAKeyMismatchError,
    EMAWeightRestoreError,
)

logger = logging.getLogger(__name__)

__all__ = ["ema_weights_swapped_in"]


def _capture(target: nn.Module) -> dict[str, Any]:
    """Independent CPU copies of *target*'s tensors.

    ``state_dict()`` hands back references to the live tensors and
    ``load_state_dict`` copies in place, so a capture that skipped the clone
    would be overwritten by the very swap it exists to undo. CPU keeps a third
    full copy of the weights out of VRAM; ``load_state_dict`` moves them back.
    """
    return {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}


def _shape_compatible(
    source: dict[str, Any], destination: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Split *source* into what *destination* accepts, refuses, and lacks.

    The third list is the one that is easy to drop and expensive to lose. A key
    *absent* from the destination is as unapplied as one whose shape clashes,
    but it leaves no trace: ``load_state_dict(..., strict=False)`` takes the
    dict it is given without remarking on what is not in it. Counting only the
    shape clashes makes a 1-of-100 overlap indistinguishable from a full swap.
    """
    accepted: dict[str, Any] = {}
    rejected: list[str] = []
    missing: list[str] = []
    for key, value in source.items():
        if key not in destination:
            missing.append(key)
        elif destination[key].shape == value.shape:
            accepted[key] = value
        else:
            rejected.append(key)
    return accepted, rejected, missing


@contextlib.contextmanager
def ema_weights_swapped_in(target: nn.Module, shadow: nn.Module) -> Iterator[None]:
    """Run the block with *shadow*'s weights installed in *target*.

    Args:
        target: the live module to mutate — **already unwrapped**. The shadow
            carries bare keys, so a DDP/FSDP/DeepSpeed/compile wrapper here
            would make every key miss and the swap a silent no-op.
        shadow: the EMA shadow module (``ModelEma.module``).

    Raises:
        EMAKeyMismatchError: no shadow tensor matched — the swap would grade
            the live weights while reporting EMA.
        EMAWeightRestoreError: the block changed a tensor's shape, so the
            original cannot be put back and training would continue from a
            blend. Suppressed when an exception is already propagating, so the
            real cause is not replaced by this one.
    """
    original = _capture(target)
    shadow_state = shadow.state_dict()
    target_state = target.state_dict()

    to_apply, kept_live, absent = _shape_compatible(shadow_state, target_state)
    unapplied = kept_live + absent

    if shadow_state and not to_apply:
        raise EMAKeyMismatchError(
            f"EMA swap matched 0 of {len(shadow_state)} shadow tensors against the target, "
            f"so validation would grade the live weights while reporting them as EMA. "
            f"Shadow key example: {next(iter(shadow_state), '<empty>')!r}; target key "
            f"example: {next(iter(target_state), '<empty>')!r}. Pass an unwrapped module."
        )
    if unapplied:
        logger.warning(
            "[VAL] EMA swap is PARTIAL: %d of %d shadow tensors applied; %d kept live on a "
            "shape clash and %d absent from the target. Reported EMA metrics grade a blend, "
            "not the shadow. First unapplied: %r",
            len(to_apply),
            len(shadow_state),
            len(kept_live),
            len(absent),
            unapplied[0],
        )

    target.load_state_dict(to_apply, strict=False)
    logger.info("[VAL] Generator weights temporarily replaced with the EMA shadow.")

    try:
        yield
    finally:
        restorable, changed_shape, vanished = _shape_compatible(original, target.state_dict())
        target.load_state_dict(restorable, strict=False)
        unrestored = changed_shape + vanished

        if not unrestored:
            logger.info("[VAL] Restored generator weights after the EMA swap.")
        elif sys.exc_info()[0] is not None:
            logger.error(
                "[VAL] %d tensor(s) could not be restored after the EMA swap; the generator "
                "keeps shadow weights for them. Reporting the in-flight failure instead.",
                len(unrestored),
            )
        else:
            raise EMAWeightRestoreError(
                f"{len(unrestored)} of {len(original)} tensor(s) could not be restored after "
                f"the EMA swap ({len(changed_shape)} changed shape during the block, "
                f"{len(vanished)} no longer exist on the target), so the generator keeps "
                f"SHADOW weights for them and training would continue from a blend. "
                f"First unrestored: {unrestored[0]!r}."
            )
