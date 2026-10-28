"""Weighted TALL locality term for the reconstruction strategy (Spec §9.3).

``training.tall_sr.lambda_locality`` is validated against ``model_type`` at config
load (``TrainingSettings``), so this module only handles the runtime seam.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Loss-dict key of the weighted locality term; the CSV header must name it too.
TALL_LOCALITY_LOSS_KEY = "loss_locality"


def _unwrap(generator: nn.Module) -> nn.Module:
    return getattr(generator, "module", generator)


def arm_tall_locality(generator: nn.Module, weight: float) -> None:
    """Ask the generator to keep the traversal graph for the coming forward.

    No-op at ``weight == 0``. Must run before the generator forward.

    Raises:
        TypeError: ``weight > 0`` but the generator has no ``track_locality`` switch.
    """
    if weight == 0.0:
        return
    inner = _unwrap(generator)
    if not hasattr(inner, "track_locality"):
        raise TypeError(
            f"training.tall_sr.lambda_locality={weight} but {type(inner).__name__} "
            "has no track_locality switch; refusing to skip the regulariser silently."
        )
    inner.track_locality = True


def tall_locality_term(generator: nn.Module, weight: float) -> torch.Tensor | None:
    """Return ``weight * L_loc`` for the generator's latest traversal, or ``None``.

    ``None`` at ``weight == 0`` (no compute). Must be called right after the
    generator forward, since a second forward overwrites the stashed traversal.

    Raises:
        TypeError: ``weight > 0`` but the generator exposes no ``last_locality_loss``.
    """
    if weight == 0.0:
        return None
    locality_fn = getattr(_unwrap(generator), "last_locality_loss", None)
    if locality_fn is None:
        raise TypeError(
            f"training.tall_sr.lambda_locality={weight} but {type(_unwrap(generator)).__name__} "
            "has no last_locality_loss(); refusing to skip the regulariser silently."
        )
    return weight * locality_fn()


def add_tall_locality(
    total_loss: torch.Tensor,
    losses: dict[str, torch.Tensor],
    locality: torch.Tensor | None,
) -> torch.Tensor:
    """Fold the locality term into ``total_loss`` and record it under its key.

    Returns ``total_loss`` unchanged when there is no term (weight 0).
    """
    if locality is None:
        return total_loss
    losses[TALL_LOCALITY_LOSS_KEY] = locality
    return total_loss + locality


__all__ = [
    "TALL_LOCALITY_LOSS_KEY",
    "add_tall_locality",
    "arm_tall_locality",
    "tall_locality_term",
]
