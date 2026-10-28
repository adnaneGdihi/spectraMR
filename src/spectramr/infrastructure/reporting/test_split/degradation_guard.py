"""Refuse a test pass whose model input is not degraded the way the arm was trained.

A model trained on R=4 measurements and handed fully sampled data scores the identity of a
clean image, not a reconstruction: every number it reports is poisoned. The guard reads the
first test batch once (one host sync per pass) and names the violation; the test pass raises
on it before any result is written.

The realised rate is the one the loader stamped (``realised_acceleration``, owned by the masking
transforms). It counts the calibration lines, so it sits below the declared rate by design; the
guard therefore flags only a fully sampled input, never a rate that is merely lower.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from spectramr.data.transforms.tio_physics import REALISED_ACCELERATION_KEY

#: A realised rate at or below this is a fully sampled input.
FULLY_SAMPLED_RATE = 1.01


def declared_acceleration(config: Any) -> float | None:
    """The ``undersampling`` block's rate, or ``None`` when it declares no undersampling."""
    block = getattr(config, "undersampling", None)
    if block is None or block.declares_no_acceleration:
        return None
    rate = float(block.base_acceleration)
    return rate if rate > 1.0 else None


def _field(batch: Any, key: str) -> Any:
    getter = getattr(batch, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except (KeyError, TypeError):
            return None
    metadata = getattr(batch, "metadata", None)
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return getattr(batch, key, None)


def realised_rates(batch: Any) -> tuple[torch.Tensor | None, bool]:
    """Per-sample realised acceleration ``(B,)``, and whether the loader stamped it.

    The loader's stamp when the batch carries one; otherwise ``1 / sampled fraction`` of the
    batch's mask (a strategy-side mask has no stamp); ``None`` with neither.
    """
    stamped = _field(batch, REALISED_ACCELERATION_KEY)
    if stamped is not None:
        return torch.as_tensor(stamped, dtype=torch.float32).reshape(-1), True
    mask = _field(batch, "mask")
    if not isinstance(mask, torch.Tensor) or mask.ndim == 0:
        return None, False
    m = mask.detach()
    m = m.abs() if m.is_complex() else m
    fraction = (m.float() != 0).flatten(1).float().mean(dim=1)
    return 1.0 / fraction.clamp_min(1e-12), False


@dataclass(frozen=True)
class DegradationGuard:
    """What the arm declares; :meth:`verdict` says whether a batch honours it."""

    declared: float | None

    @classmethod
    def from_config(cls, config: Any) -> DegradationGuard:
        return cls(declared_acceleration(config))

    def verdict(
        self,
        input_view: torch.Tensor,
        target_view: torch.Tensor,
        rates: torch.Tensor | None,
        *,
        stamped: bool = False,
    ) -> str | None:
        """The violation on this batch, or ``None``.

        ``stamped`` says ``rates`` came from the loader, which stamps only when it masked: a
        stamped rate of 1 is a fully sampled input whatever the arm declares. A mask-derived
        rate of 1 is a violation only for an arm that declares undersampling.
        """
        if input_view.shape == target_view.shape and torch.allclose(
            input_view.float(), target_view.float(), rtol=1e-4, atol=1e-6
        ):
            return (
                "the test input equals the target: the model is graded on clean data it was "
                "never trained to receive, so every test metric would be poisoned"
            )
        if rates is None or not (stamped or self.declared is not None):
            return None
        lowest = float(rates.min())
        if lowest <= FULLY_SAMPLED_RATE:
            declared = f"R={self.declared:g}" if self.declared else "a masked acquisition"
            return (
                f"the test input is fully sampled (realised R={lowest:.2f}) while the arm "
                f"was trained on {declared}: the test pass is not applying the training "
                "degradation"
            )
        return None


__all__ = [
    "FULLY_SAMPLED_RATE",
    "DegradationGuard",
    "declared_acceleration",
    "realised_rates",
]
