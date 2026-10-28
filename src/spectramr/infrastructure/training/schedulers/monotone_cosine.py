r"""Cosine learning-rate decay that never rises again.

.. math::

    \eta(t) = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})
              \bigl(1 + \cos(\pi \min(t, T)/T)\bigr)

``torch.optim.lr_scheduler.CosineAnnealingLR`` is periodic in ``t``: past
``T_max`` its recursive update climbs back towards the base rate, and on resume
``load_state_dict`` restores the checkpoint's ``T_max`` over the one the config
resolved. A run resumed under a longer budget therefore restarted its LR at the
old horizon. This schedule is closed-form in ``t``, holds ``eta_min`` past ``T``,
and keeps the configured ``T_max`` and ``eta_min`` when a state is loaded.

The state keys match ``CosineAnnealingLR`` (``T_max``, ``eta_min`` plus the
common ones), so ``check_scheduler_state_compatible`` accepts checkpoints written
by either.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["MonotoneCosineLR"]


class MonotoneCosineLR:
    """Closed-form cosine decay from each group's base LR to ``eta_min``.

    Args:
        optimizer: The optimizer whose ``param_groups`` it drives.
        T_max: Number of scheduler steps over which the LR decays.
        eta_min: Floor reached at ``T_max`` and held afterwards.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,  # noqa: N803 -- CosineAnnealingLR's keyword, which the factory and state keys use
        eta_min: float = 0.0,
    ):
        if int(T_max) <= 0:
            raise ValueError(f"MonotoneCosineLR needs T_max > 0, got {T_max!r}")
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.base_lrs = [float(g.get("initial_lr", g["lr"])) for g in optimizer.param_groups]
        for group, base in zip(optimizer.param_groups, self.base_lrs, strict=True):
            group.setdefault("initial_lr", base)
        self.last_epoch = 0
        self._last_lr = list(self.base_lrs)

    def _lr_at(self, base: float, t: int) -> float:
        progress = min(t, self.T_max) / self.T_max
        return self.eta_min + 0.5 * (base - self.eta_min) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        """Advance one scheduler step and write the new LR into every group."""
        self.last_epoch += 1
        self._last_lr = [self._lr_at(b, self.last_epoch) for b in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, self._last_lr, strict=True):
            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        """The LR of every group after the most recent step."""
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state, key-compatible with ``CosineAnnealingLR``."""
        return {
            "T_max": self.T_max,
            "eta_min": self.eta_min,
            "base_lrs": list(self.base_lrs),
            "last_epoch": self.last_epoch,
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore progress; the configured horizon and floor stay authoritative."""
        saved_t_max = state_dict.get("T_max", self.T_max)
        saved_eta_min = state_dict.get("eta_min", self.eta_min)
        if saved_t_max != self.T_max or saved_eta_min != self.eta_min:
            logger.warning(
                "MonotoneCosineLR: checkpoint T_max=%s eta_min=%s differ from the "
                "configured T_max=%s eta_min=%s; keeping the configured schedule.",
                saved_t_max,
                saved_eta_min,
                self.T_max,
                self.eta_min,
            )
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        if "base_lrs" in state_dict:
            self.base_lrs = [float(b) for b in state_dict["base_lrs"]]
        self._last_lr = [self._lr_at(b, self.last_epoch) for b in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, self._last_lr, strict=True):
            group["lr"] = lr
