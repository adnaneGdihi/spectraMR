"""Loss Computer Adapter - SSOT Bridging Pattern

This module provides adapters that wrap existing loss computers to work with
the new BaseLossComputer/LossOutput SSOT pattern without breaking existing code.

Migration Strategy:
- Phase 1: Create adapters (this file)
- Phase 2: Gradually refactor loss computers to inherit from BaseLossComputer
- Phase 3: Remove adapters when all loss computers are refactored
"""

from typing import Any

import torch

from spectramr.models.losses.computers.base import (
    DEFAULT_LOSS_DEVICE,
    BaseLossComputer,
    LossOutput,
)


class LossComputerAdapter(BaseLossComputer):
    """Adapter for existing loss computers (ILossComputer interface).

    Wraps legacy loss computers and converts their dict output to LossOutput format.
    Allows gradual migration without breaking existing code.
    """

    def __init__(
        self,
        legacy_loss_computer: Any,
        config: Any,
        device: torch.device = DEFAULT_LOSS_DEVICE,
    ):
        """Initialize adapter.

        Args:
            legacy_loss_computer: Existing loss computer instance
            config: Configuration object
            device: Compute device
        """
        self.legacy_computer = legacy_loss_computer
        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize - delegated to legacy computer."""
        # Legacy computer already initialized, nothing to do
        pass

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute losses using legacy computer and convert output.

        Args:
            pred: Model prediction
            target: Ground truth
            epoch: Current epoch
            iteration: Current iteration
            **kwargs: Additional arguments (discriminator, etc.)

        Returns:
            LossOutput in SSOT format
        """
        # Call legacy compute_loss (different signature - varies by computer)
        losses_dict = self.legacy_computer.compute_loss(
            lr_reals=kwargs.get("lr_reals"),
            hr_reals=target,
            hr_fakes=pred,
            discriminator_outputs=kwargs.get("discriminator_outputs"),
            epoch=epoch,
            iteration=iteration,
            **kwargs,
        )

        # Convert dict to LossOutput format
        total_loss = losses_dict.pop("g_total_loss", None) or losses_dict.pop("total_loss", None)

        if total_loss is None:
            # No explicit total loss, sum all numeric losses
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for v in losses_dict.values():
                if isinstance(v, torch.Tensor):
                    total_loss = total_loss + v

        # Ensure total loss has requires_grad
        if not total_loss.requires_grad:
            total_loss = total_loss.detach().requires_grad_(True)

        # Separate components from metrics
        components: dict[str, torch.Tensor] = {}
        metrics: dict[str, float | torch.Tensor] = {}

        for key, value in losses_dict.items():
            if isinstance(value, torch.Tensor):
                # Detach for safe logging
                components[key] = value.detach() if value.requires_grad else value
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)

        return LossOutput(total=total_loss, components=components, metrics=metrics)


class DictLossOutputAdapter:
    """Adapter for backward compatibility - convert LossOutput back to dict.

    Some downstream code might still expect dict output from loss computers.
    This helper converts LossOutput to dict format.
    """

    @staticmethod
    def to_dict(loss_output: LossOutput) -> dict[str, torch.Tensor]:
        """Convert LossOutput to dictionary format.

        Args:
            loss_output: LossOutput object

        Returns:
            Dictionary compatible with legacy code
        """
        result = {
            "g_total_loss": loss_output.total,
            "total_loss": loss_output.total,
        }
        result.update(loss_output.components)
        return result

    @staticmethod
    def from_dict(losses_dict: dict[str, torch.Tensor]) -> LossOutput:
        """Convert legacy dict to LossOutput.

        Args:
            losses_dict: Dictionary of losses

        Returns:
            LossOutput object
        """
        total = losses_dict.pop("g_total_loss", None) or losses_dict.pop("total_loss")

        if total is None:
            total = torch.tensor(0.0, requires_grad=True)
        elif not total.requires_grad:
            total = total.detach().requires_grad_(True)

        return LossOutput(total=total, components=losses_dict.copy(), metrics={})


__all__ = [
    "DictLossOutputAdapter",
    "LossComputerAdapter",
]
