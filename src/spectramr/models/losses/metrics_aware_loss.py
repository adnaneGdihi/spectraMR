"""Metrics-aware Loss Functions - Enhanced loss computation with metrics capture.

This module provides utilities for losses to compute and return metrics during
forward pass, enabling rich loss tracking and analysis during training.

Usage:
    from spectramr.models.losses.metrics_aware_loss import MetricsAwareLoss

    class MyLoss(MetricsAwareLoss):
        def compute_metrics(self, pred, target, loss_value):
            return {
                "mean_pred": pred.mean().item(),
                "mean_target": target.mean().item(),
                "loss": loss_value.item(),
            }

    # Use with metrics
    loss, metrics = my_loss.forward_with_metrics(pred, target)
"""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class IMetricsAwareLoss(ABC):
    """Interface for losses that can compute metrics.

    This protocol defines the contract for loss functions that optionally
    compute and return metrics alongside loss values.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{IMetricsAwareLoss} = \\text{abstract}"""

    @abstractmethod
    def compute_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        loss_value: Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics from prediction and target tensors.

        Args:
            pred: Prediction tensor
            target: Target tensor
            loss_value: Computed loss value
            **kwargs: Additional arguments for metrics computation

        Returns:
            Dictionary mapping metric names to values
        """
        pass

    @abstractmethod
    def forward_with_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss and metrics simultaneously.

        Args:
            pred: Prediction tensor
            target: Target tensor
            **kwargs: Additional loss-specific arguments

        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        pass


class MetricsAwareLossMixin(IMetricsAwareLoss):
    """Mixin to add metrics computation to existing loss functions.

    Provides default implementation of metrics computation and
    forward_with_metrics that can be inherited by loss classes.

    Usage:
        class MyLoss(MetricsAwareLossMixin, nn.Module):
            def forward(self, pred, target):
                return F.l1_loss(pred, target)

            def compute_metrics(self, pred, target, loss_value, **kwargs):
                return {
                    "pred_mean": pred.mean().item(),
                    "pred_std": pred.std().item(),
                }
    """

    def compute_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        loss_value: Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Default metrics computation.

        Subclasses should override for custom metrics.

        Args:
            pred: Prediction tensor
            target: Target tensor
            loss_value: Computed loss value
            **kwargs: Additional arguments

        Returns:
            Dictionary with basic metrics
        """
        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_min": pred.min().item(),
            "pred_max": pred.max().item(),
        }

    def forward_with_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss and metrics.

        Args:
            pred: Prediction tensor
            target: Target tensor
            **kwargs: Additional loss-specific arguments

        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        # The mixin is only ever combined with an nn.Module, which supplies forward.
        result = self.forward(pred, target, **kwargs)  # type: ignore[attr-defined]

        # forward returns (loss, metrics) when the loss was built with compute_metrics=True.
        loss_value: Tensor = result[0] if isinstance(result, tuple) else result

        # Compute metrics
        metrics = self.compute_metrics(pred, target, loss_value, **kwargs)

        return loss_value, metrics


class MetricsTrackerCallback:
    """Utility for tracking metrics across multiple losses.

    Aggregates metrics from multiple loss functions during training.
    """

    def __init__(self, prefix: str = ""):
        """Initialize metrics tracker.

        Args:
            prefix: Optional prefix for all metric names
        """
        self.prefix = prefix
        self.metrics_history: list[dict[str, float]] = []
        self.current_metrics: dict[str, float] = {}

    def update(self, loss_name: str, metrics: dict[str, float]) -> None:
        """Update metrics for a loss.

        Args:
            loss_name: Name of the loss function
            metrics: Metrics dictionary from loss.compute_metrics()
        """
        for key, value in metrics.items():
            metric_key = f"{self.prefix}{loss_name}_{key}" if self.prefix else f"{loss_name}_{key}"
            self.current_metrics[metric_key] = value

    def get_metrics(self) -> dict[str, float]:
        """Get current metrics.

        Returns:
            Current metrics dictionary
        """
        return self.current_metrics.copy()

    def reset(self) -> None:
        """Reset current metrics."""
        self.current_metrics.clear()

    def log_metrics(self, step: int) -> None:
        """Log metrics for current step.

        Args:
            step: Training step number
        """
        self.metrics_history.append(
            {
                "step": step,
                **self.current_metrics.copy(),
            }
        )


# Convenience function for adding metrics to any loss
def add_metrics_to_loss(loss_fn: nn.Module) -> nn.Module:
    """Dynamically add metrics support to a loss function.

    Args:
        loss_fn: Loss function instance

    Returns:
        Loss function with metrics support added
    """

    if isinstance(loss_fn, MetricsAwareLossMixin):
        return loss_fn  # Already has metrics support

    base_cls: type[nn.Module] = type(loss_fn)

    class WrappedLoss(MetricsAwareLossMixin, base_cls):  # type: ignore[valid-type,misc]
        """Wrapped loss with metrics support."""

        def compute_metrics(
            self, pred: Tensor, target: Tensor, loss_value: Tensor, **kwargs: Any
        ) -> dict[str, float]:
            """Default metrics for wrapped loss."""
            return {
                "loss": (
                    loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()
                ),
                "pred_mean": pred.mean().item(),
                "pred_std": pred.std().item(),
                "target_mean": target.mean().item(),
                "target_std": target.std().item(),
            }

    # ``__dict__`` carries the whole module state (parameters, buffers, hooks and the
    # loss's own attributes), so the instance is made without calling ``__init__``,
    # whose arguments this wrapper cannot know.
    wrapped: nn.Module = WrappedLoss.__new__(WrappedLoss)
    wrapped.__dict__.update(loss_fn.__dict__)

    return wrapped
