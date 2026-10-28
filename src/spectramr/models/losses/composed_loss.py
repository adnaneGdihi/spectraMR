"""Enhanced composite loss with metrics support.

Simplifies loss composition with metrics tracking and flexible weighting.
"""

from typing import Any, Self

import torch
import torch.nn as nn

from spectramr.models.losses.metrics_aware_loss import MetricsAwareLossMixin


class WeightedLoss:
    r"""Container for a loss function with weight.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Weighted} = w \mathcal{L}(\hat{y}, y)"""

    def __init__(self, name: str, loss_fn: nn.Module, weight: float = 1.0):
        """Initialize weighted loss.

        Args:
            name: Name for logging (e.g., "l1", "perceptual")
            loss_fn: Loss function module
            weight: Weight multiplier for this loss
        """
        self.name = name
        self.loss_fn = loss_fn
        self.weight = weight

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        want_metrics: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        """Compute loss, and metrics only when the caller asked for them.

        ``want_metrics`` is keyword-only on purpose: ``**kwargs`` is forwarded
        verbatim to ``loss_fn``, so a positional or plain keyword parameter
        could be swallowed by (or collide with) that forwarding.

        The default is ``False`` because the training forward pass does not use
        the metrics. Taking the ``forward_with_metrics`` branch unconditionally
        made every component run ``compute_metrics``, which is ~7
        device-to-host ``.item()`` syncs per component per step -- non-negotiable
        9 forbids exactly that in the training loop -- and the metrics it
        produced were then discarded by the caller (``ComposedLoss.forward``).

        Args:
            pred: Prediction tensor.
            target: Ground-truth tensor.
            want_metrics: When ``True``, use the component's
                ``forward_with_metrics`` path if it has one. Callers that
                actually consume the metrics dict (``compute_metrics``) must
                pass ``True``; the forward path must not.
            **kwargs: Forwarded to the wrapped loss function.

        Returns:
            ``(weighted_loss_value, metrics_dict or None)``. The metrics dict is
            ``None`` whenever ``want_metrics`` is ``False`` or the component has
            no ``forward_with_metrics``.
        """
        metrics: dict[str, float] | None = None
        # ``Any``: ``forward_with_metrics`` is duck-typed, and ``nn.Module``'s
        # ``__getattr__`` would type the bound method as a ``Tensor``.
        loss_fn: Any = self.loss_fn
        if want_metrics and hasattr(loss_fn, "forward_with_metrics"):
            loss, metrics = loss_fn.forward_with_metrics(pred, target, **kwargs)
        else:
            loss = loss_fn(pred, target, **kwargs)

        # Single unwrap owner for both branches. A loss constructed with
        # `compute_metrics=True` (SSIMLoss, LPIPSLoss, PerceptualLoss, L1Loss
        # and MSELoss all take that flag) returns `(loss, metrics)` from
        # `forward`. `forward_with_metrics` unwraps that internally, so before
        # this gate the tuple could never reach the multiply. Now that the plain
        # branch is reachable for those losses it must unwrap too: `tuple *
        # float` raises TypeError, and `tuple * int` silently returns a repeated
        # tuple rather than a loss.
        if isinstance(loss, tuple):
            loss = loss[0]

        weighted: torch.Tensor = loss * self.weight
        return weighted, metrics


def _zero_total(pred: torch.Tensor) -> torch.Tensor:
    """The accumulator seed: ``pred``'s device, and its REAL dtype.

    ``pred.dtype`` itself made the total complex for complex MRI data, and a complex
    scalar cannot be backpropagated. For every real dtype the two are identical.
    """
    return torch.zeros((), device=pred.device, dtype=pred.dtype.to_real())


class ComposedLoss(nn.Module, MetricsAwareLossMixin):
    r"""Simplified composite loss combining multiple weighted losses with metrics.

    Example:
        ```python
        l1_loss = L1Loss()
        perceptual_loss = PerceptualLoss()

        composed = ComposedLoss([
            WeightedLoss("l1", l1_loss, weight=10.0),
            WeightedLoss("perceptual", perceptual_loss, weight=0.1),
        ])

        pred = torch.randn(4, 3, 64, 64)
        target = torch.randn(4, 3, 64, 64)

        total_loss, metrics = composed.forward_with_metrics(pred, target)
        # metrics = {
        #     "loss": total_value,
        #     "l1_loss": l1_value,
        #     "l1_error_mean": ...,
        #     "perceptual_loss": perc_value,
        #     ...
        # }
        ```
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Composed} = \sum_{i} \mathcal{L}_i(\hat{y}, y)"""

    def __init__(self, losses: list[WeightedLoss]):
        """Initialize composite loss.

        Args:
            losses: List of WeightedLoss objects
        """
        super().__init__()
        self.losses = losses

        # Register loss modules with nn.ModuleList for proper state management
        self.loss_modules = nn.ModuleList([loss.loss_fn for loss in losses])

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute total loss (sum of weighted components).

        Args:
            pred: Model predictions
            target: Ground truth targets
            **kwargs: Additional arguments passed to loss functions

        Returns:
            Total weighted loss

        forward method for ComposedLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        total_loss = _zero_total(pred)

        for loss_obj in self.losses:
            loss_val, _ = loss_obj.compute(pred, target, **kwargs)
            total_loss = total_loss + loss_val

        return total_loss

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics from all component losses.

        Returns:
            Dictionary with aggregated metrics from all losses
        """
        metrics = {"loss": loss_value.item()}

        for loss_obj in self.losses:
            loss_val, loss_metrics = loss_obj.compute(pred, target, want_metrics=True, **kwargs)

            # Add component loss value
            metrics[f"{loss_obj.name}_loss"] = loss_val.item()

            # Add component metrics with loss name prefix
            if loss_metrics:
                for metric_name, metric_val in loss_metrics.items():
                    prefixed_name = f"{loss_obj.name}_{metric_name}"
                    metrics[prefixed_name] = metric_val

        return metrics

    def forward_with_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass returning (loss, metrics).

        Args:
            pred: Model predictions
            target: Ground truth targets
            **kwargs: Additional arguments passed to loss functions

        Returns:
            (loss_value, metrics_dict)
        """
        total_loss = self.forward(pred, target, **kwargs)
        metrics = self.compute_metrics(pred, target, total_loss, **kwargs)
        return total_loss, metrics

    def update_weights(self, loss_name: str, new_weight: float) -> None:
        """Update weight for a specific loss component.

        Args:
            loss_name: Name of loss to update
            new_weight: New weight value

        Raises:
            ValueError: If loss_name not found
        """
        for loss_obj in self.losses:
            if loss_obj.name == loss_name:
                loss_obj.weight = new_weight
                return

        raise ValueError(f"Loss '{loss_name}' not found in composite loss")

    def get_weights(self) -> dict[str, float]:
        """Get current weights for all loss components.

        Returns:
            Dictionary mapping loss name to weight
        """
        return {loss.name: loss.weight for loss in self.losses}

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move all loss modules to device.

        Overrides nn.Module.to() to ensure all sub-losses move too.
        """
        super().to(*args, **kwargs)
        for loss in self.loss_modules:
            if hasattr(loss, "to"):
                loss.to(*args, **kwargs)
        return self


class ConditionalComposedLoss(ComposedLoss):
    r"""Composed loss where components can be conditionally enabled.

    Allows turning loss components on/off during training.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{ConditionalComposed} = \sum_{i} m_i \mathcal{L}_i(\hat{y}, y)"""

    def __init__(self, losses: list[WeightedLoss]):
        """Initialize conditional composed loss.

        Args:
            losses: List of WeightedLoss objects
        """
        super().__init__(losses)
        self.enabled = {loss.name: True for loss in losses}

    def disable(self, loss_name: str) -> None:
        """Disable a loss component.

        Args:
            loss_name: Name of loss to disable

        Raises:
            ValueError: If loss_name not found
        """
        if loss_name not in self.enabled:
            raise ValueError(f"Loss '{loss_name}' not found")
        self.enabled[loss_name] = False

    def enable(self, loss_name: str) -> None:
        """Enable a loss component.

        Args:
            loss_name: Name of loss to enable

        Raises:
            ValueError: If loss_name not found
        """
        if loss_name not in self.enabled:
            raise ValueError(f"Loss '{loss_name}' not found")
        self.enabled[loss_name] = True

    def is_enabled(self, loss_name: str) -> bool:
        """Check if loss component is enabled.

        Args:
            loss_name: Name of loss to check

        Returns:
            True if enabled, False otherwise
        """
        return self.enabled.get(loss_name, False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute total loss (sum of enabled weighted components).

        Args:
            pred: Model predictions
            target: Ground truth targets
            **kwargs: Additional arguments passed to loss functions

        Returns:
            Total weighted loss (enabled components only)
        """
        total_loss = _zero_total(pred)

        for loss_obj in self.losses:
            if self.is_enabled(loss_obj.name):
                loss_val, _ = loss_obj.compute(pred, target, **kwargs)
                total_loss = total_loss + loss_val

        return total_loss

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics from enabled component losses.

        Returns:
            Dictionary with aggregated metrics from enabled losses
        """
        metrics = {"loss": loss_value.item()}

        for loss_obj in self.losses:
            if self.is_enabled(loss_obj.name):
                loss_val, loss_metrics = loss_obj.compute(pred, target, want_metrics=True, **kwargs)

                # Add component loss value
                metrics[f"{loss_obj.name}_loss"] = loss_val.item()

                # Add component metrics with loss name prefix
                if loss_metrics:
                    for metric_name, metric_val in loss_metrics.items():
                        prefixed_name = f"{loss_obj.name}_{metric_name}"
                        metrics[prefixed_name] = metric_val

        return metrics
