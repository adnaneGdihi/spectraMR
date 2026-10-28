"""Base Loss Module
==================

Provides a base class for loss functions to standardize
input validation and reduction operations.
"""

from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class IComputableLoss(Protocol):
    r"""Protocol for loss functions.

    All loss functions should implement this interface to ensure consistent
    behavior across the framework. The forward method must return a dict
    with at least a 'loss' key containing the primary loss tensor.

    Example:
        >>> class MyLoss(nn.Module, IComputableLoss):
        ...     def forward(self, pred, target, **ctx):
        ...         l1 = F.l1_loss(pred, target)
        ...         return {"loss": l1, "l1": l1}
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{IComputableLoss} = \text{abstract}"""

    def forward(self, pred: Tensor, target: Tensor, **context: Any) -> dict[str, Tensor]:
        """
        Compute the loss.

        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            **context: Additional context (e.g., mask, weight, iteration)

        Returns:
            Dict with 'loss' key (primary loss) and optional auxiliary losses.
        """
        ...


class BaseLoss(nn.Module):
    r"""Base class for loss functions with common validation and reduction logic.

    This class provides a standardized structure for loss functions, handling
    common boilerplate such as input tensor validation (shape, device) and
    applying reduction operations ('mean', 'sum', 'none').

    Child classes should implement the `forward` method to compute the raw,
    unreduced loss.

    Attributes:
        reduction (str): The reduction method to apply to the loss.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{BaseLoss} = \text{abstract}"""

    def __init__(self, reduction: str = "mean"):
        """Initialize the BaseLoss.

        Args:
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed.
                Default: 'mean'.

        Raises:
            ValueError: If an invalid reduction type is provided.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction

    def _validate_inputs(self, pred: Tensor, target: Tensor) -> None:
        """Validate that prediction and target tensors are compatible.

        Checks for shape and device consistency.

        Args:
            pred (Tensor): The predicted tensor.
            target (Tensor): The ground truth tensor.

        Raises:
            ValueError: If shapes or devices do not match.
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        if pred.device != target.device:
            raise ValueError(f"Device mismatch: pred {pred.device} vs target {target.device}")

    def _apply_reduction(self, loss: Tensor) -> Tensor:
        """Apply the configured reduction to the loss tensor.

        Args:
            loss (Tensor): The computed loss tensor.

        Returns:
            Tensor: The reduced loss.
        """

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Forward pass for the loss function.

        This method should be implemented by subclasses to compute the
        actual loss value. It is expected to call `_validate_inputs` and
        `_apply_reduction`.

        Args:
            pred (Tensor): The predicted tensor.
            target (Tensor): The ground truth tensor.

        Returns:
            Tensor: The computed loss.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        forward method for BaseLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # TODO: Subclasses must implement this method
        raise NotImplementedError("Subclasses must implement the forward method.")
