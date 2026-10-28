"""Loss Validation Utilities
===========================

Provides universal validation decorators and mixins for loss functions
to ensure shape compatibility and numerical stability.

CRITICAL: All loss functions should use these utilities to prevent
silent dimension mismatches and gradient instability.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _strip_trailing_singletons(t: Tensor, min_ndim: int = 3) -> Tensor:
    """Drop trailing size-1 axes (TorchIO wrapping artifact).

    Strips trailing ``size == 1`` dimensions so a model output of shape
    ``[B, C, L]`` aligns with a TorchIO target of ``[B, C, L, 1, 1]``. Only
    trailing unit axes are removed and never below ``min_ndim`` (keep at
    least ``[B, C, *]``), so real spatial extent is preserved. Non-tensors
    and tensors with no trailing singletons pass through unchanged.
    """
    if not isinstance(t, Tensor):
        return t
    while t.ndim > min_ndim and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t


def validate_loss_inputs[F: Callable[..., Any]](forward_method: F) -> F:
    """Decorator to validate loss function inputs.

    Wraps a loss forward method with automatic shape and device validation.
    Raises ValueError if pred and target have different shapes or devices.

    Usage:
        class MyLoss(nn.Module):
            @validate_loss_inputs
            def forward(self, pred: Tensor, target: Tensor) -> Tensor:
                ...

    Args:
        forward_method: The forward method to wrap.

    Returns:
        Wrapped forward method with validation.
    """

    @functools.wraps(forward_method)
    def wrapper(self: Any, pred: Tensor, target: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Shape/device-check ``pred`` and ``target``, then call the wrapped forward."""
        # TorchIO pads 1-D/2-D data with trailing unit axes that shape-rigid models
        # squeeze from their input; stripping them is numel-preserving, and a real
        # extent mismatch still raises below.
        pred = _strip_trailing_singletons(pred)
        target = _strip_trailing_singletons(target)
        if pred.shape != target.shape:
            raise ValueError(
                f"[{self.__class__.__name__}] Shape mismatch: "
                f"pred {pred.shape} vs target {target.shape}. "
                f"Ensure model output matches target dimensions."
            )

        # Device validation
        if pred.device != target.device:
            raise ValueError(
                f"[{self.__class__.__name__}] Device mismatch: "
                f"pred on {pred.device} vs target on {target.device}. "
                f"Move tensors to same device before loss computation."
            )

        # NaN diagnostics are gated on DEBUG logging: evaluating
        # ``torch.isnan(t).any()`` in a Python ``if`` forces a host-device
        # sync plus a full pass over the tensor, per loss call, on the
        # training hot path (performance.md). The always-on NaN guard lives
        # in the trainer's progress reporting, which piggybacks on an
        # existing sync. Enable DEBUG on this logger to turn these per-call
        # diagnostics on when chasing a numerical instability.
        debug_diagnostics = logger.isEnabledFor(logging.DEBUG)
        if debug_diagnostics:
            if torch.isnan(pred).any():
                logger.warning(
                    f"[{self.__class__.__name__}] NaN detected in prediction tensor. "
                    f"Check model forward pass for numerical instability."
                )
            if torch.isnan(target).any():
                logger.warning(
                    f"[{self.__class__.__name__}] NaN detected in target tensor. "
                    f"Check data loading pipeline."
                )

        # Call original forward
        result = forward_method(self, pred, target, *args, **kwargs)

        # NaN check on output (DEBUG-gated, same sync-cost rationale)
        if debug_diagnostics and isinstance(result, Tensor) and torch.isnan(result).any():
            pred_r = pred.abs() if pred.is_complex() else pred
            tgt_r = target.abs() if target.is_complex() else target
            logger.error(
                f"[{self.__class__.__name__}] NaN in loss output! "
                f"pred range: [{pred_r.min():.4f}, {pred_r.max():.4f}], "
                f"target range: [{tgt_r.min():.4f}, {tgt_r.max():.4f}]"
            )

        return result

    return wrapper  # type: ignore[return-value]


def assert_same_shape(
    pred: Tensor,
    target: Tensor,
    context: str = "loss computation",
) -> None:
    """Assert that prediction and target have the same shape.

    Use this for inline validation in loss functions that cannot use the decorator.

    Args:
        pred: Prediction tensor.
        target: Target tensor.
        context: Description of where the check is happening.

    Raises:
        ValueError: If shapes do not match.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in {context}: pred {pred.shape} vs target {target.shape}")


def check_numerical_stability(
    tensor: Tensor,
    name: str = "tensor",
    eps: float = 1e-8,
) -> bool:
    """Check tensor for numerical issues.

    Args:
        tensor: Tensor to check.
        name: Name for logging.
        eps: Epsilon for near-zero detection.

    Returns:
        True if tensor is stable, False otherwise.
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan:
        logger.error(f"NaN detected in {name}")
        return False
    if has_inf:
        logger.error(f"Inf detected in {name}")
        return False

    return True


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Safely divide tensors with epsilon guard.

    Args:
        numerator: Numerator tensor.
        denominator: Denominator tensor.
        eps: Small constant to prevent division by zero.

    Returns:
        Result of numerator / (denominator + eps).
    """
    return numerator / (denominator + eps)


def safe_sqrt(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    """Safely compute square root with epsilon guard.

    Args:
        tensor: Input tensor (must be non-negative).
        eps: Small constant to prevent sqrt(0).

    Returns:
        sqrt(tensor + eps^2).
    """
    return torch.sqrt(tensor + eps * eps)


def clamp_for_log(tensor: Tensor, min_val: float = 1e-8) -> Tensor:
    """Clamp tensor to prevent log(0).

    Args:
        tensor: Input tensor.
        min_val: Minimum value to clamp to.

    Returns:
        Clamped tensor.
    """
    return torch.clamp(tensor, min=min_val)


__all__ = [
    "assert_same_shape",
    "check_numerical_stability",
    "clamp_for_log",
    "safe_divide",
    "safe_sqrt",
    "validate_loss_inputs",
]
