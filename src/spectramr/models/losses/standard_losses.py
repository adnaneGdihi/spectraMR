"""Standard PyTorch losses wrapped for registry.

This module provides registry-compatible wrappers for standard PyTorch losses.
Metrics can be optionally computed during loss calculation.
"""

import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.metrics_aware_loss import MetricsAwareLossMixin
from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.validation import validate_loss_inputs

_REDUCTIONS = frozenset({"mean", "sum", "none"})


def _check_reduction(reduction: str) -> None:
    """Refuse an unknown ``reduction``; the complex branches would treat it as ``none``."""
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction={reduction!r} is not one of {sorted(_REDUCTIONS)}")


@register_loss(name="l1", aliases=["mae", "mean_absolute_error"], domain="agnostic")
class L1Loss(MetricsAwareLossMixin, nn.L1Loss):
    """L1 Loss (Mean Absolute Error) with optional metrics.

    **DOMAIN**: AGNOSTIC (works on any tensor)
    **Input**: [B, C, H, W] or any tensor shape
    **3D Support**: Yes

    Standard MAE. Domain-agnostic - works on k-space, image, latent, etc.
    """

    def __init__(self, reduction: str = "mean", compute_metrics: bool = False):
        """__init__.

        Args:
            reduction (str): Description.
            compute_metrics (bool): Description.
        """
        _check_reduction(reduction)
        super().__init__(reduction=reduction)
        self.compute_metrics_flag = compute_metrics

    # The (loss, metrics) tuple under compute_metrics=True is the mixin's contract,
    # which forward_with_metrics unwraps; nn.L1Loss declares a bare Tensor.
    @validate_loss_inputs
    def forward(  # type: ignore[override]
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Description.
        """
        if pred.is_complex() or target.is_complex():
            diff = pred - target
            diff_mag = torch.sqrt(diff.real.square() + diff.imag.square() + 1e-12)
            if self.reduction == "mean":
                loss_value = diff_mag.mean()
            elif self.reduction == "sum":
                loss_value = diff_mag.sum()
            else:
                loss_value = diff_mag
        else:
            loss_value = nn.functional.l1_loss(pred, target, reduction=self.reduction)

        if self.compute_metrics_flag:
            metrics = self.compute_metrics(pred, target, loss_value)
            return loss_value, metrics
        return loss_value

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute L1-specific metrics."""
        diff = pred - target
        errors = (
            torch.sqrt(diff.real.square() + diff.imag.square() + 1e-12)
            if diff.is_complex()
            else diff.abs()
        )
        if pred.is_complex():
            pred = torch.sqrt(pred.real.square() + pred.imag.square() + 1e-12)
        if target.is_complex():
            target = torch.sqrt(target.real.square() + target.imag.square() + 1e-12)

        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "error_mean": errors.mean().item(),
            "error_std": errors.std().item(),
            "error_min": errors.min().item(),
            "error_max": errors.max().item(),
            "error_median": errors.median().item(),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
        }


@register_loss(name="l2", aliases=["mse", "mean_squared_error"], domain="agnostic")
class MSELoss(MetricsAwareLossMixin, nn.MSELoss):
    """L2 Loss (Mean Squared Error) with optional metrics.

    **DOMAIN**: AGNOSTIC (works on any tensor)
    **Input**: [B, C, H, W] or any tensor shape
    **3D Support**: Yes

    Standard MSE. Domain-agnostic - works on k-space, image, latent, etc.
    """

    def __init__(self, reduction: str = "mean", compute_metrics: bool = False):
        """__init__.

        Args:
            reduction (str): Description.
            compute_metrics (bool): Description.
        """
        _check_reduction(reduction)
        super().__init__(reduction=reduction)
        self.compute_metrics_flag = compute_metrics

    # Same tuple contract as L1Loss.forward above.
    @validate_loss_inputs
    def forward(  # type: ignore[override]
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Description.
        """
        if pred.is_complex() or target.is_complex():
            diff = pred - target
            squared_err = diff.real.square() + diff.imag.square()
            if self.reduction == "mean":
                loss_value = squared_err.mean()
            elif self.reduction == "sum":
                loss_value = squared_err.sum()
            else:
                loss_value = squared_err
        else:
            loss_value = nn.functional.mse_loss(pred, target, reduction=self.reduction)

        if self.compute_metrics_flag:
            metrics = self.compute_metrics(pred, target, loss_value)
            return loss_value, metrics
        return loss_value

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute MSE-specific metrics."""
        diff = pred - target
        errors = diff.real.square() + diff.imag.square() if diff.is_complex() else diff.square()

        if pred.is_complex():
            pred = torch.sqrt(pred.real.square() + pred.imag.square() + 1e-12)
        if target.is_complex():
            target = torch.sqrt(target.real.square() + target.imag.square() + 1e-12)

        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "mse": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "rmse": (
                (loss_value.item() ** 0.5)
                if loss_value.numel() == 1
                else (loss_value.mean().item() ** 0.5)
            ),
            "error_squared_mean": errors.mean().item(),
            "error_squared_std": errors.std().item(),
            "error_squared_max": errors.max().item(),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
        }


# `huber` alias intentionally dropped — `huber` is the canonical name of the
# diffusion-Huber loss in `diffusion_losses.py:78`. The standard `smooth_l1`
# handle remains; if you specifically want PyTorch's `nn.SmoothL1Loss`
# semantics, use `smooth_l1` (or the legacy alias `diffusion_huber`). See
# TODO/audit/12_losses.md F3.
@register_loss(name="smooth_l1", aliases=["diffusion_huber"], domain="agnostic")
class SmoothL1Loss(nn.SmoothL1Loss):
    """Smooth L1 Loss (Huber Loss).
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SmoothL1} = \\text{SmoothL1}(|\\hat{y} - y|)"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.is_complex() or target.is_complex():
            diff = pred - target
            diff_mag = torch.sqrt(diff.real.square() + diff.imag.square() + 1e-12)
            beta = getattr(self, "beta", 1.0)
            return nn.functional.smooth_l1_loss(
                diff_mag,
                torch.zeros_like(diff_mag),
                reduction=self.reduction,
                beta=beta,
            )
        return super().forward(pred, target)


@register_loss(
    name="bce", aliases=["binary_crossentropy", "binary_cross_entropy"], domain="agnostic"
)
class BCELoss(nn.BCELoss):
    """Binary Cross Entropy Loss."""

    pass


@register_loss(name="bce_with_logits", aliases=["bce_logits", "bcewithlogits"], domain="agnostic")
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Binary Cross Entropy with Logits Loss."""

    pass


@register_loss(name="cross_entropy", aliases=["crossentropy", "ce"], domain="agnostic")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss."""

    pass


@register_loss(name="tv", aliases=["total_variation"])
class TotalVariationLoss(nn.Module):
    """Total Variation Loss."""

    def __init__(self, reduction: str = "mean") -> None:
        """__init__.

        Args:
            reduction (str): Description.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """forward.

        Args:
            pred (torch.Tensor): Prediction tensor
            target (torch.Tensor, optional): Ignored. Target image.
        Returns:
            torch.Tensor: Total variation loss penalty.

        forward method for TotalVariationLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x = pred
        if x.ndim < 3:
            raise ValueError(
                f"TotalVariationLoss needs [B, C, *spatial] with at least one spatial "
                f"axis; got shape {tuple(x.shape)}"
            )
        # One squared-difference term per spatial axis, each normalised by its own
        # element count; a unit axis has no neighbours and would divide 0 by 0.
        penalty: torch.Tensor | None = None
        for axis in range(2, x.ndim):
            n = x.size(axis)
            if n < 2:
                continue
            diff = x.narrow(axis, 1, n - 1) - x.narrow(axis, 0, n - 1)
            if x.is_complex():
                axis_tv = (diff.real.square() + diff.imag.square()).sum()
            else:
                axis_tv = torch.pow(diff, 2).sum()
            term = axis_tv / self._tensor_size(diff)
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            zero: torch.Tensor = x.real.sum() * 0.0
            return zero
        tv: torch.Tensor = 2 * penalty / x.size(0)
        return tv

    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        """Elements per sample: the product of every axis after the batch axis."""
        return math.prod(t.shape[1:])
