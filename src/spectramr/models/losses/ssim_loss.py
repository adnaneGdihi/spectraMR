"""SSIM Loss Implementation
========================

Structural Similarity Index Measure (SSIM) loss for image quality assessment.
Used in super resolution, reconstruction, and image restoration tasks.
Metrics can be optionally computed during loss calculation.

References:
- Wang et al. "Image quality assessment: from error visibility to
  structural similarity" IEEE TIP 2004
- Zhao et al. "Loss Functions for Image Restoration with Neural Networks"
  IEEE TIP 2016

"""

from typing import Any

import torch
import torch.nn.functional as F

from spectramr.core.metrics.evaluation_metrics import compute_ssim_map, gaussian_kernel
from spectramr.models.losses.metrics_aware_loss import MetricsAwareLossMixin
from spectramr.models.losses.registry import register_loss

from .base_loss import BaseLoss


def _resolve_twin(
    name: str,
    value: float | None,
    twin_name: str,
    twin: float | None,
    *,
    default: float,
) -> float:
    """Resolve a knob that has two accepted spellings; raise when they disagree."""
    if value is not None and twin is not None and value != twin:
        raise ValueError(
            f"{name}={value!r} and {twin_name}={twin!r} name the same window parameter; "
            "pass one of them"
        )
    if value is not None:
        return value
    return twin if twin is not None else default


@register_loss(name="ssim", aliases=["SSIMLoss"], domain="image")
class SSIMLoss(MetricsAwareLossMixin, BaseLoss):
    """Structural Similarity Index Measure (SSIM) loss with optional metrics.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: No (2D window sliding)

    Perceptual image quality metric. Do NOT use with k-space [B, 2, H, W].
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float | None = None,
        data_range: float = 1.0,
        win_sigma: float | None = None,
        K1: float = 0.01,  # noqa: N803 - Wang et al. (2004) name the constants K1/K2
        K2: float = 0.03,  # noqa: N803
        reduction: str = "mean",
        auto_range: bool = True,
        compute_metrics: bool = False,
    ) -> None:
        """__init__.

        Args:
            window_size (int): Gaussian window side (odd).
            sigma (float | None): Gaussian window standard deviation (1.5).
            data_range (float): Dynamic range L in C1=(K1 L)^2, C2=(K2 L)^2;
                with ``auto_range`` it is the floor of the per-batch range.
            win_sigma (float | None): Spelling of ``sigma`` shared with
                ``MSSSIMLoss``; must agree with ``sigma`` when both are given.
            K1 (float): Luminance stabiliser constant.
            K2 (float): Contrast stabiliser constant.
            reduction (str): ``'mean'``, ``'sum'`` or ``'none'``.
            auto_range (bool): Resolve L from the batch instead of ``data_range``.
            compute_metrics (bool): Also return a metrics dict (host syncs).
        """
        super().__init__(reduction=reduction)
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be 'mean', 'sum', or 'none'",
            )
        sigma = _resolve_twin("sigma", sigma, "win_sigma", win_sigma, default=1.5)

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = float(data_range)
        self.win_sigma = sigma
        self.K1 = K1
        self.K2 = K2
        self.reduction = reduction
        self.auto_range = auto_range
        self.compute_metrics_flag = compute_metrics

        # Hint type for static analyzers; buffer will populate attribute
        self.window: torch.Tensor
        self.register_buffer(
            "window",
            gaussian_kernel(window_size, sigma, torch.device("cpu")),
        )

    def _validate_ssim_inputs(self, img1: torch.Tensor, img2: torch.Tensor) -> None:
        # Basic checks from BaseLoss (shape/device equality)
        """_validate_ssim_inputs.

        Args:
            img1 (torch.Tensor): Description.
            img2 (torch.Tensor): Description.
        """
        super()._validate_inputs(img1, img2)
        # Additional SSIM-specific constraints
        if img1.ndim < 3 or img2.ndim < 3:
            raise ValueError("Input images must have at least 3 dimensions [C, H, W]")

    @staticmethod
    def _to_real(x: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to real magnitude for SSIM computation."""
        if torch.is_complex(x):
            return x.abs()
        return x

    def _resolve_data_range(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        override: float | None,
    ) -> float | torch.Tensor:
        """Resolve the SSIM data range without any GPU sync.

        With ``auto_range=True`` the range is returned as a 0-dim float64
        tensor computed on-device — no ``.item()`` calls, so a
        default-configured ``ssim`` loss never syncs the GPU per training
        step. ``compute_ssim_map`` consumes it via
        ``C = (k * data_range) ** 2``, which broadcasts a 0-dim tensor
        exactly like the former Python float.

        Args:
            img1 (torch.Tensor): Description.
            img2 (torch.Tensor): Description.
            override (Optional[float]): Description.
        Returns:
            float | torch.Tensor: Scalar range (0-dim tensor when auto).
        """
        if override is not None:
            return float(override)

        floor = max(float(self.data_range), 1e-6)
        if not self.auto_range:
            return floor

        # Use real-valued tensors for amin/amax (ComplexFloat unsupported).
        # float64 scalars reproduce the former `.item()` (Python float)
        # arithmetic bit-for-bit inside compute_ssim_map.
        r1 = self._to_real(img1).detach()
        r2 = self._to_real(img2).detach()

        min_val = torch.minimum(r1.amin(), r2.amin()).double()
        max_val = torch.maximum(r1.amax(), r2.amax()).double()
        dynamic_range = max_val - min_val
        resolved = torch.where(min_val < 0, dynamic_range, max_val)
        return resolved.clamp(min=floor)

    def resolve_data_range(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        data_range: float | None = None,
    ) -> float | torch.Tensor:
        """resolve_data_range.

        Args:
            img1 (torch.Tensor): Description.
            img2 (torch.Tensor): Description.
            data_range (Optional[float]): Description.
        Returns:
            float | torch.Tensor: Description.
        """
        return self._resolve_data_range(img1, img2, data_range)

    def _get_window(self, img: torch.Tensor) -> torch.Tensor:
        """_get_window.

        Args:
            img (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        window = self.window
        if window.device != img.device or window.dtype != img.dtype:
            window = window.to(device=img.device, dtype=img.dtype)
        channels = img.shape[-3] if img.ndim >= 3 else 1
        if window.shape[0] != channels:
            window = window.expand(
                channels,
                1,
                self.window_size,
                self.window_size,
            )
        return window

    def compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        data_range: float | None = None,
    ) -> torch.Tensor:
        """compute_ssim.

        Args:
            img1 (torch.Tensor): Description.
            img2 (torch.Tensor): Description.
            data_range (Optional[float]): Description.
        Returns:
            torch.Tensor: Description.
        """
        self._validate_ssim_inputs(img1, img2)
        resolved_range = self._resolve_data_range(img1, img2, data_range)
        return self.compute_ssim_with_resolved_range(
            img1,
            img2,
            resolved_range,
        )

    def compute_ssim_with_resolved_range(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        data_range: float | torch.Tensor,
    ) -> torch.Tensor:
        """compute_ssim_with_resolved_range.

        Args:
            img1 (torch.Tensor): Description.
            img2 (torch.Tensor): Description.
            data_range (float | torch.Tensor): Scalar range; a 0-dim tensor
                stays on-device (no GPU sync).
        Returns:
            torch.Tensor: Description.
        """
        self._validate_ssim_inputs(img1, img2)

        # Handle 5D tensors by flattening batch and depth
        original_shape = img1.shape
        if img1.ndim == 5:
            B, C, D, H, W = img1.shape
            img1 = img1.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            img2 = img2.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        elif img1.ndim == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        window = self._get_window(img1)
        ssim_map = compute_ssim_map(
            img1,
            img2,
            window,
            self.window_size,
            data_range,  # type: ignore[arg-type]  # a 0-dim tensor range broadcasts in C1/C2
            self.K1,
            self.K2,
        )

        # Average over spatial dimensions
        ssim_val = ssim_map.mean(dim=(-2, -1))

        # Restore original batch structure if needed
        if len(original_shape) == 5:
            B, C, D, H, W = original_shape
            # ssim_val is [B*D, C]
            ssim_val = ssim_val.view(B, D, C).mean(dim=1)  # Average over depth

        return ssim_val.mean(dim=-1)  # Average over channels

    def forward(  # type: ignore[override]  # optionally returns (loss, metrics)
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
            data_range (Optional[float]): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Description.
        """
        self.window = self.window.to(pred.device)
        # SSIM is image-space only; convert complex inputs to magnitude
        pred = self._to_real(pred)
        target = self._to_real(target)
        ssim_per_image = self.compute_ssim(pred, target, data_range=data_range)
        loss_per_image = 1.0 - ssim_per_image
        loss_value = self._apply_reduction(loss_per_image)

        if self.compute_metrics_flag:
            metrics = self.compute_metrics(pred, target, loss_value, data_range)
            return loss_value, metrics
        return loss_value

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        data_range: float | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute SSIM-specific metrics."""
        # Ensure real-valued inputs for metric computation
        pred = self._to_real(pred)
        target = self._to_real(target)
        ssim_value = self.compute_ssim(pred, target, data_range=data_range)
        ssim_mean = ssim_value.mean().item()

        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "ssim": ssim_mean,  # Actual SSIM score (0-1, higher is better)
            "ssim_inverse": (1.0 - ssim_mean),  # Loss representation
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
        }


@register_loss(name="ms_ssim", aliases=["MSSSIMLoss", "ms-ssim"])
class MSSSIMLoss(BaseLoss):
    """Multi-Scale Structural Similarity Index Measure (MS-SSIM) loss.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{MSSSIM} = 1 - \\prod_j (\\text{SSIM}_j)^{\\beta_j}"""

    scale_weights: torch.Tensor

    def __init__(
        self,
        window_size: int | None = None,
        sigma: float | None = None,
        data_range: float = 1.0,
        win_size: int | None = None,
        win_sigma: float | None = None,
        weights: list[float] | None = None,
        K1: float = 0.01,  # noqa: N803 - Wang et al. (2004) name the constants K1/K2
        K2: float = 0.03,  # noqa: N803
        reduction: str = "mean",
        auto_range: bool = True,
    ) -> None:
        """__init__.

        Args:
            window_size (int | None): Gaussian window side (odd, default 11).
            sigma (float | None): Gaussian window standard deviation (default 1.5).
            data_range (float): Dynamic range L (floor of the batch range
                when ``auto_range``).
            win_size (int | None): Spelling of ``window_size``; must agree
                with it when both are given.
            win_sigma (float | None): Spelling of ``sigma``; must agree with
                it when both are given.
            weights (Optional[list[float]]): Per-scale exponents (Wang 2003).
            K1 (float): Luminance stabiliser constant.
            K2 (float): Contrast stabiliser constant.
            reduction (str): ``'mean'``, ``'sum'`` or ``'none'``.
            auto_range (bool): Resolve L from the batch instead of ``data_range``.
        """
        super().__init__(reduction=reduction)
        win_size = int(_resolve_twin("window_size", window_size, "win_size", win_size, default=11))
        win_sigma = _resolve_twin("sigma", sigma, "win_sigma", win_sigma, default=1.5)
        self.window_size = win_size
        self.sigma = win_sigma
        self.data_range = float(data_range)
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.K1 = K1
        self.K2 = K2
        self.reduction = reduction
        self.auto_range = auto_range
        self.weights = tuple(
            weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        )
        # Built once: a per-step torch.tensor(list, device=cuda) is a blocking copy.
        self.register_buffer(
            "scale_weights", torch.tensor(self.weights).unsqueeze(1), persistent=False
        )
        self._ssim = SSIMLoss(
            window_size=win_size,
            sigma=win_sigma,
            data_range=data_range,
            win_sigma=win_sigma,
            K1=K1,
            K2=K2,
            reduction="none",
            auto_range=auto_range,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float | None = None,
    ) -> torch.Tensor:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
            data_range (Optional[float]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if pred.ndim < 3 or target.ndim < 3:
            raise ValueError("Input images must have at least 3 dimensions [C, H, W]")
        if pred.shape != target.shape:
            raise ValueError("Input images must have the same shape")

        resolved_range = self._ssim.resolve_data_range(
            pred,
            target,
            data_range,
        )

        # Handle 5D tensors by flattening batch and depth
        original_shape = pred.shape
        if pred.ndim == 5:
            B, C, D, H, W = pred.shape
            current_img1 = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            current_img2 = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        elif pred.ndim == 3:
            current_img1 = pred.unsqueeze(0)
            current_img2 = target.unsqueeze(0)
        else:
            current_img1 = pred
            current_img2 = target
        ssim_values: list[torch.Tensor] = []
        weights_used: list[float] = []

        for index, weight in enumerate(self.weights):
            ssim_val = self._ssim.compute_ssim_with_resolved_range(
                current_img1,
                current_img2,
                resolved_range,
            )
            ssim_values.append(ssim_val)
            weights_used.append(weight)

            if index == len(self.weights) - 1:
                break
            if min(current_img1.shape[-2:]) < 2:
                break

            current_img1 = F.avg_pool2d(current_img1, kernel_size=2, stride=2)
            current_img2 = F.avg_pool2d(current_img2, kernel_size=2, stride=2)

        if not ssim_values:
            raise RuntimeError("MS-SSIM requires at least one valid scale")

        ssim_stack = torch.stack(ssim_values, dim=0).clamp(min=1e-6)
        weight_tensor = self.scale_weights[: len(weights_used)].to(
            device=pred.device, dtype=pred.dtype
        )

        weighted = torch.pow(ssim_stack, weight_tensor)
        ms_ssim = torch.prod(weighted, dim=0)

        # Restore original batch structure if needed
        if len(original_shape) == 5:
            B, C, D, H, W = original_shape
            # ms_ssim is [B*D]
            ms_ssim = ms_ssim.view(B, D).mean(dim=1)  # Average over depth

        loss_per_image = 1.0 - ms_ssim
        return self._apply_reduction(loss_per_image)


# Utility functions for easy access
def ssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience function for SSIM loss."""
    loss_fn = SSIMLoss(**kwargs)
    loss: torch.Tensor = loss_fn(img1, img2)
    return loss


def msssim_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience function for MS-SSIM loss."""
    loss_fn = MSSSIMLoss(**kwargs)
    loss: torch.Tensor = loss_fn(img1, img2)
    return loss


def ssim(img1: torch.Tensor, img2: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Convenience function for SSIM metric (higher is better)."""
    return 1.0 - ssim_loss(img1, img2, **kwargs)


def msssim(img1: torch.Tensor, img2: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Convenience function for MS-SSIM metric (higher is better)."""
    return 1.0 - msssim_loss(img1, img2, **kwargs)


__all__ = [
    "MSSSIMLoss",
    "SSIMLoss",
    "msssim",
    "msssim_loss",
    "ssim",
    "ssim_loss",
]
