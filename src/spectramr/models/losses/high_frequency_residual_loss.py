"""High-frequency residual loss for latent-diffusion residual heads.

Plan: ``TODO/backlog_multi_contrast_remaining.md`` Item 2.

Latent diffusion is much cheaper than pixel-space diffusion, but the
autoencoder bottleneck loses high-frequency content (fine lesion
detail, edges). To mitigate this, a small pixel-space residual head
runs on top of the decoded latent-diffusion output. This loss isolates
the high-frequency component of the target so the residual head only
learns the *missing* high-frequency signal, not a full re-encoding.

Definition::

    target_hf = target - gaussian_lowpass(target, sigma)
    pred_hf   = pred   - gaussian_lowpass(pred,   sigma)
    L_hf      = L1(pred_hf, target_hf)

The Gaussian lowpass is implemented as a separable depthwise convolution
so it's per-channel, AMP-safe, and runs at full speed on the GPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def _gaussian_kernel_1d(sigma: float, *, truncate: float = 3.0) -> torch.Tensor:
    """Build a 1-D Gaussian kernel of half-width ``ceil(truncate * sigma)``.

    Returns a tensor of shape ``[1, 1, K]`` ready for ``F.conv2d``-style
    separable convolution.
    """
    radius = max(1, math.ceil(truncate * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def _gaussian_lowpass(
    x: torch.Tensor,
    *,
    sigma: float,
    truncate: float = 3.0,
    kernel_1d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Separable depthwise Gaussian lowpass on ``[B, C, H, W]`` tensors.

    Works on real-valued tensors (magnitude images). For complex tensors
    apply to magnitude and phase separately or convert to ``[B, 2C, H, W]``
    real-imag stack. ``kernel_1d`` lets a caller pass the prebuilt
    ``_gaussian_kernel_1d(sigma, truncate=truncate)`` instead of rebuilding it.
    """
    if x.dim() != 4:
        raise ValueError(f"_gaussian_lowpass expects [B, C, H, W], got shape {tuple(x.shape)}.")
    C = x.shape[1]
    if kernel_1d is None:
        kernel_1d = _gaussian_kernel_1d(sigma, truncate=truncate)
    kernel_1d = kernel_1d.to(device=x.device, dtype=x.dtype)
    radius = kernel_1d.shape[-1] // 2
    # Expand to [C, 1, 1, K] / [C, 1, K, 1] for depthwise conv.
    kernel_h = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    kernel_v = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel_h, groups=C)
    x = F.conv2d(x, kernel_v, groups=C)
    return x


@register_loss(name="high_frequency_residual", aliases=["HighFrequencyResidualLoss"])
class HighFrequencyResidualLoss(nn.Module):
    """L1 loss between high-frequency residuals of prediction and target.

    Args:
        sigma: Gaussian lowpass standard deviation in pixels. Larger
            values isolate more aggressive low frequencies as "context",
            leaving more in the high-frequency residual. Default 2.0
            gives detail roughly 4–6 px wide.
        truncate: Gaussian kernel half-width in units of ``sigma``.
            Default 3.0 is standard.
        loss_type: ``"l1"`` (default), ``"l2"``, or ``"smooth_l1"``.

    See ``TODO/backlog_multi_contrast_remaining.md`` §Item 2.
    """

    kernel_1d: torch.Tensor

    def __init__(
        self,
        *,
        sigma: float = 2.0,
        truncate: float = 3.0,
        loss_type: str = "l1",
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if loss_type not in ("l1", "l2", "smooth_l1"):
            raise ValueError(
                f"loss_type must be one of 'l1' | 'l2' | 'smooth_l1', got {loss_type!r}."
            )
        self.sigma = sigma
        self.truncate = truncate
        self.loss_type = loss_type
        self.register_buffer(
            "kernel_1d", _gaussian_kernel_1d(sigma, truncate=truncate), persistent=False
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        """Return the high-frequency residual loss.

        Args:
            pred: Predicted image, shape ``[B, C, H, W]``.
            target: Target image, same shape and dtype as ``pred``.

        Returns:
            Scalar tensor — the loss value.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}."
            )
        pred_lp = _gaussian_lowpass(
            pred, sigma=self.sigma, truncate=self.truncate, kernel_1d=self.kernel_1d
        )
        target_lp = _gaussian_lowpass(
            target, sigma=self.sigma, truncate=self.truncate, kernel_1d=self.kernel_1d
        )
        pred_hf = pred - pred_lp
        target_hf = target - target_lp
        if self.loss_type == "l1":
            return F.l1_loss(pred_hf, target_hf)
        if self.loss_type == "l2":
            return F.mse_loss(pred_hf, target_hf)
        return F.smooth_l1_loss(pred_hf, target_hf)


__all__ = ["HighFrequencyResidualLoss", "_gaussian_lowpass"]
