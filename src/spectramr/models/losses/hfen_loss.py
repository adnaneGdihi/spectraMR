r"""High-Frequency Error Norm (HFEN) for MRI Reconstruction Quality.

Applies a Laplacian of Gaussian (LoG) kernel to both prediction and
target before computing the normalised L2 error.  This isolates
**structural boundary fidelity** from bulk contrast, which is critical
for evaluating trabecular bone, vessel walls, and other fine anatomical
features that standard PSNR/SSIM metrics may overlook.

.. math::

    \text{HFEN} = \frac{\lVert \text{LoG} \ast \hat{x}
                         - \text{LoG} \ast x \rVert_2}
                       {\lVert \text{LoG} \ast x \rVert_2 + \epsilon}

The LoG kernel is constructed analytically as:

.. math::

    \text{LoG}(r) = -\frac{1}{\pi\sigma^4}
                     \Bigl(1 - \frac{r^2}{2\sigma^2}\Bigr)
                     \exp\!\Bigl(-\frac{r^2}{2\sigma^2}\Bigr)

where :math:`r^2 = x^2 + y^2`.

Reference:
    Ravishankar & Bresler, "MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning", *IEEE TMI* (2011).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spectramr.models.losses.registry import register_loss


def _build_log_kernel(
    kernel_size: int = 15,
    sigma: float = 1.5,
    device: torch.device | None = None,
) -> Tensor:
    """Construct a 2D Laplacian of Gaussian (LoG) convolution kernel.

    Args:
        kernel_size: Odd integer specifying kernel width/height.
        sigma: Gaussian standard deviation.
        device: Target device.

    Returns:
        Normalised LoG kernel ``(1, 1, K, K)``.
    """
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r2 = x**2 + y**2
    s2 = sigma**2

    # Analytical LoG
    log_kernel = -(1.0 / (math.pi * s2**2)) * (1.0 - r2 / (2.0 * s2)) * torch.exp(-r2 / (2.0 * s2))

    # Zero-mean normalisation (remove DC bias)
    log_kernel = log_kernel - log_kernel.mean()

    return log_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)


@register_loss(
    name="hfen",
    aliases=[
        "high_frequency_error_norm",
        "HFENLoss",
        "HighFrequencyErrorNorm",
        # legacy alias — was on the old physics_losses.HFENLoss decorator
        "high_frequency_error",
    ],
    domain="image",
)
class HFENLoss(nn.Module):
    r"""High-Frequency Error Norm loss/metric.

    **DOMAIN**: IMAGE (operates on magnitude images)
    **Input**: ``[B, C, H, W]`` real-valued tensors, or ``[B, C, H, W, D]``
    volumetric tensors (TorchIO depth-last layout).
    **3D Support**: Yes — a 5-D volume is treated slice-wise: the trailing
    depth axis is folded into the batch dimension and the 2-D LoG is applied
    to every in-plane ``(H, W)`` slice independently, then averaged. (The
    pre-2026-06 contract was 4-D only and ``_apply_log`` raised ``ValueError:
    too many values to unpack`` on a 5-D input — which crashed every
    volumetric Hilbert-Mamba arm, e.g. ``exp_hm_07_25d_mamba``.)

    Can be used both as a **training loss** and as a **validation metric**.
    When used as a loss, the normalisation by target energy is optional
    (``normalize=False`` yields a pure L2 over high-frequency features).

    Example::

        loss_fn = HFENLoss(sigma=1.5, normalize=True)
        hfen = loss_fn(pred, target)  # scalar ∈ [0, ∞)

    Args:
        kernel_size: LoG kernel size (odd integer).
        sigma: Gaussian sigma controlling the high-frequency band.
            Smaller sigma → higher frequencies only.
            Larger sigma → broader high-frequency band.
        normalize: If ``True``, return the ratio HFEN; if ``False``,
            return the unnormalised L2 error (suitable for loss).
        reduction: ``'mean'`` or ``'sum'`` across batch.
    """

    log_kernel: Tensor

    def __init__(
        self,
        kernel_size: int = 15,
        sigma: float = 1.5,
        normalize: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.normalize = normalize
        self.reduction = reduction

        # Build LoG kernel (registered as buffer for device tracking)
        kernel = _build_log_kernel(kernel_size, sigma)
        self.register_buffer("log_kernel", kernel)
        self.pad = kernel_size // 2

    def _apply_log(self, x: Tensor) -> Tensor:
        """Apply LoG filter to each channel independently."""
        if x.dim() != 4:
            raise ValueError(f"HFEN expects [B, C, H, W] after depth folding, got {tuple(x.shape)}")
        channels = x.shape[1]
        # Group conv: apply same kernel to each channel
        kernel = self.log_kernel.to(dtype=x.dtype).expand(channels, 1, -1, -1)
        return F.conv2d(x, kernel, padding=self.pad, groups=channels)

    def forward(self, pred: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        """Compute HFEN.

        Args:
            pred: Predicted image ``(B, C, H, W)`` or volumetric
                ``(B, C, H, W, D)`` (TorchIO depth-last layout).
            target: Ground-truth image (same shape as ``pred``).

        Returns:
            HFEN value (scalar).
        """
        # Handle complex inputs: use magnitude
        if pred.is_complex():
            pred = pred.abs()
        if target.is_complex():
            target = target.abs()

        # Volumetric (5-D) inputs: the LoG kernel is 2-D, so apply it
        # slice-wise by folding the trailing depth axis into the batch
        # dimension (TorchIO emits [B, C, H, W, D]). For D == 1 this is a
        # plain squeeze; for D > 1 each slice contributes one per-sample
        # HFEN value and the reduction averages over (batch · depth).
        if pred.dim() == 5:
            b, c, h, w, d = pred.shape
            pred = pred.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
            target = target.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

        log_pred = self._apply_log(pred)
        log_target = self._apply_log(target)

        # Per-sample L2 error. The floor under the root only keeps sqrt'(0) finite;
        # sqrt(floor) is what HFEN(x, x) scores, so it must stay negligible.
        diff_sq = ((log_pred - log_target) ** 2).sum(dim=(1, 2, 3))
        diff_norm = diff_sq.clamp_min(1e-20).sqrt()

        if self.normalize:
            target_norm = torch.sqrt((log_target**2).sum(dim=(1, 2, 3)) + 1e-8)
            hfen = diff_norm / target_norm
        else:
            hfen = diff_norm

        if self.reduction == "mean":
            return hfen.mean()
        elif self.reduction == "sum":
            return hfen.sum()
        return hfen

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, sigma={self.sigma}, "
            f"normalize={self.normalize}, reduction='{self.reduction}'"
        )


__all__ = ["HFENLoss"]
