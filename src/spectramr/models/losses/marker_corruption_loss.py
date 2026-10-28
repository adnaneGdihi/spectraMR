"""Marker-Anchored Corruption Loss for Virtual Fiducial training.

The core insight: **markers are known ground-truth geometry** placed in the
FOV before corruption.  By comparing the model's predicted marker region
to the known ideal marker, we directly measure how well the model has
learned to invert the corruption.  This corruption-inversion knowledge
transfers to the anatomy region.

Physics motivation:
    The Digital Twin embeds 4 corner markers at known positions with known
    intensities, then applies realistic scanner corruptions (B0, B1, motion,
    noise, chemical shift, eddy, spikes, crosstalk).  All corruptions affect
    the markers identically to the anatomy.  If ``G(corrupted)[marker_mask]``
    matches ``ideal_marker[marker_mask]``, the model has successfully learned
    the inverse of the corruption operator within the marker region.

Loss:
    L_marker = w₁ · |G(corrupted)[M] - ideal[M]|₁
             + w₂ · (1 - SSIM(G(corrupted)[M], ideal[M]))
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from spectramr.models.losses.registry import register_loss


@register_loss("marker_corruption", aliases=["MarkerCorruptionLoss", "marker"])
class MarkerCorruptionLoss(nn.Module):
    r"""Marker-anchored corruption loss for VF training.

    Computes the L1 distance between the model's prediction in the marker
    region and the known ideal marker signal.  The marker mask is provided
    by the :class:`DigitalTwinSimulator`.

    Args:
        lambda_l1: Weight for L1 component in marker region.
        lambda_consistency: Weight for marker-anatomy boundary consistency.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{MarkerCorruption} = \lambda_1 \| (\hat{y} - y) \odot M \|_1 + \lambda_2 \mathcal{L}_{consistency}"""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_consistency: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        marker_mask: Tensor | None = None,
        ideal_marker: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Compute marker corruption loss.

        Args:
            prediction: Model output ``[B, C, H, W]`` (real-stacked or complex).
            target: Clean ground truth ``[B, C, H, W]``.
            marker_mask: Binary mask ``[1, 1, H, W]`` from simulator.
            ideal_marker: Ideal marker signal ``[1, 1, H, W]`` (complex).
            **kwargs: Ignored (protocol compliance).

        Returns:
            Scalar loss tensor.

        forward method for MarkerCorruptionLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ideal_marker (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_mask is None or ideal_marker is None:
            # Fallback: plain L1 if marker info is not available
            return nn.functional.l1_loss(prediction, target)

        # ── Marker region L1 ──
        # Expand mask to match prediction channels
        B, C, H, W = prediction.shape
        mask = marker_mask.expand(B, 1, H, W)  # [B, 1, H, W]

        # ── Normalize prediction & ideal_marker to the same real-stacked layout ──
        # Prediction may be complex (C=1, dtype=complex) or real-stacked (C=2).
        # Ideal marker is typically complex [1, 1, H, W]. Reduce both to a
        # magnitude representation so the loss is well-defined regardless of
        # how many real channels the strategy passes.
        if torch.is_complex(prediction):
            pred_real = prediction.abs()
        elif C == 2:
            # real/imag stacked → magnitude
            pred_real = torch.sqrt(prediction[:, :1].pow(2) + prediction[:, 1:2].pow(2) + 1e-12)
        else:
            pred_real = prediction.abs() if prediction.is_complex() else prediction

        ideal_real = ideal_marker.abs() if torch.is_complex(ideal_marker) else ideal_marker

        # Broadcast ideal across batch and channels of pred_real
        # (use repeat — `expand` cannot grow non-singleton dims).
        if ideal_real.shape[0] != pred_real.shape[0]:
            ideal_real = ideal_real.expand(pred_real.shape[0], -1, -1, -1)
        if ideal_real.shape[1] != pred_real.shape[1]:
            if ideal_real.shape[1] == 1:
                ideal_real = ideal_real.expand(-1, pred_real.shape[1], -1, -1)
            else:
                # Mismatched non-singleton channels: collapse both to magnitude
                ideal_real = ideal_real.mean(dim=1, keepdim=True).expand(
                    -1, pred_real.shape[1], -1, -1
                )

        # Use the magnitude tensors going forward
        prediction = pred_real
        ideal_stacked = ideal_real

        # Extract marker regions from prediction and ideal
        # Use mask on each channel pair
        mask_expanded = mask.expand_as(prediction)
        pred_marker = prediction * mask_expanded
        ideal_marker_expanded = ideal_stacked * mask_expanded

        # L1 in marker region (normalised by mask area)
        mask_area = mask_expanded.sum().clamp(min=1.0)
        l1_marker = (pred_marker - ideal_marker_expanded).abs().sum() / mask_area

        # ── Boundary consistency ──
        # The prediction should smoothly transition from anatomy to marker.
        # Compute gradient magnitude at mask boundary to penalise sharp jumps.
        if self.lambda_consistency > 0:
            # Dilate mask by 1 pixel to get boundary
            kernel = torch.ones(1, 1, 3, 3, device=prediction.device)
            dilated = nn.functional.conv2d(mask.float(), kernel, padding=1).clamp(0, 1)
            boundary = (dilated > 0) & (mask < 0.5)
            boundary = boundary.float().expand_as(prediction)

            # Gradient at boundary
            boundary_area = boundary.sum().clamp(min=1.0)
            grad_x = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).abs()
            grad_y = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]).abs()
            boundary_x = boundary[:, :, :, 1:]
            boundary_y = boundary[:, :, 1:, :]
            consistency = (
                (grad_x * boundary_x).sum() + (grad_y * boundary_y).sum()
            ) / boundary_area
        else:
            consistency = torch.tensor(0.0, device=prediction.device)

        return self.lambda_l1 * l1_marker + self.lambda_consistency * consistency


__all__ = ["MarkerCorruptionLoss"]
