"""Soft Dynamic Time Warping Loss for misregistered MRI pairs.

Computes a differentiable approximation of Dynamic Time Warping between
predicted and target 1D sequences.  Designed for use with Hilbert-linearized
MRI sequences where paired ULF/HF scans may not be perfectly registered.

A spatial shift in 3D mathematically results in localized insertions/deletions
in the 1D Hilbert sequence.  Standard L1/L2 losses produce blurry averages
under such misalignment.  Soft-DTW allows elastic alignment while maintaining
differentiability.

To keep computation tractable, the loss operates on **windowed subsequences**
of length ``window_size`` (default 64), giving O(B · K · w²) work where
K = L / stride_size, in 2w - 1 batched anti-diagonal steps.

Math:
    DTW(x, y) = min_π Σ_{(i,j)∈π} d(x_i, y_j)
    Soft-DTW replaces min with soft-min: min^γ(a) = -γ log Σ exp(-a/γ)

Reference:
    Cuturi & Blondel (2017). "Soft-DTW: a Differentiable Loss Function for
    Time-Series." ICML 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


def _soft_dtw_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute Soft-DTW distance between two sequences.

    The DP recursion ``R[i, j] = d[i-1, j-1] + softmin(R[i-1, j-1], R[i-1, j], R[i, j-1])``
    is swept one anti-diagonal ``i + j = s`` at a time: every cell on a diagonal
    depends only on the two previous diagonals, so ``N + M - 1`` vectorised steps
    replace ``N * M`` scalar ones.

    Args:
        pred: Predicted sequence ``[B, L, D]``.
        target: Target sequence ``[B, L, D]``.
        gamma: Soft-min smoothing parameter.

    Returns:
        Scalar DTW distance per batch element ``[B]``.
    """
    batch, n = pred.shape[0], pred.shape[1]
    m = target.shape[1]

    # Pairwise squared Euclidean distance: [B, N, M]
    dist = torch.cdist(pred, target, p=2.0).pow(2)
    # A diagonal of the column-flipped cost is an anti-diagonal of the cost,
    # ordered by increasing row index i.
    flipped = dist.flip(-1)

    inf = float("inf")
    # diag_s[:, i] holds R[i, s - i]; out-of-range cells are +inf.
    prev2 = torch.full((batch, n + 1), inf, device=pred.device, dtype=dist.dtype)
    prev2[:, 0] = 0.0  # R[0, 0]
    prev1 = torch.full((batch, n + 1), inf, device=pred.device, dtype=dist.dtype)
    for s in range(2, n + m + 1):
        lo = max(1, s - m)
        hi = min(n, s - 1)
        cost = torch.diagonal(flipped, offset=m + 1 - s, dim1=1, dim2=2)  # rows lo..hi
        r0 = prev2[:, lo - 1 : hi]  # R[i-1, j-1]
        r1 = prev1[:, lo - 1 : hi]  # R[i-1, j]
        r2 = prev1[:, lo : hi + 1]  # R[i, j-1]
        stack = torch.stack([r0, r1, r2], dim=-1)
        soft_min = -gamma * torch.logsumexp(-stack / gamma, dim=-1)
        current = torch.cat(
            [
                prev1.new_full((batch, lo), inf),
                cost + soft_min,
                prev1.new_full((batch, n - hi), inf),
            ],
            dim=1,
        )
        prev2, prev1 = prev1, current

    return prev1[:, n]


@register_loss("soft_dtw", aliases=["SoftDTW", "sdtw"])
class SoftDTWLoss(nn.Module):
    """Windowed Soft-DTW loss for Hilbert-linearized MRI sequences.

    Applies Soft-DTW on non-overlapping windows of the flattened sequence
    to keep memory and compute tractable.

    Args:
        gamma: Soft-min smoothing parameter (lower → harder alignment).
        window_size: Size of each DTW window.
        stride_size: Stride between windows (default = window_size).
        normalize: Whether to normalize by window count.

    Example::

        loss_fn = SoftDTWLoss(gamma=0.1, window_size=64)
        loss = loss_fn(prediction, target)  # prediction/target: [B, C, H, W]
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SoftDTW} = \\text{SoftDTW}_\\gamma(\\hat{y}, y)"""

    def __init__(
        self,
        gamma: float = 0.1,
        window_size: int = 64,
        stride_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.window_size = window_size
        self.stride_size = stride_size or window_size
        self.normalize = normalize

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute windowed Soft-DTW loss.

        Args:
            prediction: Predicted tensor ``[B, C, H, W]`` or ``[B, L, D]``.
            target: Target tensor (same shape as prediction).

        Returns:
            Scalar loss value.

        forward method for SoftDTWLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Volumetric (5-D) inputs: the Soft-DTW sequence is per-slice, so
        # fold the trailing depth axis into the batch dimension (TorchIO
        # emits [B, C, H, W, D]) and align each in-plane slice independently.
        # For D == 1 this is a plain squeeze; for D > 1 each slice becomes its
        # own DTW alignment. This is what lets volumetric Hilbert-Mamba arms
        # (e.g. exp_hm_08_sdtw_mamba, hdsf_mamba_unet backbone) use the loss
        # at all — the pre-2026-06 contract raised on any 5-D input.
        if prediction.dim() == 5:
            b, c, h, w, depth = prediction.shape
            prediction = prediction.permute(0, 4, 1, 2, 3).reshape(b * depth, c, h, w)
            target = target.permute(0, 4, 1, 2, 3).reshape(b * depth, c, h, w)

        # Flatten to [B, L, D] if spatial
        if prediction.dim() == 4:
            b4, c4 = prediction.shape[0], prediction.shape[1]
            pred_seq = prediction.reshape(b4, c4, -1).permute(0, 2, 1)
            tgt_seq = target.reshape(b4, c4, -1).permute(0, 2, 1)
        elif prediction.dim() == 3:
            pred_seq = prediction
            tgt_seq = target
        else:
            raise ValueError(f"SoftDTWLoss expects 3D, 4D, or 5D input, got {prediction.dim()}D")

        batch, length, feat = pred_seq.shape
        if length < self.window_size:
            # Sequence shorter than window: compute on full sequence
            return _soft_dtw_distance(pred_seq, tgt_seq, self.gamma).mean()

        # Windows are independent alignments, so they run as one batch of
        # [B * K, window, D] sequences: [B, K, D, w] -> [B * K, w, D].
        w = self.window_size
        pred_win = pred_seq.unfold(1, w, self.stride_size).permute(0, 1, 3, 2)
        tgt_win = tgt_seq.unfold(1, w, self.stride_size).permute(0, 1, 3, 2)
        num_windows = pred_win.shape[1]
        dtw_dist = _soft_dtw_distance(
            pred_win.reshape(batch * num_windows, w, feat),
            tgt_win.reshape(batch * num_windows, w, feat),
            self.gamma,
        ).view(batch, num_windows)

        # Per-window batch mean, summed over windows.
        total_loss = dtw_dist.mean(dim=0).sum()
        if self.normalize:
            total_loss = total_loss / num_windows
        return total_loss


__all__ = ["SoftDTWLoss"]
