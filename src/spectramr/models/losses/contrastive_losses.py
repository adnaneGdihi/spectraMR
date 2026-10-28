from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spectramr.models.losses.registry import register_loss


@register_loss(name="contrastive", aliases=["ContrastiveLoss"])
class ContrastiveLoss(nn.Module):
    r"""InfoNCE-style contrastive loss with temperature scaling.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Contrastive} = - \mathbb{E} \left[ \log \frac{\exp(sim(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(sim(z_i, z_k)/\tau)} \right]"""

    def __init__(self, temperature: float = 0.07):
        """Initialize contrastive loss.

        Args:
            temperature: Softmax temperature parameter.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: Float[Tensor, "B D"],
        labels: Int[Tensor, " B"] | None = None,  # noqa: F722
        mask: Bool[Tensor, "B B"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute InfoNCE loss.

        Args:
            features: Input feature vectors.
            labels: Optional labels for supervised contrastive learning.
                If provided, mask is generated where labels match.
            mask: Optional boolean mask where mask[i, j] is True if i and j
                are positive pairs.

        Returns:
            Scalar loss value.

        forward method for ContrastiveLoss.

        Executes PyTorch tensor operations.

        Args:
            features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            labels (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        # [B, B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask if labels provided
        if labels is not None:
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(device)

        if mask is None:
            # Standard SimCLR mask (positives are diagonals, but we self-mask)
            # This path assumes features has 2 views per sample interleaved or stacked
            # Defaulting to identity mask which doesn't make sense for loss
            # forcing user to provide labels or mask for generalized usage
            raise ValueError("Must provide labels or mask for ContrastiveLoss")

        # Mask out self-contrast
        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        logits_mask = (~self_mask).to(features.dtype)
        mask = mask * logits_mask  # Remove diagonal from positives

        # Loss = -1/|P(i)| * sum_{p in P(i)} log( exp(sim(i,p)) / sum_{a in A(i)} exp(sim(i,a)) ),
        # the denominator over every sample except self. It is taken as a
        # logsumexp because exp(1/tau) overflows float32 once tau < ~0.011; the
        # self term is excluded with the dtype's finite minimum rather than -inf
        # so a single-sample row stays finite in value and gradient.
        excluded = torch.finfo(similarity_matrix.dtype).min
        log_denom = torch.logsumexp(
            similarity_matrix.masked_fill(self_mask, excluded), dim=1, keepdim=True
        )
        log_prob = (similarity_matrix - log_denom).masked_fill(self_mask, 0.0)

        # Compute mean of log-likelihood over positive pairs
        # mask * log_prob selects relevant pairs
        # Sum over j, then average over i (only for i that have positives)

        mask_sum = mask.sum(dim=1)
        # Avoid division by zero
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob_pos.mean()

        return loss


@register_loss(name="contrastive_anatomy_disentanglement", aliases=["CAContrastiveLoss", "cacd"])
class CAContrastiveLoss(nn.Module):
    r"""Contrastive Anatomy-Contrast Disentanglement (CACD) Loss.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{CAContrastive} = \lambda_{geo} \mathcal{L}_{con}(z_{geo}) + \lambda_{phy} \mathcal{L}_{con}(z_{phy})"""

    def __init__(
        self,
        lambda_geo: float = 1.0,
        lambda_phy: float = 1.0,
        temperature: float = 0.07,
    ):
        """Initialize CACD loss.

        Args:
            lambda_geo: Weight for anatomy loss.
            lambda_phy: Weight for physics loss.
            temperature: Contrastive temperature.
        """
        super().__init__()
        self.lambda_geo = lambda_geo
        self.lambda_phy = lambda_phy
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

    def forward(
        self,
        z_geo: Float[Tensor, "B D_geo"],
        z_phy: Float[Tensor, "B D_phy"],
        subject_ids: Int[Tensor, " B"],  # noqa: F722
        contrast_ids: Int[Tensor, " B"],  # noqa: F722
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """Compute disentanglement losses.

        Args:
            z_geo: Anatomical embeddings.
            z_phy: Physics embeddings.
            subject_ids: Tensor of subject IDs.
            contrast_ids: Tensor of contrast/sequence IDs.

        Returns:
            Tuple of (total_loss, loss_geo, loss_phy).

        forward method for CAContrastiveLoss.

        Executes PyTorch tensor operations.

        Args:
            z_geo (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z_phy (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            subject_ids (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            contrast_ids (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[Float[Tensor, ''], Float[Tensor, ''], Float[Tensor, '']]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Anatomy Loss: Positives are same subject (invariant to contrast)
        # labels = subject_ids
        loss_geo = self.contrastive_loss(z_geo, labels=subject_ids)

        # Physics Loss: Positives are same contrast (invariant to subject)
        # labels = contrast_ids
        loss_phy = self.contrastive_loss(z_phy, labels=contrast_ids)

        total_loss = self.lambda_geo * loss_geo + self.lambda_phy * loss_phy

        return total_loss, loss_geo, loss_phy
