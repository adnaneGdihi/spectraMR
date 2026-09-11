"""Domain-Adversarial Loss for Field Disentanglement.

Forces an encoder to produce field-invariant latent representations by
adversarially confusing a domain classifier via gradient negation. The reversal
itself lives in :mod:`spectramr.models.blocks.domain_adaptation`, its canonical
home (NN6); this module supplies the classifier and the loss built around it.

Mathematical Foundation:
    L_GRL = min_{E_anat} max_{D_field} E[BCE(D_field(E_anat(x)), field_label)]

    During forward pass, GRL acts as identity. During backward pass, it
    negates gradients by -α, forcing the encoder to maximally confuse
    the field-strength classifier → field-invariant anatomy codes.

References:
    - Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural
      Networks." JMLR 17(1):1-35.
    - Kamnitsas, K. et al. (2017). "Unsupervised domain adaptation in
      brain lesion segmentation with adversarial networks." MICCAI.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.models.blocks.domain_adaptation import GradientReversalLayer
from spectramr.models.losses.registry import register_loss

logger = logging.getLogger(__name__)


class FieldDomainClassifier(nn.Module):
    """Binary classifier predicting field strength from latent features.

    Args:
        in_features: Dimension of the latent code.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, in_features: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict field strength probability.

        Args:
            z: Latent features [B, D] or [B, D, H, W].

        Returns:
            Logits [B, 1].

        forward method for FieldDomainClassifier.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if z.ndim == 4:
            # Global average pool spatial dims
            z = z.mean(dim=(-2, -1))
        return self.net(z)


@register_loss(name="domain_adversarial_grl", aliases=["domain_adv_grl"], domain="latent")
class DomainAdversarialLoss(nn.Module):
    """Domain-adversarial loss combining GRL + field classifier.

    Optimized jointly with the main reconstruction loss. The GRL
    negates classifier gradients, forcing the encoder to produce
    field-invariant representations.

    Args:
        latent_dim: Dimension of the encoder's latent space.
        hidden_dim: Hidden dimension for the field classifier.
        alpha: Initial GRL strength.
        loss_weight: Weight for the adversarial loss term.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{DomainAdversarial} = \text{BCE}(D_{domain}(GRL(z)), y_{domain})"""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        alpha: float = 1.0,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.classifier = FieldDomainClassifier(latent_dim, hidden_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_weight = loss_weight

    def forward(
        self,
        z_anat: torch.Tensor,
        field_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute domain-adversarial loss.

        Args:
            z_anat: Anatomical latent features [B, D] or [B, D, H, W].
            field_labels: Binary field labels [B, 1] (0=ULF, 1=HF).

        Returns:
            Dict with 'domain_loss' (weighted) and 'domain_acc' (accuracy).

        forward method for DomainAdversarialLoss.

        Executes PyTorch tensor operations.

        Args:
            z_anat (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            field_labels (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Apply gradient reversal
        z_reversed = self.grl(z_anat)

        # Classify field strength
        logits = self.classifier(z_reversed)

        # BCE loss
        loss = self.criterion(logits, field_labels.float())

        # Accuracy for monitoring
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == field_labels.float()).float().mean()

        return {
            "domain_loss": loss * self.loss_weight,
            "domain_acc": acc,
        }

    def set_alpha(self, alpha: float) -> None:
        """Update GRL strength (e.g., for annealing schedule)."""
        self.grl.set_alpha(alpha)

    def anneal_alpha(self, progress: float, gamma: float = 10.0) -> float:
        """Ganin et al. sigmoid annealing schedule.

        Args:
            progress: Training progress in [0, 1].
            gamma: Steepness of the sigmoid.

        Returns:
            Current alpha value.
        """
        import math

        alpha = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
        self.set_alpha(alpha)
        return alpha


__all__ = [
    "DomainAdversarialLoss",
    "FieldDomainClassifier",
]
