from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.domain_adaptation import grad_reverse
from spectramr.models.losses.registry import register_loss


@dataclass
class DomainAdversarialConfig:
    """Configuration for domain adversarial loss."""

    lambda_domain: float = 1.0
    grl_max_iters: int = 1000
    grl_gamma: float = 10.0
    domain_classifier_hidden: int = 1024
    domain_classifier_dropout: float = 0.5
    stability_epsilon: float = 1e-6
    gradient_penalty_weight: float = 10.0


class IDomainAdversarialLoss(Protocol):
    """Protocol for domain adversarial loss computation.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{DomainAdversarial} = \text{BCE}(D_{domain}(GRL(z)), y_{domain})"""

    def compute_domain_loss(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain adversarial loss."""
        ...

    def get_domain_accuracy(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> float:
        """Get domain classification accuracy."""
        ...


class GradientReversalLayerModule(nn.Module):
    """Gradient Reversal Layer module wrapper.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{GradientReversal}(x) = \begin{cases} x & \text{forward} \\ -\alpha \frac{\\partial L}{\\partial x} & \text{backward} \\end{cases}"""

    def __init__(self, max_iters: int = 1000, gamma: float = 10.0):
        """__init__.

        Args:
            max_iters (int): Description.
            gamma (float): Description.
        """
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.iter_num = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal.

        forward method for GradientReversalLayerModule.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        self.iter_num += 1
        coeff = self._get_coefficient()
        return grad_reverse(x, coeff)

    def _get_coefficient(self) -> float:
        """Get reversal coefficient based on training progress."""
        progress = min(self.iter_num / self.max_iters, 1.0)
        return 2.0 / (1.0 + np.exp(-self.gamma * progress)) - 1.0


@register_loss(name="domain_adversarial", aliases=["DomainAdversarialLoss"])
class DomainAdversarialLoss(IDomainAdversarialLoss):
    """Domain adversarial loss with Gradient Reversal Layer (GRL) and stability heuristics.

    Implements domain-adversarial training for unsupervised domain adaptation,
    with automatic stability monitoring and gradient penalty.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{DomainAdversarial} = \text{BCE}(D_{domain}(GRL(z)), y_{domain})"""

    def __init__(self, config: DomainAdversarialConfig, feature_dim: int):
        """__init__.

        Args:
            config (DomainAdversarialConfig): Description.
            feature_dim (int): Description.
        """
        self.config = config

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, config.domain_classifier_hidden),
            nn.ReLU(),
            nn.Dropout(config.domain_classifier_dropout),
            nn.Linear(
                config.domain_classifier_hidden,
                config.domain_classifier_hidden // 2,
            ),
            nn.ReLU(),
            nn.Dropout(config.domain_classifier_dropout),
            nn.Linear(config.domain_classifier_hidden // 2, 1),
        )

        # Gradient Reversal Layer
        self.grl = GradientReversalLayerModule(
            max_iters=config.grl_max_iters, gamma=config.grl_gamma
        )

        # Stability tracking
        self.domain_accuracy_history: list[float] = []
        self.gradient_norm_history: list[float] = []

    def compute_domain_loss(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain adversarial loss with stability monitoring."""
        # Apply GRL
        reversed_features = self.grl(features)

        # Domain classification
        domain_logits = self.domain_classifier(reversed_features)
        domain_loss = F.binary_cross_entropy_with_logits(
            domain_logits.squeeze(),
            domain_labels.float(),
        )

        # Add gradient penalty for stability
        if self.config.gradient_penalty_weight > 0:
            penalty = self._compute_gradient_penalty(reversed_features, domain_logits)
            domain_loss = domain_loss + self.config.gradient_penalty_weight * penalty

        # Stability monitoring
        with torch.no_grad():
            domain_preds = torch.sigmoid(domain_logits).squeeze()
            accuracy = ((domain_preds > 0.5) == domain_labels).float().mean().item()
            self.domain_accuracy_history.append(accuracy)

            # Keep only recent history
            if len(self.domain_accuracy_history) > 100:
                self.domain_accuracy_history = self.domain_accuracy_history[-100:]

        return domain_loss

    def _compute_gradient_penalty(
        self,
        features: torch.Tensor,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty for domain classifier."""
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=features,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def get_domain_accuracy(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> float:
        """Get domain classification accuracy."""
        with torch.no_grad():
            reversed_features = self.grl(features)
            domain_logits = self.domain_classifier(reversed_features)
            domain_preds = torch.sigmoid(domain_logits).squeeze()
            accuracy = ((domain_preds > 0.5) == domain_labels).float().mean().item()

        return accuracy
