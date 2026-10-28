"""Gradient reversal -- the canonical home (NN6).

Identity in the forward pass, ``-alpha * grad`` in the backward pass, so that minimising
a downstream domain loss *removes* domain information from the features upstream
(Ganin & Lempitsky 2015; Ganin et al. 2016, JMLR 17(1):1-35).

This module is the single owner of that invariant (NN17): anything needing reversal
imports from here instead of defining its own ``autograd.Function``. The AST walk in
``tests/unit/models/blocks/test_domain_adaptation.py`` fails on a second definition
anywhere under ``src/spectramr``.

A wrapper that only decides *how alpha is chosen* is an adapter, not a duplicate, and may
live wherever its schedule does -- the DANN sigmoid ramp in
``models/losses/domain_adaptation_loss.py`` and the ``lambda_`` spelling in
``infrastructure/training/strategies/privileged_learning_strategy.py`` are both of that
kind. What the detector forbids is a second implementation of the reversal itself.
"""

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward, ``-alpha * grad_output`` backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Functional form of :class:`GradientReversalFunction`.

    ``alpha`` is forwarded uncoerced: a caller holding it as a tensor would pay a device
    sync for ``float()``, which NN9 forbids on the training path.
    """
    return GradientReversalFunction.apply(x, alpha)


class GradientReversalLayer(nn.Module):
    """Module form, with ``alpha`` chosen by the caller.

    Typically annealed 0 -> 1 during training via :meth:`set_alpha`. For the DANN sigmoid
    schedule driven by an internal iteration counter, use ``GradientReversalLayerModule``
    from ``models/losses/domain_adaptation_loss.py`` instead.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        """Update reversal strength, for an externally-driven schedule."""
        self.alpha = alpha

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class DomainClassifier(nn.Module):
    """A simple domain classifier for DANN.
    It takes feature maps from a backbone network and predicts the domain.
    """

    def __init__(self, in_channels: int, num_domains: int = 2):
        """__init__.

        Args:
            in_channels (int): Description.
            num_domains (int): Description.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_domains),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            alpha (float): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DomainClassifier.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            alpha (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = grad_reverse(x, alpha)
        domain_output = self.classifier(features)
        return domain_output


class nn_Flatten(nn.Module):
    """nn_Flatten class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for nn_Flatten.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x.view(x.size(0), -1)
