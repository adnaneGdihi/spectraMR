"""KIDOT: Knowledge-Informed Dynamic Optimal Transport.

This module implements Unpaired Domain Adaptation using Wasserstein Distance
(Earth Mover's Distance) with Gradient Penalty (WGAN-GP).
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spectramr.models.registry import register_model


# ``input_domain`` is deliberately LEFT UNDECLARED (#1920). The 3-D conv stack below would
# read as ``image``, but nothing in the tree instantiates this class -- the only reference
# is ``transport.py``'s own function parameter -- so there is no call site to read the
# intended input space off. Declaring one from the architecture alone would be inventing
# a contract no caller has agreed to (NN18).
@register_model(role="discriminator", name="wasserstein_discriminator", training_mode="gan")
class WassersteinDiscriminator(nn.Module):
    """Critic network for estimating Wasserstein Distance."""

    def __init__(self, in_channels: int = 1, ndf: int = 64):
        """__init__.

        Args:
            in_channels (int): Description.
            ndf (int): Description.
        """
        super().__init__()

        # Simple 3D PatchGAN-style discriminator
        self.main = nn.Sequential(
            # Input is (C, D, H, W)
            nn.Conv3d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # WGAN-GP typically avoids BatchNorm
            nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, 1, 4, 1, 0, bias=False),
            # No Sigmoid for Wasserstein Critic!
        )

    def forward(self, x: Float[Tensor, "B C D H W"]) -> Float[Tensor, "B 1 D2 H2 W2"]:
        """forward.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
        Returns:
            Float[Tensor, 'B 1 D2 H2 W2']: Description.

        forward method for WassersteinDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B 1 D2 H2 W2']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.main(x)


class KidotTransport:
    """Manager for Optimal Transport Domain Adaptation."""

    def __init__(
        self,
        discriminator: WassersteinDiscriminator,
        optimizer_D: torch.optim.Optimizer,
        lambda_gp: float = 10.0,
    ):
        """__init__.

        Args:
            discriminator (WassersteinDiscriminator): Description.
            optimizer_D (torch.optim.Optimizer): Description.
            lambda_gp (float): Description.
        """
        self.discriminator = discriminator
        self.optimizer_D = optimizer_D
        self.lambda_gp = lambda_gp

    def compute_gradient_penalty(
        self, real_samples: Tensor, fake_samples: Tensor, device: torch.device
    ) -> Tensor:
        """Calculates the gradient penalty loss for WGAN GP."""
        # Random weight term for interpolation between real and fake samples
        B = real_samples.size(0)
        alpha = torch.rand(B, 1, 1, 1, 1, device=device)

        # Get random interpolation
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        d_interpolates = self.discriminator(interpolates)

        fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_critic(self, real_data: Tensor, fake_data: Tensor) -> float:
        """Updates the critic (discriminator) to approximate Wasserstein distance."""
        self.optimizer_D.zero_grad()

        real_validity = self.discriminator(real_data)
        fake_validity = self.discriminator(fake_data.detach())

        # Wasserstein Loss: D(x) - D(G(z))
        # We minimize -Loss, i.e., Minimize D(G(z)) - D(x)
        # Or Maximize D(x) - D(G(z))
        loss_D = -torch.mean(real_validity) + torch.mean(fake_validity)

        # Gradient Penalty
        gp = self.compute_gradient_penalty(real_data, fake_data, real_data.device)

        total_loss = loss_D + self.lambda_gp * gp
        total_loss.backward()
        self.optimizer_D.step()

        return total_loss.item()

    def calculate_generator_loss(self, fake_data: Tensor) -> Tensor:
        """Calculates loss for the generator (reconstruction network).

        Generator wants to maximize D(G(z)), i.e., minimize -D(G(z)).
        """
        fake_validity = self.discriminator(fake_data)
        return -torch.mean(fake_validity)
