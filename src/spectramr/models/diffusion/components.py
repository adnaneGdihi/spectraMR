#!/usr/bin/env python3
"""Diffusion Components Module
===========================

This module provides diffusion model components and U-Net architectures
for backward compatibility with existing tests and code.
"""

import torch

from spectramr.models.blocks.block_registry import (
    DiffusionAttentionStrategy as AttentionBlock,
)
from spectramr.models.diffusion.architectures.score_based_unet import ScoreBasedUNet


class StandardDiffusionUNet(ScoreBasedUNet):
    """Standard Diffusion U-Net with parameter mapping for backward compatibility."""

    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128, **kwargs):
        # Map parameter names to ScoreBasedUNet parameters
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=time_emb_dim,
            **kwargs,
        )
        # Store in_channels and out_channels for backward compatibility
        self.in_channels = in_channels
        self.out_channels = out_channels

    def get_parameter_count(self):
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape):
        """Get output shape for given input shape."""
        # Assume same spatial dimensions for diffusion models
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])


class DiffusionPrior(torch.nn.Module):
    """Diffusion Prior for MRI Reconstruction.

    This module implements a diffusion model that acts as a prior for
    reconstruction tasks, providing regularization and guidance during
    the reconstruction process. It can be used in conjunction with
    physics-based reconstruction methods.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        guidance_scale=1.0,
        num_inference_steps=50,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_emb_dim (Any): Description.
            guidance_scale (Any): Description.
            num_inference_steps (Any): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # Use the standard diffusion U-Net as the backbone
        self.denoising_model = StandardDiffusionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            **kwargs,
        )

        # Initialize diffusion process
        from spectramr.models.diffusion.base_diffusion import Diffusion

        self.diffusion = Diffusion(timesteps=1000, device="cpu")

    def forward(self, x, timestep=None):
        """Forward pass through the denoising model.

        forward method for DiffusionPrior.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            timestep (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if timestep is None:
            # If no timestep provided, assume we're denoising
            timestep = torch.randint(
                0,
                self.diffusion.timesteps,
                (x.shape[0],),
                device=x.device,
            )

        return self.denoising_model(x, timestep)

    def sample_prior(self, shape, device="cpu", guidance_fn=None):
        """Sample from the diffusion prior.

        Args:
            shape: Shape of output tensor (batch_size, channels, height, width)
            device: Device to place the tensor on
            guidance_fn: Optional guidance function for conditional sampling

        Returns:
            Generated samples from the prior

        """
        # Start from noise
        x = torch.randn(shape, device=device)

        # Denoise step by step
        self.diffusion.to(device)

        for t in reversed(range(self.num_inference_steps)):
            timestep = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.denoising_model(x, timestep)

            # Apply guidance if provided
            if guidance_fn is not None:
                # Compute gradient of guidance function
                with torch.enable_grad():
                    x_detached = x.detach().requires_grad_(True)
                    guidance_score = guidance_fn(x_detached)
                    guidance_grad = torch.autograd.grad(
                        guidance_score.sum(),
                        x_detached,
                    )[0]
                predicted_noise = predicted_noise - (self.guidance_scale * guidance_grad)

            # Denoise step
            x = self.diffusion.denoise_step(x, predicted_noise, timestep)

        return x

    def compute_prior_loss(self, x, target=None):
        """Compute the prior loss for regularization.

        Args:
            x: Input tensor to regularize
            target: Optional target for supervised regularization

        Returns:
            Prior regularization loss

        """
        # Add noise to the input
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0,
            self.diffusion.timesteps,
            (x.shape[0],),
            device=x.device,
        )

        # Forward diffusion
        noisy_x = self.diffusion.q_sample(x, timesteps, noise)

        # Predict noise
        predicted_noise = self.denoising_model(noisy_x, timesteps)

        # Compute denoising loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Add guidance loss if target is provided
        if target is not None:
            # Encourage predictions to be close to target
            guidance_loss = torch.nn.functional.mse_loss(x, target)
            loss = loss + 0.1 * guidance_loss

        return loss

    def get_parameter_count(self):
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape):
        """Get output shape for given input shape."""
        # Assume same spatial dimensions
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])


__all__ = [
    "AttentionBlock",
    "DiffusionPrior",
    "StandardDiffusionUNet",
]
