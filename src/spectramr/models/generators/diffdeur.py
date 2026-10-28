"""DiffDeuR: Zero-Shot VLF Enhancement
=====================================

Implementation of Experiment 2.1 from the Research Roadmap.
Uses Unconditional Diffusion Prior and Implicit Neural Representation (INR)
for zero-shot super-resolution/enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS
from spectramr.models.diffusion.components import StandardDiffusionUNet
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class ImplicitNeuralRepresentation(nn.Module):
    """Coordinate-based MLP for continuous image representation."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4):
        """__init__.

        Args:
            hidden_dim (int): Description.
            num_layers (int): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2D coordinates (x, y)
            nn.SiLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, 1),  # Intensity value
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Query INR at given coordinates.

        Args:
            coords: [B, N, 2] coordinates in [-1, 1]

        Returns:
            Intensity values [B, N, 1]

        forward method for ImplicitNeuralRepresentation.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(coords)


@register_model(name="diffdeur", training_mode="reconstruction")
class DiffDeuR(nn.Module, IGenerator):
    """DiffDeuR Zero-Shot Enhancement Pipeline."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        _prior_model_path: str = None,  # Path to pre-trained HF diffusion model
        hidden_dim: int = 256,
        num_timesteps: int = DEFAULT_DIFFUSION_TIMESTEPS,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            _prior_model_path (str): Description.
            hidden_dim (int): Description.
            num_timesteps (int): Horizon the prior was trained on; SDS draws t from it.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = int(num_timesteps)

        # 1. Generative Prior (Pre-trained Diffusion Model)
        # In a real scenario, we would load weights here.
        # For now, we instantiate a fresh model.
        self.prior = StandardDiffusionUNet(
            in_channels=out_channels, out_channels=out_channels, time_emb_dim=256
        )
        # Freeze prior
        for param in self.prior.parameters():
            param.requires_grad = False

        # 2. INR (Optimized at inference time)
        # We don't initialize INR here because it's per-scan.
        # But we provide a factory method or class for it.
        self.inr_cls = ImplicitNeuralRepresentation
        self.hidden_dim = hidden_dim

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "DiffDeuR"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.prior.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        If timesteps is provided, delegates to the diffusion prior (for pre-training).
        Otherwise, runs the full zero-shot enhancement pipeline via INR
        optimisation + score distillation sampling.

        Args:
            x: Input tensor — noisy image for training, VLF image for inference.
            timesteps: Diffusion timesteps (training mode only).

        forward method for DiffDeuR.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if timesteps is not None:
            return self.prior(x, timesteps)

        # Inference: run zero-shot enhancement
        return self.enhance(x)

    def enhance(
        self, vlf_image: torch.Tensor, num_steps: int = 100, lr: float = 1e-3
    ) -> torch.Tensor:
        """Run Zero-Shot Enhancement Optimization.

        Args:
            vlf_image: Low-field input image [1, 1, H, W]
            num_steps: Optimization steps

        Returns:
            Enhanced HF image
        """
        B, C, H, W = vlf_image.shape
        device = vlf_image.device

        from spectramr.infrastructure.training.builders import OptimizationBuilder

        # Initialize INR
        inr = self.inr_cls(hidden_dim=self.hidden_dim).to(device)
        optimizer = OptimizationBuilder.create_single_optimizer(inr.parameters(), learning_rate=lr)

        # Create coordinates grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1).reshape(1, -1, 2)  # [1, H*W, 2]

        # Optimization loop with Score Distillation Sampling
        for step in range(num_steps):
            optimizer.zero_grad()

            # 1. Query INR for current reconstruction
            reconstructed = inr(coords)  # [1, H*W, 1]
            reconstructed = reconstructed.reshape(B, C, H, W)

            # 2. Data Fidelity Loss with resolution-aware degradation operator D
            # Model D as: blur + downsample (common degradation model)
            # D(x) = downsample(blur(x, σ), scale_factor)

            # Compute scale factor from VLF resolution
            vlf_h, vlf_w = vlf_image.shape[-2:]
            scale_factor = max(1, H // vlf_h, W // vlf_w)

            # Apply Gaussian blur (simulate optical blur)
            blur_sigma = scale_factor * 0.5
            kernel_size = int(4 * blur_sigma + 1) | 1  # Ensure odd

            # Create Gaussian kernel
            x_grid = torch.arange(kernel_size, device=device).float() - kernel_size // 2
            kernel_1d = torch.exp(-(x_grid**2) / (2 * blur_sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)

            # Apply blur and downsample
            padding = kernel_size // 2
            blurred = F.conv2d(reconstructed, kernel_2d, padding=padding, groups=C)
            degraded = F.interpolate(
                blurred, size=(vlf_h, vlf_w), mode="bilinear", align_corners=False
            )

            # Compute data fidelity loss
            data_loss = F.mse_loss(degraded, vlf_image)

            # 3. Score Distillation Sampling (SDS) Loss from Diffusion Prior
            # This encourages the reconstruction to match the manifold learned by the prior
            # SDS: ∇_θ ||ε_θ(z_t, t) - ε||^2 where z_t = √α_t z_0 + √(1-α_t) ε

            # Sample random timestep
            t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

            # Add noise to reconstruction
            noise = torch.randn_like(reconstructed)

            # Simple linear noise schedule (same as in prior training)
            alpha_t = 1 - t.float() / self.num_timesteps
            alpha_t = alpha_t.view(-1, 1, 1, 1)

            noisy_img = torch.sqrt(alpha_t) * reconstructed + torch.sqrt(1 - alpha_t) * noise

            # Predict noise using diffusion prior
            with torch.no_grad():
                noise_pred = self.prior(noisy_img, t)

            # SDS gradient (DreamFusion paper - Poole et al., ICLR 2023)
            # ∇_θ L_SDS = w(t) * (ε_φ(z_t, t) - ε)
            # We use a surrogate loss that produces the correct gradient
            # without backpropagating through the UNet
            w = torch.sqrt(alpha_t)  # Weighting term
            grad = w * (noise_pred - noise)

            # Surrogate: pretend loss is 0.5 * ||z - target||^2 where target = z - grad
            # This gives ∇_z L = z - target = grad (exactly what we want)
            target = (reconstructed - grad).detach()
            sds_loss = 0.5 * F.mse_loss(reconstructed, target)

            # 4. Total Loss
            total_loss = data_loss + 0.1 * sds_loss  # Weight SDS lower

            total_loss.backward()
            optimizer.step()

        # Final reconstruction
        with torch.no_grad():
            enhanced = inr(coords).reshape(B, C, H, W)

        return enhanced

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate interface for IGenerator."""
        # For DiffDeuR, 'z' is treated as VLF input
        return self.enhance(z, **kwargs)
