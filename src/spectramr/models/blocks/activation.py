"""Activation Functions
===================

Custom activation functions used across different model architectures.
"""

import torch
import torch.nn.functional as F
from torch import nn


def swish_jit(x: torch.Tensor) -> torch.Tensor:
    """Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def mish_jit(x: torch.Tensor) -> torch.Tensor:
    """Mish activation: x * tanh(softplus(x))."""
    return x * torch.tanh(F.softplus(x))


def complex_mod_relu_jit(
    real: torch.Tensor, imag: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Complex ModReLU activation."""
    # Calculate magnitude
    mag = torch.sqrt(real * real + imag * imag + 1e-8)

    # Apply ModReLU: ReLU(|z| + b)
    mod_mag = F.relu(mag + bias)

    # Rescale: z_out = mod_mag * (z / |z|) = z * (mod_mag / |z|)
    scale = mod_mag / (mag + 1e-8)

    real_out = real * scale
    imag_out = imag * scale

    # Return interleaved format: [R1, I1, R2, I2, ...]
    # Use flatten(1, 2) for TorchScript compatibility instead of unpacking shape
    return torch.stack([real_out, imag_out], dim=2).flatten(1, 2)


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Swish.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return swish_jit(x)


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Mish.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return mish_jit(x)


class ComplexActivation(nn.Module):
    """Complex-aware activation function (ModReLU).

    Applies ReLU to the magnitude of the complex number while preserving phase.
    ModReLU(z) = ReLU(|z| + b) * (z / |z|)

    Args:
        channels: Number of complex channels (input tensor has 2*channels).
        bias_init: Initial value for the learnable bias 'b'.
    """

    def __init__(self, channels: int, bias_init: float = 0.0):
        """__init__.

        Args:
            channels (int): Description.
            bias_init (float): Description.
        """
        super().__init__()
        self.bias = nn.Parameter(torch.full((1, channels, 1, 1), bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 2*C, H, W).
               Assumes interleaved layout: [R1, I1, R2, I2, ...]
        """
        B, C_total, H, W = x.shape
        C = C_total // 2

        # Split into real and imaginary parts (Interleaved parsing)
        real = x[:, 0::2, :, :]
        imag = x[:, 1::2, :, :]

        # b > 0 adds a phase-coherent floor of magnitude b at every bin; on k-space that is a
        # Dirac at the image centre (the crosshair), so only the dead-zone side is allowed.
        return complex_mod_relu_jit(real, imag, self.bias.clamp(max=0.0))


__all__ = ["ComplexActivation", "Mish", "Swish"]
