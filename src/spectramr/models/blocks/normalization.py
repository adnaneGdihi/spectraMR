"""Normalization Building Blocks
============================

Common normalization layers used across different model architectures.
"""

import torch
from torch import nn


def get_adaptive_groups(channels: int, min_channels_per_group: int = 16) -> int:
    """Calculate the number of groups for group normalization."""
    for num_groups in range(min(32, channels), 0, -1):
        if channels % num_groups == 0:
            channels_per_group = channels // num_groups
            if channels_per_group >= min_channels_per_group or num_groups == 1:
                return num_groups
    return 1


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style transfer."""

    def __init__(self, channels: int, style_dim: int):
        """__init__.

        Args:
            channels (int): Description.
            style_dim (int): Description.
        """
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            style (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for AdaptiveInstanceNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        style_params = self.style(style)
        gamma, beta = style_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return torch.as_tensor(self.norm(x) * gamma + beta)


class ConditionalBatchNorm(nn.Module):
    """Conditional Batch Normalization for conditional generation."""

    def __init__(self, channels: int, condition_dim: int):
        """__init__.

        Args:
            channels (int): Description.
            condition_dim (int): Description.
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.condition = nn.Linear(condition_dim, channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            condition (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ConditionalBatchNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            condition (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        condition_params = self.condition(condition)
        gamma, beta = condition_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return torch.as_tensor(self.bn(x) * gamma + beta)


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization (AdaGN) for diffusion models."""

    def __init__(self, channels: int, num_groups: int, emb_channels: int):
        """__init__.

        Args:
            channels (int): Description.
            num_groups (int): Description.
            emb_channels (int): Description.
        """
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False)
        self.proj = nn.Linear(emb_channels, channels * 2)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for AdaptiveGroupNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if emb is None:
            return self.norm(x)
        params = self.proj(emb)
        scale, shift = params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1.0 + scale) + shift


class ComplexAdaptiveGroupNorm(nn.Module):
    """Complex-valued Adaptive Group Normalization.

    Handles Interleaved [R1, I1, R2, I2, ...] layout.
    Normalizes complex channels by their magnitude variance only.
    IMPORTANT: Does NOT subtract the mean to preserve the DC component in k-space.
    """

    def __init__(self, channels: int, num_groups: int, emb_channels: int):
        """__init__.

        Args:
            channels (int): Description.
            num_groups (int): Description.
            emb_channels (int): Description.
        """
        super().__init__()
        self.channels = channels  # Total real channels (2*C)
        self.num_groups = num_groups  # Groups for complex channels (C)
        self.eps = 1e-5

        # Scale only: a per-channel shift is a constant over the feature map, which on a
        # k-space block is a delta at the k centre and a cross-hair in the output image.
        C = channels // 2
        self.proj = nn.Linear(emb_channels, C)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ComplexAdaptiveGroupNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C2, H, W = x.shape
        C = C2 // 2

        # 1. Parse Interleaved Input
        real = x[:, 0::2, ...]
        imag = x[:, 1::2, ...]

        # 2. Normalize Magnitude Variance (WITHOUT mean subtraction)
        # Reshape for grouping: [B, G, C/G, H, W]
        G = self.num_groups
        real_g = real.view(B, G, C // G, H, W)
        imag_g = imag.view(B, G, C // G, H, W)

        # Magnitude variance: E[|z|^2] (relative to origin, not mean)
        # This preserves the DC energy (contrast) while stabilizing variance.
        var = (real_g.pow(2) + imag_g.pow(2)).mean(dim=(2, 3, 4), keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)

        real_norm = (real_g * inv_std).view(B, C, H, W)
        imag_norm = (imag_g * inv_std).view(B, C, H, W)

        # 3. Modulation
        if emb is None:
            # Identity modulation if no embedding
            real_out, imag_out = real_norm, imag_norm
        else:
            scale = self.proj(emb).view(B, C, 1, 1)
            real_out = (1.0 + scale) * real_norm
            imag_out = (1.0 + scale) * imag_norm

        # 4. Return Interleaved: [R1, I1, R2, I2, ...]
        return torch.stack([real_out, imag_out], dim=2).flatten(1, 2)


__all__ = [
    "AdaptiveGroupNorm",
    "AdaptiveInstanceNorm",
    "ComplexAdaptiveGroupNorm",
    "ConditionalBatchNorm",
    "get_adaptive_groups",
]
