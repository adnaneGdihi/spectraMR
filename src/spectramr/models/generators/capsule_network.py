"""Capsule Network Generator for disentangled MRI reconstruction.

Uses capsule layers with dynamic routing to learn part-whole spatial
relationships and disentangled representations of MRI anatomy.

Reference:
    Sabour et al., "Dynamic Routing Between Capsules", NeurIPS 2017.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _PrimaryCapsules(nn.Module):
    """Convert CNN features to primary capsule vectors."""

    def __init__(self, in_ch: int, num_capsules: int, capsule_dim: int) -> None:
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_ch, num_capsules * capsule_dim, 3, padding=1)

    def _squash(self, x: torch.Tensor) -> torch.Tensor:
        sq_norm = (x**2).sum(dim=-1, keepdim=True)
        # Epsilon inside the root: sqrt has an infinite derivative at a zero norm.
        scale = sq_norm / (1 + sq_norm) / (sq_norm + 1e-8).sqrt()
        return scale * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _PrimaryCapsules.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, _, H, W = x.shape
        h = self.conv(x)  # [B, N*D, H, W]
        h = h.reshape(B, self.num_capsules, self.capsule_dim, H, W)
        h = h.permute(0, 3, 4, 1, 2).reshape(B * H * W, self.num_capsules, self.capsule_dim)
        return self._squash(h), (B, H, W)


class _RoutingCapsules(nn.Module):
    """Dynamic routing between capsules."""

    def __init__(
        self, in_caps: int, in_dim: int, out_caps: int, out_dim: int, routing_iters: int = 3
    ) -> None:
        super().__init__()
        self.routing_iters = routing_iters
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.randn(1, in_caps, out_caps, out_dim, in_dim) * 0.01)

    def _squash(self, x: torch.Tensor) -> torch.Tensor:
        sq_norm = (x**2).sum(dim=-1, keepdim=True)
        # Epsilon inside the root: sqrt has an infinite derivative at a zero norm.
        scale = sq_norm / (1 + sq_norm) / (sq_norm + 1e-8).sqrt()
        return scale * x

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """forward method for _RoutingCapsules.

        Executes PyTorch tensor operations.

        Args:
            u (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # u: [BHW, in_caps, in_dim]
        N = u.shape[0]
        # Per-sample prediction vectors u_hat[n,i,j] = W[i,j] @ u[n,i] (Sabour 2017).
        u_hat = torch.einsum("bil,ijkl->bijk", u, self.W[0])
        # u_hat: [N, in_caps, out_caps, out_dim]

        b = torch.zeros(N, u.shape[1], self.out_caps, device=u.device, dtype=u.dtype)
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=-1)  # [N, in, out]
            s = torch.einsum("bij,bijd->bjd", c, u_hat)  # [N, out, dim]
            v = self._squash(s)
            if _ < self.routing_iters - 1:
                b = b + torch.einsum("bjd,bijd->bij", v, u_hat)
        return v  # [N, out_caps, out_dim]


@register_model(name="capsule_network", training_mode="reconstruction")
class CapsuleNetworkGenerator(IGenerator, nn.Module):
    """Capsule-based generator for disentangled MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 32)
        num_capsules: int = kwargs.get("num_capsules", 8)
        capsule_dim: int = kwargs.get("capsule_dim", 16)
        routing_iters: int = kwargs.get("routing_iterations", 3)

        # CNN encoder for initial features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim * 2), feature_dim * 2),
            nn.SiLU(),
        )

        self.primary_caps = _PrimaryCapsules(feature_dim * 2, num_capsules, capsule_dim)
        self.routing = _RoutingCapsules(
            num_capsules, capsule_dim, num_capsules, capsule_dim, routing_iters
        )

        # Decoder to reconstruct from capsule representation
        self.decoder = nn.Sequential(
            nn.Conv2d(num_capsules * capsule_dim, feature_dim * 2, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim * 2), feature_dim * 2),
            nn.SiLU(),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, out_channels, 3, padding=1),
        )

    @property
    def name(self) -> str:
        return "capsule_network"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for CapsuleNetworkGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        feat = self.encoder(x)
        primary, (_, pH, pW) = self.primary_caps(feat)  # [BHW, caps, dim]
        routed = self.routing(primary)  # [BHW, caps, dim]
        # Reshape back to spatial
        routed = routed.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, caps*dim, H, W]
        return self.decoder(routed)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
