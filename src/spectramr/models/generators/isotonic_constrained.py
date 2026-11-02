"""Isotonic Constrained Generator for monotonicity-aware MRI reconstruction.

The output is a non-decreasing function of the input: if ``x' >= x`` pointwise then
``f(x') >= f(x)`` pointwise. Every convolution uses non-negative weights
(softplus-reparameterised), every activation is non-decreasing, and no layer
subtracts a data-dependent mean, so the composition is monotone by construction.

Reference:
    J. Sill, "Monotonic Networks", NIPS 1998.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _NonNegConv2d(nn.Module):
    """2-D convolution whose effective weight ``softplus(raw)`` is non-negative."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        fan_in = in_ch * kernel_size * kernel_size
        # Initialise near a local average (weights ~ 1/fan_in) so depth does not
        # blow the activations up, with jitter to break symmetry.
        raw = torch.full(
            (out_ch, in_ch, kernel_size, kernel_size), math.log(math.expm1(1.0 / fan_in))
        )
        self.raw_weight = nn.Parameter(raw + 0.1 * torch.randn_like(raw))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, F.softplus(self.raw_weight), self.bias, padding=self.padding)


class _IsotonicLayer(nn.Module):
    """Monotone gated residual: ``x + g(x) * t(x)`` with ``g, t`` non-negative and non-decreasing."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gate = _NonNegConv2d(ch, ch, 1)
        self.transform = _NonNegConv2d(ch, ch, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the monotone gated residual to a ``(B, C, H, W)`` feature map."""
        # A product of non-negative non-decreasing maps is non-decreasing.
        return x + F.softplus(self.gate(x)) * F.softplus(self.transform(x))


@register_model(name="isotonic_constrained", training_mode="reconstruction")
class IsotonicConstrainedGenerator(IGenerator, nn.Module):
    """Reconstruction network whose output is non-decreasing in its input."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 64)
        num_layers: int = kwargs.get("num_layers", 8)
        num_constraints: int = kwargs.get("num_constraints", 4)

        # GroupNorm and SiLU are gone: mean subtraction and SiLU's dip both break
        # monotonicity (#2404). LeakyReLU is strictly increasing.
        self.intro = _NonNegConv2d(in_channels, feature_dim, 3)

        blocks: list[nn.Module] = []
        for i in range(num_layers):
            blocks.append(_NonNegConv2d(feature_dim, feature_dim, 3))
            blocks.append(nn.LeakyReLU(0.1))
            if (i + 1) % max(1, num_layers // num_constraints) == 0:
                blocks.append(_IsotonicLayer(feature_dim))
        self.backbone = nn.Sequential(*blocks)

        self.out_conv = _NonNegConv2d(feature_dim, out_channels, 3)

    @property
    def name(self) -> str:
        return "isotonic_constrained"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for IsotonicConstrainedGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.out_conv(self.backbone(self.intro(x)))

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
