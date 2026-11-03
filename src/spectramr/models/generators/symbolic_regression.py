"""Symbolic Regression Generator for physics-guided MRI signal discovery.

A CNN backbone predicts per-pixel latent physical variables (one per declared unit key); the
symbolic head combines them through the ``num_symbolic_terms`` selected terms of a dimensionally
typed library (``symbolic_term_library``), so the closed-form terms are the reconstruction and
enter the training objective directly.

Reference:
    Cranmer, "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl",
    arXiv:2305.01582, 2023.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.generators.symbolic_term_library import (
    PhysicsSymbolicHead,
    build_symbolic_spec,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="symbolic_regression", training_mode="reconstruction", strict_kwargs=True)
class SymbolicRegressionGenerator(IGenerator, nn.Module):
    """Neural backbone + symbolic regression head for MRI signal discovery."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        feature_dim: int = 64,
        num_layers: int = 8,
        num_symbolic_terms: int = 5,
        type: str = "PhysicsPySR",
        constraints: dict[str, Any] | None = None,
        allowed_operators: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        spec = build_symbolic_spec(type, constraints, allowed_operators, num_symbolic_terms)

        # CNN backbone for feature extraction
        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                    nn.GroupNorm(min(32, feature_dim), feature_dim),
                    nn.SiLU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.symbolic_head = PhysicsSymbolicHead(feature_dim, spec)
        self.out_proj = nn.Conv2d(1, out_channels, 1)

    @property
    def solver_record(self) -> dict[str, Any]:
        """The symbolic head's declaration and term library, stamped into provenance."""
        return self.symbolic_head.spec.record()

    @property
    def name(self) -> str:
        return "symbolic_regression"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for SymbolicRegressionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.backbone(self.intro(x))  # [B, D, H, W]
        return self.out_proj(self.symbolic_head(h))

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
