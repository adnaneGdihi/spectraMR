"""Differentiable Rendering Generator for MRI slice simulation.

A convolutional backbone predicts a latent through-plane tissue volume (proton density and
partial-volume material fractions per sub-slice) and a differentiable slice-profile renderer
(:class:`~spectramr.models.blocks.diff_slice_sim.DiffSliceSim`) integrates it into the 2-D image
through the declared slice-select profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
import torch.nn as nn

from spectramr.models.blocks.diff_slice_sim import (
    RENDER_ENGINES,
    SLICE_PROFILES,
    build_renderer,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


def _conv_block(ch: int) -> list[nn.Module]:
    return [nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(min(32, ch), ch), nn.SiLU()]


def _non_negative_int(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"differentiable_rendering {key} must be a non-negative int, got {value!r}."
        )
    return value


@register_model(
    name="differentiable_rendering",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
    strict_kwargs=True,
)
class DifferentiableRenderingGenerator(IGenerator, nn.Module):
    """Backbone + residual refinement + DiffSliceSim slice-profile renderer."""

    OPTION_SCHEMA: ClassVar[dict[str, list[Any]]] = {
        "rendering.engine": sorted(RENDER_ENGINES),
        "rendering.physics.slice_profile": sorted(SLICE_PROFILES),
        "rendering.physics.partial_volume_effect": [True, False],
    }

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        feature_dim: int = 64,
        num_layers: int = 8,
        rendering_layers: int = 3,
        material_dim: int = 32,
        rendering: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if _non_negative_int("feature_dim", feature_dim) < 1:
            raise ValueError("differentiable_rendering feature_dim must be >= 1.")
        num_layers = _non_negative_int("num_layers", num_layers)
        rendering_layers = _non_negative_int("rendering_layers", rendering_layers)

        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        self.backbone = nn.Sequential(
            *[m for _ in range(num_layers) for m in _conv_block(feature_dim)]
        )
        # Residual refinement of the volume features before the renderer reads them.
        self.refine_stages = nn.ModuleList(
            [nn.Sequential(*_conv_block(feature_dim)) for _ in range(rendering_layers)]
        )
        self.renderer = build_renderer(
            rendering if rendering is not None else {},
            feature_dim=feature_dim,
            out_channels=out_channels,
            material_dim=material_dim,
        )
        self._arch = {
            "feature_dim": feature_dim,
            "num_layers": num_layers,
            "rendering_layers": rendering_layers,
        }

    @property
    def name(self) -> str:
        return "differentiable_rendering"

    @property
    def solver_record(self) -> dict[str, Any]:
        """The resolved rendering physics and architecture, stamped into provenance."""
        return {**self.renderer.record, **self._arch}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Render ``x`` ``(B, C_in, H, W)`` to ``(B, C_out, H, W)``."""
        h = self.backbone(self.intro(x))
        for refine in self.refine_stages:
            h = h + refine(h)
        return self.renderer(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
