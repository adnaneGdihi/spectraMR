"""X-Field-FM: a Swin-V2 U-Net whose every stage is FiLM-conditioned on B0 (and sequence).

One :class:`~spectramr.models.blocks.field_film_modulation.FieldFiLMBlock` per encoder and
decoder stage maps ``(field_strength, sequence_features)`` to per-channel ``(gamma, beta)``
applied to that stage's output (Perez et al. 2018), identity at initialisation. With
``field_conditioning: false`` the model is the plain backbone, which a single-field arm uses.

``tissue_heads`` adds T1 (ms, softplus) and 3-class tissue-probability heads on the last
decoder stage's features; the contrast-consistency prior reads them from ``last_aux_maps``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.field_film_modulation import FieldFiLMBlock
from spectramr.models.generators.swin_v2_unet import SwinV2UNet
from spectramr.models.registry import register_model

__all__ = ["CrossFieldFM"]


@register_model(
    name="cross_field_fm",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    strict_kwargs=True,
)
class CrossFieldFM(nn.Module):
    """Swin-V2 U-Net with per-stage field FiLM and optional T1 / tissue heads."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        base_channels: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 8,
        field_conditioning: bool = True,
        field_conditioning_dim: int = 32,
        sequence_conditioning_dim: int = 0,
        tissue_heads: bool = False,
        t1_scale_ms: float = 1000.0,
    ) -> None:
        super().__init__()
        self.backbone = SwinV2UNet(
            in_channels, out_channels, spatial_dims, base_channels, depths, num_heads, window_size
        )
        self.sequence_dim = int(sequence_conditioning_dim)
        self.field_conditioning = bool(field_conditioning)
        self._cond: tuple[torch.Tensor, torch.Tensor] | None = None
        self.films = nn.ModuleDict()
        if self.field_conditioning:
            stages = [("encoders", i, m) for i, m in enumerate(self.backbone.encoders)]
            stages += [("decoders", i, m) for i, m in enumerate(self.backbone.decoders)]
            for kind, i, module in stages:
                film = FieldFiLMBlock(
                    base_channels * 2**i, self.sequence_dim, field_conditioning_dim
                )
                self.films[f"{kind}_{i}"] = film
                module.register_forward_hook(self._film_hook(film))
        self.t1_head = self.tissue_head = None
        self.last_aux_maps: dict[str, torch.Tensor] | None = None
        if tissue_heads:
            self.t1_scale_ms = float(t1_scale_ms)
            self.t1_head = nn.Conv2d(base_channels, 1, 1)
            self.tissue_head = nn.Conv2d(base_channels, 3, 1)
            self.backbone.decoders[0].register_forward_hook(self._capture_features)
        self._features: torch.Tensor | None = None

    def _film_hook(self, film: FieldFiLMBlock):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            if self._cond is None:
                raise RuntimeError("cross_field_fm: a stage ran without field conditioning")
            gamma, beta = film(*self._cond)
            return gamma[:, :, None, None] * output + beta[:, :, None, None]

        return hook

    def _capture_features(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self._features = output

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor | None = None,
        sequence_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.field_conditioning:
            if field_strength is None:
                raise ValueError("cross_field_fm conditions on B0: pass field_strength [B]")
            b0 = field_strength.reshape(-1).to(x.dtype)
            if sequence_features is None:
                if self.sequence_dim:
                    raise ValueError(
                        f"sequence_conditioning_dim={self.sequence_dim} needs sequence_features"
                    )
                sequence_features = x.new_zeros(b0.shape[0], 0)
            self._cond = (b0, sequence_features.to(x.dtype))
        try:
            image = self.backbone(x)
        finally:
            self._cond = None
        if self.t1_head is not None:
            f = self._features
            self.last_aux_maps = {
                "t1_ms": nn.functional.softplus(self.t1_head(f)) * self.t1_scale_ms,
                "tissue_probabilities": self.tissue_head(f).softmax(dim=1),
            }
        return image
