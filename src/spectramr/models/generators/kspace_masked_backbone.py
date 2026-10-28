"""Masked-token k-space backbone: a bidirectional transformer that fills missing lines.

Lines the current mask removed are replaced by a learned ``[MASK]`` embedding
(BERT/MAE style) before attention, then every line is predicted in one pass. It is
a one-step restoration net inside the unchanged cold-diffusion loop.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.generators.kspace_line_tokens import (
    LineHead,
    LinePositions,
    LineStem,
    LineTrunk,
    expand_batch,
    line_support,
    validate_feature_domain_kspace,
    validate_line_axis,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(
    name="kspace_masked",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain=("kspace",),
    output_domain=("kspace",),
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class KSpaceMaskedBackbone(GradCheckpointingMixin, nn.Module, IGenerator):
    """Bidirectional line transformer with a learned mask token."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        image_size: int = 256,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        line_axis: int = -1,
        feature_domain: str = "kspace",
        max_timesteps: float | None = None,
        contrast_emb_dim: int | None = None,
    ):
        super().__init__()
        validate_feature_domain_kspace(feature_domain, "KSpaceMaskedBackbone")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.line_axis = validate_line_axis(line_axis)
        self.stem = LineStem(in_channels, image_size, dim, self.line_axis)
        self.positions = LinePositions(dim, image_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.trunk = LineTrunk(
            dim, depth, heads, mlp_ratio, max_timesteps, contrast_emb_dim, "KSpaceMaskedBackbone"
        )
        self.head = LineHead(dim, out_channels, image_size, self.line_axis)
        self._mask_checked: set[bool] = set()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        owner = "KSpaceMaskedBackbone"
        support = line_support(
            mask,
            self.line_axis,
            owner,
            check=self.training not in self._mask_checked,
        )
        self._mask_checked.add(self.training)
        b = x.shape[0]
        tok = self.stem(x)
        if support.shape[1] != tok.shape[1]:
            raise ValueError(
                f"{owner}: the mask has {support.shape[1]} lines but the field has {tok.shape[1]}."
            )
        known = expand_batch(support, b, owner).unsqueeze(-1)
        tok = torch.where(known, tok, self.mask_token.to(tok.dtype)) + self.positions(tok.shape[1])
        cond = self.trunk.condition(timesteps, b, x.device, kwargs)
        return self.head(self.trunk(tok, cond, self._checkpointing_active()), cond)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @property
    def name(self) -> str:
        return "KSpaceMasked"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, timesteps=kwargs.pop("timesteps", None), **kwargs)
