"""Autoregressive k-space backbone: a prefix-causal transformer over centre-out k-space lines.

A one-step restoration net for the cold-diffusion loop (no token-by-token decoding).
Acquired lines form a bidirectional prefix, like an LLM's prompt; each missing line is
a learned mask token that attends to the whole prefix and to the missing lines nearer
the centre, so the gaps are filled coarse to fine. A strictly shifted causal model was
rejected: it would predict every acquired line blind to its own measurement.
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
    centre_out_order,
    expand_batch,
    line_support,
    prefix_causal_mask,
    validate_feature_domain_kspace,
    validate_line_axis,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(
    name="kspace_ar",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain=("kspace",),
    output_domain=("kspace",),
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class KSpaceARBackbone(GradCheckpointingMixin, nn.Module, IGenerator):
    """Prefix-causal line transformer; see the module docstring."""

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
        validate_feature_domain_kspace(feature_domain, "KSpaceARBackbone")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.line_axis = validate_line_axis(line_axis)
        self.stem = LineStem(in_channels, image_size, dim, self.line_axis)
        self.positions = LinePositions(dim, image_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.trunk = LineTrunk(
            dim, depth, heads, mlp_ratio, max_timesteps, contrast_emb_dim, "KSpaceARBackbone"
        )
        self.head = LineHead(dim, out_channels, image_size, self.line_axis)
        order = centre_out_order(image_size)
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("inverse", torch.argsort(order), persistent=False)
        self._mask_checked: set[bool] = set()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        owner = "KSpaceARBackbone"
        support = line_support(
            mask, self.line_axis, owner, check=self.training not in self._mask_checked
        )
        self._mask_checked.add(self.training)
        b, n = x.shape[0], self.order.shape[0]
        tok = self.stem(x)
        if tok.shape[1] != n or support.shape[1] != n:
            raise ValueError(
                f"{owner} was built for {n} lines (image_size) but got {tok.shape[1]} "
                f"field lines and a mask with {support.shape[1]}; it needs a square field."
            )
        known = expand_batch(support, b, owner)[:, self.order]  # centre-out
        tok = torch.where(known.unsqueeze(-1), tok[:, self.order], self.mask_token.to(tok.dtype))
        tok = tok + self.positions(n)[:, self.order]
        cond = self.trunk.condition(timesteps, b, x.device, kwargs)
        out = self.trunk(tok, cond, self._checkpointing_active(), prefix_causal_mask(known))
        return self.head(out[:, self.inverse], cond)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @property
    def name(self) -> str:
        return "KSpaceAR"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, timesteps=kwargs.pop("timesteps", None), **kwargs)
