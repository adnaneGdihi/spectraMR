"""DiffiT -- Diffusion Vision Transformer (Hatamizadeh et al., ECCV 2024) over k-space.

DiffiT's contribution is Time-dependent Multihead Self-Attention: instead of
scaling a block's output by a time-derived gate the way DiT's adaLN-Zero does,
the timestep enters the **projections themselves** --

    q = x W_q + t W_qt,   k = x W_k + t W_kt,   v = x W_v + t W_vt

-- so what a token attends *to* changes with the denoising step, not just how
strongly the result is mixed back in. The paper's argument is that early steps
need long-range layout and late steps need local detail, and a gate on a fixed
attention map cannot express that.

Paired deliberately with :mod:`dit_backbone`: identical stem, positions, depth
and MLP, so an arm swapping between them isolates the conditioning mechanism
rather than the architecture. The stem and head are the shared k-space
adaptation; see ``diffusion_transformer_stem``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.models.generators.diffusion_transformer_stem import (
    DomainHead,
    DomainStem,
    LearnedPositions,
    TimeConditioning,
    apply_contrast_conditioning,
    build_contrast_projection,
)
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class TimeDependentAttention(nn.Module):
    """TMSA: the timestep is projected into q, k and v additively.

    ``time_qkv`` is zero-initialised so the block starts as ordinary self
    attention and the time pathway has to earn its contribution -- without that
    the random time projection perturbs every attention map before the spatial
    weights have learned anything.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must divide evenly into heads={heads}.")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.time_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.time_qkv.weight)
        nn.init.zeros_(self.time_qkv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x) + self.time_qkv(cond).unsqueeze(1)
        q, k, v = qkv.reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class DiffiTBlock(nn.Module):
    """Pre-norm TMSA + MLP, both plain residuals.

    No adaLN gate here on purpose: the time signal is already inside the
    attention, and adding a gate as well would reintroduce exactly the mechanism
    this backbone exists to be compared against.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TimeDependentAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cond)
        return x + self.mlp(self.norm2(x))


@register_model(
    name="diffit",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain=("kspace", "image"),
    output_domain=("kspace", "image"),
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class DiffiTBackbone(GradCheckpointingMixin, nn.Module, IGenerator):
    """Diffusion Vision Transformer backbone for the k-space cold-diffusion stack."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        image_size: int = 256,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        patch_size: int = 8,
        mlp_ratio: float = 4.0,
        feature_domain: str = "kspace",
        max_timesteps: float | None = None,
        contrast_emb_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem = DomainStem(in_channels, dim, patch_size, feature_domain)
        self.feature_domain = self.stem.feature_domain
        self.positions = LearnedPositions(dim, max(1, int(image_size) // patch_size))
        self.time = TimeConditioning(dim, max_timesteps=max_timesteps)
        # Built eagerly so the projection is part of the optimizer's parameter
        # set from the start. See
        # ``diffusion_transformer_stem.build_contrast_projection`` for why it
        # is sized this way and left at its default init.
        self.contrast_proj = build_contrast_projection(contrast_emb_dim, dim)
        self.blocks = nn.ModuleList(DiffiTBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.head = DomainHead(dim, out_channels, patch_size)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        grid = self.stem.grid(x.shape[-2], x.shape[-1])
        tokens = self.stem(x) + self.positions(grid)
        cond = self.time(timesteps, x.shape[0], x.device, max_timesteps=kwargs.get("max_timesteps"))
        cond = apply_contrast_conditioning(
            cond, kwargs.get("contrast_emb"), self.contrast_proj, owner="DiffiTBackbone"
        )

        ckpt = self._checkpointing_active()
        for block in self.blocks:
            tokens = (
                checkpoint(block, tokens, cond, use_reentrant=False)
                if ckpt
                else block(tokens, cond)
            )
        return self.head(tokens, cond, grid)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @property
    def name(self) -> str:
        return "DiffiT"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, timesteps=kwargs.pop("timesteps", None), **kwargs)
