"""HAT -- Hybrid Attention Transformer (Chen et al., CVPR 2023) over k-space.

HAT's claim is that window self-attention alone activates too few input pixels,
so each block runs window attention **and** a convolutional channel-attention
branch over the same features and adds them. Unlike DiT and U-ViT this trunk
never patchifies: it works at full resolution, which suits k-space, where
downsampling by a patch stride throws away the outer frequencies that carry edge
detail.

Published as a super-resolution network with no time input. The diffusion
adaptation here is FiLM on the feature map from the shared timestep embedding --
the lightest conditioning that still reaches every block, and deliberately not
adaLN-Zero, so this arm and :mod:`dit_backbone` differ in trunk rather than in
two things at once.

Windowing comes from ``models.blocks.swin_windows``, the documented single owner;
seven divergent copies of ``window_partition`` exist in this tree and picking one
of the others would silently change the geometry (non-negotiable 17).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.models.blocks.attention import ChannelAttention
from spectramr.models.blocks.attention_domains import validate_feature_domain
from spectramr.models.blocks.swin_windows import window_partition, window_reverse
from spectramr.models.generators.diffusion_transformer_stem import (
    TimeConditioning,
    TokenAttention,
    apply_contrast_conditioning,
    both_domain_views,
    build_contrast_projection,
)
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class HybridAttentionBlock(nn.Module):
    """Window self-attention summed with a convolutional channel-attention branch.

    ``cab_weight`` is the paper's alpha: the conv branch is a small correction to
    the attention branch, not an equal partner, and at 1.0 it drowns it.
    """

    def __init__(self, dim: int, heads: int, window_size: int, cab_weight: float):
        super().__init__()
        self.window_size = window_size
        self.cab_weight = cab_weight
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TokenAttention(dim, heads)
        self.cab = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            ChannelAttention(dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, C, H, W]``."""
        b, c, h, w = x.shape
        conv_branch = self.cab(x)

        nhwc = x.permute(0, 2, 3, 1)
        windows = window_partition(self.norm1(nhwc), self.window_size)
        tokens = windows.view(-1, self.window_size * self.window_size, c)
        attended = self.attn(tokens).view(-1, self.window_size, self.window_size, c)
        attn_branch = window_reverse(attended, self.window_size, h, w).permute(0, 3, 1, 2)

        x = x + attn_branch + self.cab_weight * conv_branch
        flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        flat = flat + self.mlp(self.norm2(flat))
        return flat.view(b, h, w, c).permute(0, 3, 1, 2)


@register_model(
    name="hat",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain=("kspace", "image"),
    output_domain=("kspace", "image"),
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class HATBackbone(GradCheckpointingMixin, nn.Module, IGenerator):
    """Hybrid Attention Transformer backbone for the k-space cold-diffusion stack."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        dim: int = 96,
        depth: int = 6,
        heads: int = 6,
        window_size: int = 8,
        cab_weight: float = 0.05,
        feature_domain: str = "kspace",
        max_timesteps: float | None = None,
        contrast_emb_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"HAT consumes an interleaved real/imag field; in_channels must be "
                f"even, got {in_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = int(window_size)
        self.feature_domain = validate_feature_domain(feature_domain)
        # One shallow conv per domain view, as in DomainStem: the two views carry
        # different statistics, so a shared filter bank would fit one and waste
        # the other. Stride 1 -- HAT's whole argument is full-resolution context.
        self.shallow_kspace = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.shallow_image = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.time = TimeConditioning(dim, max_timesteps=max_timesteps)
        # Built eagerly so the projection is part of the optimizer's parameter
        # set from the start. See
        # ``diffusion_transformer_stem.build_contrast_projection`` for why it
        # is sized this way and left at its default init.
        self.contrast_proj = build_contrast_projection(contrast_emb_dim, dim)
        self.film = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.blocks = nn.ModuleList(
            HybridAttentionBlock(dim, heads, self.window_size, cab_weight) for _ in range(depth)
        )
        # Not zero-initialised, for the reason DomainHead states: a forward
        # that emits zero is indistinguishable from one that ignores its
        # measurement, and the Tier-2 probe rejects both.
        self.out_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def _check_window(self, height: int, width: int) -> None:
        if height % self.window_size or width % self.window_size:
            raise ValueError(
                f"window_size={self.window_size} does not tile a {height}x{width} "
                f"field. Padding to fit would wrap k-space across the window edge, "
                f"so set model_kwargs.window_size to a divisor of both."
            )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        self._check_window(x.shape[-2], x.shape[-1])
        k_view, i_view = both_domain_views(x, self.feature_domain)
        h = self.shallow_kspace(k_view) + self.shallow_image(i_view)

        cond = self.time(timesteps, x.shape[0], x.device, max_timesteps=kwargs.get("max_timesteps"))
        cond = apply_contrast_conditioning(
            cond, kwargs.get("contrast_emb"), self.contrast_proj, owner="HATBackbone"
        )
        shift, scale = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

        shallow = h
        ckpt = self._checkpointing_active()
        for block in self.blocks:
            h = checkpoint(block, h, use_reentrant=False) if ckpt else block(h)
        return self.out_conv(h + shallow)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @property
    def name(self) -> str:
        return "HAT"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, timesteps=kwargs.pop("timesteps", None), **kwargs)
