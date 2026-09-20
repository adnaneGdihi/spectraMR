"""DiT -- Diffusion Transformer (Peebles & Xie, ICCV 2023) over k-space.

The published trunk unchanged: patch tokens, a plain pre-norm transformer, and
adaLN-Zero conditioning, whose zero-initialised gates make every block start as
the identity so the network begins training as a clean residual path.

What is adapted is the stem: MRI is k-space-native, so the tokens come from
:class:`DomainStem`, which embeds both the k-space and the image view and sums
them. Without that a patch is a frequency band under ``force_pure_kspace: true``
and a picture under ``false``, with nothing in the architecture registering the
difference. See ``diffusion_transformer_stem`` for why that is the one owner.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.models.generators.diffusion_transformer_stem import (
    DomainHead,
    DomainStem,
    LearnedPositions,
    TimeConditioning,
    TokenAttention,
    apply_contrast_conditioning,
    build_contrast_projection,
    modulate,
)
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class DiTBlock(nn.Module):
    """Pre-norm attention + MLP, both gated by adaLN-Zero.

    The six modulation parameters come from one zero-initialised projection of
    the conditioning vector, so ``gate_*`` is 0 at init and the block is exactly
    the identity -- the property the paper credits for its training stability.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = TokenAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


@register_model(
    name="dit",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain=("kspace", "image"),
    output_domain=("kspace", "image"),
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class DiTBackbone(GradCheckpointingMixin, nn.Module, IGenerator):
    """Diffusion Transformer backbone for the k-space cold-diffusion stack."""

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
        reference_grid = max(1, int(image_size) // patch_size)
        self.positions = LearnedPositions(dim, reference_grid)
        self.time = TimeConditioning(dim, max_timesteps=max_timesteps)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.head = DomainHead(dim, out_channels, patch_size)
        # Built eagerly (not lazily in forward) so the projection is part of
        # the optimizer's parameter set and DDP's bucketing from the start.
        # See ``diffusion_transformer_stem.build_contrast_projection`` for why
        # it is sized this way and left at its default init.
        self.contrast_proj = build_contrast_projection(contrast_emb_dim, dim)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        grid = self.stem.grid(x.shape[-2], x.shape[-1])
        tokens = self.stem(x) + self.positions(grid)

        cond = self.time(timesteps, x.shape[0], x.device, max_timesteps=kwargs.get("max_timesteps"))
        cond = apply_contrast_conditioning(
            cond, kwargs.get("contrast_emb"), self.contrast_proj, owner="DiTBackbone"
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
        return "DiT"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, timesteps=kwargs.pop("timesteps", None), **kwargs)
