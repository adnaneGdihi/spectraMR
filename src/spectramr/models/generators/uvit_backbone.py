"""U-ViT -- "All are Worth Words" (Bao et al., CVPR 2023) over k-space.

A ViT with the U-Net's long skips kept and its convolutional hierarchy dropped:
the first half's block outputs are concatenated onto the second half's inputs and
projected back down, so shallow high-frequency detail reaches the output without
being resampled. Time arrives as an extra **token** rather than through adaLN,
which is the paper's distinguishing choice against :mod:`dit_backbone`.

The depth must be odd -- ``depth // 2`` encoder blocks, one middle block, and
``depth // 2`` decoder blocks each consuming one skip. The stem and head are the
shared k-space adaptation; see ``diffusion_transformer_stem``.

One deviation from the published model: the head is conditioned on the time
vector as well as carrying the time token, because it is reused from the shared
module where DiT needs that modulation. It reduces to the paper's
``LayerNorm -> Linear`` when the zero-initialised modulation stays at zero.
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
)
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class UViTBlock(nn.Module):
    """Pre-norm transformer block with an optional long-skip merge.

    ``skip_proj`` is present only on decoder blocks. It is a plain ``Linear`` over
    the concatenated pair rather than an addition, which is what lets the block
    learn how much of the shallow signal to keep per channel.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, *, takes_skip: bool):
        super().__init__()
        self.skip_proj = nn.Linear(dim * 2, dim) if takes_skip else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TokenAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.skip_proj is not None:
            if skip is None:
                raise ValueError(
                    "UViTBlock was built as a decoder block but received no skip "
                    "tensor; the encoder/decoder split and the skip stack disagree."
                )
            x = self.skip_proj(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@register_model(
    name="uvit",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain=("kspace", "image"),
    output_domain=("kspace", "image"),
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class UViTBackbone(GradCheckpointingMixin, nn.Module, IGenerator):
    """U-ViT backbone for the k-space cold-diffusion stack."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        image_size: int = 256,
        dim: int = 384,
        depth: int = 13,
        heads: int = 6,
        patch_size: int = 8,
        mlp_ratio: float = 4.0,
        feature_domain: str = "kspace",
        max_timesteps: float | None = None,
        contrast_emb_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if depth < 3 or depth % 2 == 0:
            raise ValueError(
                f"U-ViT needs an ODD depth >= 3 so the encoder and decoder halves "
                f"pair one-to-one around a single middle block; got depth={depth}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half = depth // 2
        self.stem = DomainStem(in_channels, dim, patch_size, feature_domain)
        self.feature_domain = self.stem.feature_domain
        self.positions = LearnedPositions(dim, max(1, int(image_size) // patch_size))
        self.time = TimeConditioning(dim, max_timesteps=max_timesteps)
        # Built eagerly so the projection is part of the optimizer's parameter
        # set from the start. See
        # ``diffusion_transformer_stem.build_contrast_projection`` for why it
        # is sized this way and left at its default init.
        self.contrast_proj = build_contrast_projection(contrast_emb_dim, dim)
        self.encoder = nn.ModuleList(
            UViTBlock(dim, heads, mlp_ratio, takes_skip=False) for _ in range(self.half)
        )
        self.middle = UViTBlock(dim, heads, mlp_ratio, takes_skip=False)
        self.decoder = nn.ModuleList(
            UViTBlock(dim, heads, mlp_ratio, takes_skip=True) for _ in range(self.half)
        )
        self.head = DomainHead(dim, out_channels, patch_size)
        self.refine = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _run(self, block: UViTBlock, x: torch.Tensor, skip: torch.Tensor | None, ckpt: bool):
        if not ckpt:
            return block(x, skip)
        if skip is None:
            return checkpoint(block, x, use_reentrant=False)
        return checkpoint(block, x, skip, use_reentrant=False)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        grid = self.stem.grid(x.shape[-2], x.shape[-1])
        cond = self.time(timesteps, x.shape[0], x.device, max_timesteps=kwargs.get("max_timesteps"))
        cond = apply_contrast_conditioning(
            cond, kwargs.get("contrast_emb"), self.contrast_proj, owner="UViTBackbone"
        )

        # Time rides as token 0 and is dropped before unpatchify, so the head
        # only ever sees the N patch tokens the grid accounts for.
        tokens = torch.cat([cond.unsqueeze(1), self.stem(x) + self.positions(grid)], dim=1)

        ckpt = self._checkpointing_active()
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            tokens = self._run(block, tokens, None, ckpt)
            skips.append(tokens)
        tokens = self._run(self.middle, tokens, None, ckpt)
        for block in self.decoder:
            tokens = self._run(block, tokens, skips.pop(), ckpt)

        field = self.head(tokens[:, 1:], cond, grid)
        return field + self.refine(field)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @property
    def name(self) -> str:
        return "UViT"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator."""
        return self.forward(z, timesteps=kwargs.pop("timesteps", None), **kwargs)
