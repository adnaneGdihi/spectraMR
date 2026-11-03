"""Prompt-aware Hermitian FNO: a shared backbone with acquisition-conditioned LoRA.

The backbone is ``hermitian_fno`` (lift, ``depth`` Hermitian spectral blocks, projection).
Each block's pointwise residual conv carries a low-rank weight update whose per-rank
coefficients are a function of the acquisition prompt::

    W(p) = W_0 + B diag(alpha(e(p))) A,    e = PhysicsPromptEmbedding,  rank r << width

``p`` is the per-sample physical acquisition vector (TR, TE, flip angle, B0) and ``e`` its
log-normalised MLP embedding. ``B`` is zero-initialised, so at step 0 the network equals the
unconditioned backbone, and the update stays inside an ``r``-dimensional family around
``W_0`` (one shared model, not a per-protocol ensemble). The adapters are
:class:`FieldConditionedLoRAConv2d` with a 1x1 kernel; a pointwise conv scaled per rank
component commutes with a width flip, so the blocks keep the backbone's Hermitian
equivariance.

The embedding lives on the model, so it trains on ``opt_g`` from the first step and is
saved and restored with the generator checkpoint.

References:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.models.blocks.field_conditioned_lora import FieldConditionedLoRAConv2d
from spectramr.models.blocks.hermitian_equivariant import HermitianSpectralConv2d
from spectramr.models.blocks.physics_prompt import PhysicsPromptEmbedding
from spectramr.models.registry import register_model

#: Forward keyword that carries the ``[B, prompt_features]`` acquisition vector.
PROMPT_KWARG = "acquisition_prompt"


@register_model(
    "prompt_lora_hermitian_fno",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain=("image", "complex_image"),
    output_domain=("image", "complex_image"),
    strict_kwargs=True,
)
class PromptLoRAHermitianFNO(nn.Module):
    """Hermitian FNO whose pointwise convs take an acquisition-prompt LoRA update.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        width: Hidden width.
        modes: Spectral modes per axis.
        depth: Number of Hermitian FNO blocks.
        lora_rank: Rank of each block's prompt-conditioned update. Must be >= 1; the
            unconditioned control is ``hermitian_fno``.
        prompt_features: Length of the acquisition vector (4 for TR, TE, FA, B0).
        prompt_embed_dim: Width of the prompt embedding that drives the adapters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
        depth: int = 4,
        lora_rank: int = 4,
        prompt_features: int = 4,
        prompt_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        if lora_rank < 1:
            raise ValueError(
                f"lora_rank must be >= 1, got {lora_rank}; a rank-0 model ignores the prompt "
                "and is the hermitian_fno control."
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.prompt_features = int(prompt_features)
        self.lora_rank = int(lora_rank)
        self.prompt_embedding = PhysicsPromptEmbedding(
            in_features=prompt_features, embed_dim=prompt_embed_dim
        )
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [HermitianSpectralConv2d(width, width, modes, modes) for _ in range(depth)]
        )
        self.pointwise = nn.ModuleList(
            [
                FieldConditionedLoRAConv2d(
                    width, width, kernel_size=1, rank=lora_rank, field_dim=prompt_embed_dim
                )
                for _ in range(depth)
            ]
        )
        self.activation = nn.GELU()
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, *, acquisition_prompt: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x`` under the per-sample acquisition prompt.

        Args:
            x: ``[B, in_channels, H, W]`` real image tensor.
            acquisition_prompt: ``[B, prompt_features]`` physical acquisition vector.

        Returns:
            ``[B, out_channels, H, W]``.
        """
        if acquisition_prompt.dim() != 2 or acquisition_prompt.shape[0] != x.shape[0]:
            raise ValueError(
                f"acquisition_prompt must be [B={x.shape[0]}, {self.prompt_features}]; "
                f"got {tuple(acquisition_prompt.shape)}."
            )
        embedding = self.prompt_embedding(acquisition_prompt)
        h = self.lift(x)
        for spectral, pointwise in zip(self.spectral, self.pointwise, strict=True):
            h = self.activation(spectral(h) + pointwise(h, field=embedding))
        return self.project(h)


__all__ = ["PROMPT_KWARG", "PromptLoRAHermitianFNO"]
