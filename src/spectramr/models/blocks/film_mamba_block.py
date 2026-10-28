"""Contrast-conditioned bidirectional FiLM-Mamba block.

Wraps :class:`spectramr.models.blocks.mamba_block.MambaBlock` with FiLM
modulation around two Mamba passes (forward and reversed). The
reverse pass gives bidirectional context — anatomy along the SFC
traversal is non-causal — so a unidirectional Mamba would break
the symmetry of voxel-level reconstruction.

Per-token modulation::

    (γ_in, β_in, γ_out, β_out) = MLP(contrast_embedding)
    x_mod  = (1 + γ_in) * x + β_in
    h_fwd  = Mamba_fwd(x_mod)
    h_bwd  = flip(Mamba_bwd(flip(x_mod)))
    y      = (1 + γ_out) * proj([h_fwd, h_bwd]) + β_out

The MLP shape mirrors :class:`FiLMConditioner` in
:mod:`spectramr.infrastructure.physics.fno_epg`. Reuses the
``mamba_ssm`` import + gated-CNN/GRU fallback already provided by
the parent ``MambaBlock`` (so the smoke test passes even without
the optional ``mamba_ssm`` dep).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.blocks.contrast_conditioning import require_contrast_rows
from spectramr.models.blocks.mamba_block import MambaBlock


class FiLMMambaBlock(nn.Module):
    """Bidirectional Mamba block with FiLM contrast conditioning.

    Args:
        d_model: token dimension.
        contrast_dim: dimension of the contrast embedding.
        d_state: state dimension passed to MambaBlock.
        d_conv: depthwise-conv kernel passed to MambaBlock.
        expand: inner expansion factor passed to MambaBlock.
        film_hidden_mult: hidden-layer multiplier inside the FiLM MLP
            (hidden = mult * d_model).
        dropout: residual dropout.
    """

    def __init__(
        self,
        d_model: int,
        contrast_dim: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        film_hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.contrast_dim = contrast_dim

        self.mamba_fwd = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # FiLM MLP: contrast embedding → (γ_in, β_in, γ_out, β_out),
        # 4 * d_model parameters total.
        hidden = max(film_hidden_mult * d_model, contrast_dim)
        self.film = nn.Sequential(
            nn.Linear(contrast_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * d_model),
        )

        self.bidir_proj = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        contrast_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM-modulated bidirectional Mamba.

        Args:
            x: ``(B, L, d_model)`` token sequence (already permuted
                into SFC order).
            contrast_embedding: ``(B, contrast_dim)`` per-volume
                contrast vector.

        Returns:
            ``(B, L, d_model)`` modulated output sequence.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x of shape (B, L, {self.d_model}), got {tuple(x.shape)}")
        if contrast_embedding.ndim != 2 or contrast_embedding.shape[-1] != self.contrast_dim:
            raise ValueError(
                f"Expected contrast_embedding of shape (B, {self.contrast_dim}), "
                f"got {tuple(contrast_embedding.shape)}"
            )
        require_contrast_rows(contrast_embedding, x.shape[0], owner="FiLMMambaBlock")

        film_params = self.film(contrast_embedding)  # (B, 4*d_model)
        gamma_in, beta_in, gamma_out, beta_out = film_params.chunk(4, dim=-1)
        # Broadcast (B, d_model) → (B, 1, d_model) so it modulates per-token.
        gamma_in = gamma_in.unsqueeze(1)
        beta_in = beta_in.unsqueeze(1)
        gamma_out = gamma_out.unsqueeze(1)
        beta_out = beta_out.unsqueeze(1)

        x_mod = (1.0 + gamma_in) * x + beta_in
        h_fwd = self.mamba_fwd(x_mod)
        h_bwd = self.mamba_bwd(torch.flip(x_mod, dims=(1,))).flip(dims=(1,))
        h = self.bidir_proj(torch.cat([h_fwd, h_bwd], dim=-1))
        out = (1.0 + gamma_out) * h + beta_out
        return self.dropout(out)


__all__ = ["FiLMMambaBlock"]
