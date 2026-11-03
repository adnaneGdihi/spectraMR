"""Encode-once / render-anywhere cross-field translator (MICCAI MRIxFields2026).

A source-field-aware anatomy encoder :math:`E(x, b_s) \\to q` and a continuous-field
renderer :math:`R(q, b_t) \\to \\hat{x}`, each modulated by a :class:`FieldFiLMBlock`: the
encoder on the SOURCE field it must factor out of ``q``, the renderer on the TARGET field
(Tesla) and a contrast one-hot. On Task 1 the target is pinned at 7 T, so the source field is
the only field that varies per sample (#197). The field enters as a
*continuous argument* (not a per-field selector), so one parameter set realises
every ordered field pair — the architectural keystone for MICCAI arms B-3.8
(any-to-any, Task 3) and B-1.9 (fixed target → 7T, Task 1).

Magnitude-only (challenge data is magnitude, MNI, ``[0, 1]``): asserts 1→1
channels and raises on complex input (no silent RSS collapse).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.field_film_modulation import FieldFiLMBlock
from spectramr.models.registry import register_model

_N_CONTRAST = 3  # T1w, T2w, T2-FLAIR


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
    )


@register_model(
    name="anatomy_field_renderer",
    training_mode="cross_field_translation",
    supports_contrast_conditioning=True,
)
class AnatomyFieldRenderer(nn.Module):
    """Anatomy encoder + continuous-field FiLM renderer (magnitude-only)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 64,
        width: int = 64,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "AnatomyFieldRenderer is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        self.encoder_in = _conv_block(in_channels, width)
        self.encoder_film = FieldFiLMBlock(num_channels=width, sequence_dim=1)
        self.encoder_out = nn.Sequential(
            _conv_block(width, width),
            nn.Conv2d(width, latent_channels, 1),
        )
        self.decoder_in = _conv_block(latent_channels, width)
        self.film = FieldFiLMBlock(num_channels=width, sequence_dim=_N_CONTRAST)
        self.decoder_out = nn.Sequential(
            _conv_block(width, width),
            nn.Conv2d(width, out_channels, 1),
        )

    @staticmethod
    def _check_real(x: torch.Tensor) -> None:
        if torch.is_complex(x):
            raise ValueError("AnatomyFieldRenderer expects magnitude (real) input; got complex.")

    def encode(self, x: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W]`` at source field ``field_strength`` (Tesla) -> ``[B, latent, H, W]``."""
        self._check_real(x)
        h = self.encoder_in(x)
        b = field_strength.reshape(-1, 1).to(device=h.device, dtype=h.dtype)
        h = self.encoder_film.apply_to(h, b, b.new_zeros(b.shape[0], 1))
        return self.encoder_out(h)

    def render(
        self,
        q: torch.Tensor,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render the latent at the target field (Tesla) + contrast."""
        h = self.decoder_in(q)
        b = field_strength.reshape(-1, 1).float()
        if contrast_id is None:
            seq = torch.zeros(b.shape[0], _N_CONTRAST, device=b.device, dtype=b.dtype)
        else:
            seq = torch.nn.functional.one_hot(contrast_id.long().reshape(-1), _N_CONTRAST).to(
                b.dtype
            )
        h = self.film.apply_to(h, b, seq)
        return self.decoder_out(h)

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        field_strength_target: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Translate ``x`` from its source field ``field_strength`` to ``field_strength_target``."""
        return self.render(self.encode(x, field_strength), field_strength_target, contrast_id)
