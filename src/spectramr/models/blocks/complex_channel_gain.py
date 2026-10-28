"""Phase-preserving timestep gain for complex k-space feature maps.

The k-space U-Net blocks condition on the timestep multiplicatively, because an
additive per-channel term is constant over every k-space bin, which is a Dirac
impulse at the image origin -- the same reason their convolutions carry no bias.
They used to project the embedding to one gain per REAL channel. On the
interleaved ``[R1, I1, R2, I2, ...]`` layout that scales the real and imaginary
parts of one complex coefficient by different factors, which rotates its phase
by an amount that depends on ``t``: the conditioning itself corrupted the phase
the complex convolutions exist to preserve.

:class:`ComplexChannelGain` emits one real gain per COMPLEX channel and applies
it to both parts, so ``z -> (1 + g) z`` commutes with any global phase rotation
``z -> e^{i theta} z``. There is deliberately no shift: a complex shift is the
additive impulse above, and a real-only shift breaks the same equivariance the
gain restores.
"""

from __future__ import annotations

import torch
from torch import nn


class ComplexChannelGain(nn.Module):
    """Scale each complex channel by ``1 + g_c(emb)`` with one real ``g_c``.

    Args:
        emb_dim: Width of the conditioning embedding.
        channels: Channel count of the feature map as it is stored: real
            interleaved channels when ``interleaved`` (so twice the complex
            count), complex channels for a complex-dtype map, or plain real
            channels when ``interleaved=False`` on a real map.
        interleaved: The map holds complex coefficients as adjacent
            ``(Re, Im)`` channel pairs. ``False`` gives one gain per stored
            channel, which is the right reading for a complex-dtype map and for
            a real map whose channels carry no complex pairing.

    Raises:
        ValueError: ``interleaved`` with an odd ``channels``.
    """

    def __init__(self, emb_dim: int, channels: int, *, interleaved: bool = True) -> None:
        super().__init__()
        if interleaved and channels % 2:
            raise ValueError(
                f"ComplexChannelGain: an interleaved (Re, Im) map needs an even channel "
                f"count, got {channels}."
            )
        self.interleaved = bool(interleaved)
        self.channels = int(channels)
        self.gains = self.channels // 2 if self.interleaved else self.channels
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, self.gains))
        # Zero-initialised so every block starts at the exact identity; this
        # replaces the former ``emb / 10`` damping.
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def gain(self, emb: torch.Tensor) -> torch.Tensor:
        """The ``[B, channels]`` per-stored-channel factor ``1 + g``."""
        g = 1.0 + self.proj(emb)
        return g.repeat_interleave(2, dim=1) if self.interleaved else g

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        """Apply the gain; ``emb=None`` is the unconditioned identity.

        Args:
            x: ``[B, channels, *spatial]`` feature map, real or complex.
            emb: ``[B, emb_dim]`` conditioning embedding, or ``None``.

        Returns:
            ``x`` scaled per complex channel, same shape and dtype.
        """
        if emb is None:
            return x
        if x.shape[1] != self.channels:
            raise ValueError(
                f"ComplexChannelGain was built for {self.channels} channels and received "
                f"{x.shape[1]}."
            )
        g = self.gain(emb)
        g = g.reshape(*g.shape, *([1] * (x.dim() - 2)))
        return x * g.to(dtype=x.real.dtype)


__all__ = ["ComplexChannelGain"]
