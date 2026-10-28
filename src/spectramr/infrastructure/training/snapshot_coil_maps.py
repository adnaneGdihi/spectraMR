"""Per-coil preview of image-domain sensitivity maps for the debug snapshot.

A coil-RSS preview of RSS-normalised maps is identically 1, so the generic
previewer cannot show whether the maps are right. This grid shows what can:
each coil's magnitude (one shared scale, so relative coil weights stay visible)
and phase, then the coil RSS, for the first sample of the batch. Cold
diffusion's conditioning half (``smaps``, ``model_input__smaps``) is ``fft2c``
of such maps, so it is brought back to image space and drawn the same way.
"""

from __future__ import annotations

import math

import torch

#: Snapshot keys (and ``channel_segments`` render names) that hold coil maps, in
#: whichever domain the snapshot declares for them.
COIL_MAP_KEYS: frozenset[str] = frozenset({"coil_maps", "smaps", "model_input__smaps"})


def _as_complex_coils(maps: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` complex from complex or real/imag-interleaved ``[B, 2C, H, W]``."""
    if torch.is_complex(maps):
        return maps
    channels = maps.shape[1]
    if channels % 2:
        raise ValueError(
            f"coil maps must be complex or real/imag interleaved; got {channels} real channels."
        )
    paired = maps.reshape(maps.shape[0], channels // 2, 2, *maps.shape[-2:])
    return torch.complex(paired[:, :, 0].float(), paired[:, :, 1].float())


def render_coil_map_grid(
    maps: torch.Tensor, *, in_kspace: bool = False, log_scaled: bool = False
) -> torch.Tensor:
    """``[2C + 1, 1, H, W]`` in ``[0, 1]``: C magnitudes, C phases, then the coil RSS.

    Laid out with ``nrow=C`` this is one row of magnitudes, one row of phases
    and the RSS on its own row. Magnitudes share the largest coil magnitude as
    their scale; phase maps ``[-pi, pi]`` onto ``[0, 1]``; the RSS is divided
    by its own maximum. ``in_kspace`` maps are ``ifft2c``'d first (after
    undoing ``log1p`` when ``log_scaled``).
    """
    if maps.dim() != 4:
        raise ValueError(f"coil maps must be [B, C, H, W]; got shape {tuple(maps.shape)}.")
    coils = _as_complex_coils(maps.detach())[:1].to(torch.complex64).cpu()
    if in_kspace:
        from spectramr.infrastructure.physics.fft_ops import ifft2c
        from spectramr.infrastructure.training.utils.kspace_view import decompress_for_view

        coils = ifft2c(decompress_for_view(coils, log_scaled=log_scaled, channel_dim=1))
    coils = coils[0]
    magnitude = coils.abs()
    magnitude = magnitude / magnitude.amax().clamp_min(1e-12)
    phase = (coils.angle() + math.pi) / (2 * math.pi)
    rss = torch.sqrt((coils.abs() ** 2).sum(dim=0, keepdim=True))
    rss = rss / rss.amax().clamp_min(1e-12)
    return torch.cat([magnitude, phase, rss]).unsqueeze(1).clamp(0.0, 1.0)


__all__ = ["COIL_MAP_KEYS", "render_coil_map_grid"]
