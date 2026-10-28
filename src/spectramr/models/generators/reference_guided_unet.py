"""Reference-guided U-Net: a separate encoder for the source contrast, fused at every scale.

A cross-contrast arm that concatenates the source contrast onto the backbone
input lets the first convolution mix source and target channels, so no later
layer sees the source as a distinct signal. This backbone gives the source its
own encoder (no timestep conditioning: the source is not degraded) and injects
it into the target encoder at every scale through a learned gate, then lets the
target attend to the source at the bottleneck. The design follows the
multi-scale reference fusion of MINet (Feng et al., MICCAI 2021), MTrans (Feng
et al., IEEE TMI 2022) and DisC-Diff (Mao et al., MICCAI 2023).

It runs on the image-domain tensor ``FourierBridgeNetwork`` produces
(``force_pure_kspace: false``): source and target correspond pixel for pixel in
image space, not bin for bin in k-space. The generator orders the input
``[reference | x_t | smaps]``, and ``reference_channels`` says where the split is.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.generators.diffusion_transformer_stem import (
    TimeConditioning,
    apply_contrast_conditioning,
    build_contrast_projection,
)


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


class _ResBlock(nn.Module):
    """Pre-norm residual block, FiLM-conditioned when ``cond_dim`` is given."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int | None) -> None:
        super().__init__()
        self.norm1, self.conv1 = _norm(c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2, self.conv2 = _norm(c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * c_out) if cond_dim is not None else None
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.film is not None:
            scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
            h = h * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class _GatedFusion(nn.Module):
    """``f_t + sigmoid(g([f_t, f_r])) * v(f_r)``: the target decides, per pixel, how much source to take.

    The gate bias starts negative so the target path is not swamped by the
    source before the gate has learned where the contrasts agree.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.value = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Conv2d(2 * channels, channels, 1)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([target, source], dim=1)))
        return target + gate * self.value(source)


class _CrossAttention(nn.Module):
    """Target queries attend over source keys/values; residual, pre-norm."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"heads={heads} does not divide the bottleneck width {channels}.")
        self.heads = heads
        self.norm_t, self.norm_s = _norm(channels), _norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Conv2d(channels, 2 * channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        b, c, h, w = target.shape

        def tokens(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)

        q = tokens(self.q(self.norm_t(target)))
        k, v = (tokens(t) for t in self.kv(self.norm_s(source)).chunk(2, dim=1))
        attended = F.scaled_dot_product_attention(q, k, v)
        return target + self.out(attended.transpose(-1, -2).reshape(b, c, h, w))


class ReferenceGuidedUNet(nn.Module):
    """Two-encoder U-Net over ``[reference | target]`` image channels.

    Args:
        in_channels: Total input width, reference included.
        out_channels: Output width; the first ``out_channels`` target channels
            (the noisy state, ahead of any S-map channels) are its residual base.
        reference_channels: Leading input channels that hold the source contrast.
        width: First-level feature width; level ``l`` has ``width * 2**l``.
        levels: Encoder levels; spatial size must be divisible by ``2**(levels-1)``.
        heads: Bottleneck cross-attention heads.
        max_timesteps: Diffusion horizon for the timestep embedding.
        contrast_emb_dim: Width of the generator's contrast embedding, if any.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reference_channels: int,
        width: int,
        levels: int = 3,
        heads: int = 4,
        max_timesteps: float | None = None,
        contrast_emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        target_channels = in_channels - reference_channels
        if reference_channels <= 0 or target_channels < out_channels:
            raise ValueError(
                f"reference_guided_unet needs a reference and at least out_channels={out_channels} "
                f"target channels; got in_channels={in_channels}, "
                f"reference_channels={reference_channels}."
            )
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}.")
        self.reference_channels, self.out_channels, self.levels = (
            reference_channels,
            out_channels,
            levels,
        )
        cond_dim = 4 * width
        self.time = TimeConditioning(cond_dim, max_timesteps=max_timesteps)
        self.contrast_proj = build_contrast_projection(contrast_emb_dim, cond_dim)

        widths = [width * 2**level for level in range(levels)]
        self.target_stem = nn.Conv2d(target_channels, width, 3, padding=1)
        self.source_stem = nn.Conv2d(reference_channels, width, 3, padding=1)
        previous = [width, *widths[:-1]]
        self.target_enc = nn.ModuleList(
            _ResBlock(p, c, cond_dim) for p, c in zip(previous, widths, strict=True)
        )
        self.source_enc = nn.ModuleList(
            _ResBlock(p, c, None) for p, c in zip(previous, widths, strict=True)
        )
        self.fusion = nn.ModuleList(_GatedFusion(c) for c in widths)
        self.target_down = nn.ModuleList(nn.Conv2d(c, c, 3, 2, 1) for c in widths[:-1])
        self.source_down = nn.ModuleList(nn.Conv2d(c, c, 3, 2, 1) for c in widths[:-1])

        deepest = widths[-1]
        self.mid_in = _ResBlock(deepest, deepest, cond_dim)
        self.cross = _CrossAttention(deepest, heads)
        self.mid_out = _ResBlock(deepest, deepest, cond_dim)

        self.up = nn.ModuleList(
            nn.Conv2d(widths[level + 1], widths[level], 3, padding=1) for level in range(levels - 1)
        )
        self.dec = nn.ModuleList(
            _ResBlock(2 * widths[level], widths[level], cond_dim) for level in range(levels - 1)
        )
        self.head = nn.Sequential(
            _norm(width), nn.SiLU(), nn.Conv2d(width, out_channels, 3, padding=1)
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Reconstruct the target channels of ``x`` guided by its reference channels."""
        stride = 2 ** (self.levels - 1)
        if x.shape[-2] % stride or x.shape[-1] % stride:
            raise ValueError(
                f"reference_guided_unet with levels={self.levels} needs a spatial size "
                f"divisible by {stride}; got {tuple(x.shape[-2:])}."
            )
        source, target = x[:, : self.reference_channels], x[:, self.reference_channels :]
        cond = self.time(
            timesteps,
            x.shape[0],
            x.device,
            max_timesteps=kwargs.get("max_timesteps"),
            scale=kwargs.get("timestep_scale", "unit_interval"),
        )
        cond = apply_contrast_conditioning(
            cond, kwargs.get("contrast_emb"), self.contrast_proj, owner="ReferenceGuidedUNet"
        )

        h, s = self.target_stem(target), self.source_stem(source)
        skips = []
        for level in range(self.levels):
            h = self.target_enc[level](h, cond)
            s = self.source_enc[level](s)
            h = self.fusion[level](h, s)
            if level < self.levels - 1:
                skips.append(h)
                h, s = self.target_down[level](h), self.source_down[level](s)

        h = self.mid_out(self.cross(self.mid_in(h, cond), s), cond)
        for level in reversed(range(self.levels - 1)):
            h = self.up[level](F.interpolate(h, scale_factor=2.0, mode="nearest"))
            h = self.dec[level](torch.cat([h, skips[level]], dim=1), cond)
        return target[:, : self.out_channels] + self.head(h)
