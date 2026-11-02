"""Progressive-growing GAN generator.

Implements the generator of Karras *et al.* [Karras et al., ICLR 2018],
"Progressive Growing of GANs for Improved Quality, Stability, and
Variation". Training proceeds in phases; at phase ``s`` the generator
emits images at resolution ``4 * 2**s``. A fade-in interpolant blends a
newly-added phase into the previous one as ``alpha`` is annealed 0 -> 1:

.. math::

    G_t(z) = (1 - \\alpha_t)\\,\\mathrm{Up}(G_{s-1}(z)) + \\alpha_t\\,G_s(z).

Pixelwise feature normalisation (:class:`PixelNorm`) stabilises the
generator activations. The phase index ``current_phase`` and the fade-in
coefficient ``alpha`` are mutated by
:class:`~spectramr.infrastructure.training.strategies.progressive_gan_strategy.ProgressiveGANStrategy`.

The model is registered under the default ``gan`` training mode so it
runs through :class:`GANTrainingStrategy` without any strategy-factory
edit; the progressive strategy is an opt-in refinement that schedules the
phase growth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.progressive import FadeIn, PixelNorm
from spectramr.models.registry import register_model

#: Side of the pooled grid the conditioning image is reduced to before the latent projection.
_COND_GRID = 8


class _ProGANPhaseBlock(nn.Module):
    """One resolution phase: 2x upsample + two 3x3 convs with pixel norm.

    Each block also owns a ``to_rgb`` 1x1 projection mapping its feature
    channels to the output image channels at this resolution.
    """

    def __init__(self, in_ch: int, out_ch: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pixel_norm = PixelNorm()
        self.to_rgb = nn.Conv2d(out_ch, out_channels, kernel_size=1)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor, *, upsample: bool = True) -> torch.Tensor:
        """Run the block, optionally upsampling the input by 2x first."""
        if upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.pixel_norm(self.act(self.conv1(x)))
        x = self.pixel_norm(self.act(self.conv2(x)))
        return x


@register_model(
    name="progressive_gan",
    training_mode="gan",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
)
class ProgressiveGAN(nn.Module):
    """Progressive-growing GAN generator.

    Args:
        in_channels: Channels of the conditioning input image. When the
            forward input is a flat latent vector this is ignored.
        out_channels: Channels of the generated image.
        z_dim: Latent dimension projected into the 4x4 base feature map.
        max_resolution: Final output resolution (power of two, >= 4).
        base_channels: Feature channels of the 4x4 base block; halved
            (floored at ``min_channels``) at each subsequent phase.
        min_channels: Lower bound on per-phase feature channels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        z_dim: int = 512,
        max_resolution: int = 256,
        base_channels: int = 512,
        min_channels: int = 16,
    ) -> None:
        super().__init__()
        if max_resolution < 4 or (max_resolution & (max_resolution - 1)) != 0:
            raise ValueError(f"max_resolution must be a power of two >= 4, got {max_resolution}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_dim = z_dim
        self.max_resolution = max_resolution
        self.base_channels = base_channels

        # Phase 0 produces 4x4; each later phase doubles the resolution.
        self.num_phases = int(math.log2(max_resolution)) - 1  # 4 -> max_res

        # Project a latent vector to the 4x4 base feature map.
        self.project = nn.Linear(z_dim, base_channels * 4 * 4)
        self.base_pixel_norm = PixelNorm()
        self.base_act = nn.LeakyReLU(0.2, inplace=True)
        # to_rgb for the 4x4 base block.
        self.base_to_rgb = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        phases: list[nn.Module] = []
        prev_ch = base_channels
        for s in range(1, self.num_phases):
            out_ch = max(base_channels // (2**s), min_channels)
            phases.append(_ProGANPhaseBlock(prev_ch, out_ch, out_channels))
            prev_ch = out_ch
        self.phases = nn.ModuleList(phases)

        self.fade_in = FadeIn()

        # Image conditioning keeps a coarse spatial layout (a mean-pooled input made the
        # output invariant to pixel permutations, #2384) and is built here so an
        # optimizer constructed before the first forward owns its parameters.
        self.cond_pool = nn.AdaptiveAvgPool2d(_COND_GRID)
        self.cond_proj = nn.Linear(in_channels * _COND_GRID * _COND_GRID, z_dim)

        # Mutated by ProgressiveGANStrategy (0 == 4x4 base only).
        self.current_phase: int = self.num_phases - 1
        self.alpha: float = 1.0

    # ------------------------------------------------------------------ #
    # Phase control (called by the strategy)
    # ------------------------------------------------------------------ #
    def set_phase(self, phase: int, alpha: float = 1.0) -> None:
        """Set the active growth phase and fade-in coefficient.

        Args:
            phase: Phase index in ``[0, num_phases - 1]``.
            alpha: Fade-in coefficient in ``[0, 1]``.

        Raises:
            ValueError: If ``phase`` or ``alpha`` are out of range.
        """
        if not 0 <= phase < self.num_phases:
            raise ValueError(f"phase must be in [0, {self.num_phases - 1}], got {phase}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.current_phase = phase
        self.alpha = alpha

    def current_resolution(self) -> int:
        """Output resolution at the active phase."""
        return 4 * (2**self.current_phase)

    # ------------------------------------------------------------------ #
    # Latent extraction
    # ------------------------------------------------------------------ #
    def _to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Coerce a forward input into a ``[B, z_dim]`` latent vector.

        Accepts a flat latent (``[B, z_dim]`` / ``[B, z_dim, 1, 1]``) or a
        conditioning image (``[B, in_channels, H, W]``), which is average-pooled to
        a ``_COND_GRID x _COND_GRID`` grid and linearly mapped to the latent, so
        the generator runs under image-conditioned ``gan`` training as well as
        pure latent sampling.
        """
        if x.dim() == 2:
            if x.shape[1] != self.z_dim:
                raise ValueError(f"latent input has dim {x.shape[1]}, expected z_dim={self.z_dim}")
            return x
        if x.dim() == 4:
            b, c = x.shape[0], x.shape[1]
            if c == self.z_dim and x.shape[2] == 1 and x.shape[3] == 1:
                return x.reshape(b, self.z_dim)
            if c != self.in_channels:
                raise ValueError(
                    f"conditioning image has {c} channels, expected in_channels={self.in_channels}"
                )
            return self.cond_proj(self.cond_pool(x).reshape(b, -1))
        raise ValueError(f"Unsupported forward input rank {x.dim()} for ProgressiveGAN")

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate an image at the active phase resolution.

        Args:
            x: Either a latent vector or a conditioning image (see
                :meth:`_to_latent`).

        Returns:
            Generated image ``[B, out_channels, R, R]`` with
            ``R == current_resolution()``.
        """
        z = self._to_latent(x)
        b = z.shape[0]

        # 4x4 base feature map.
        feat = self.project(z).reshape(b, self.base_channels, 4, 4)
        feat = self.base_pixel_norm(self.base_act(feat))

        if self.current_phase == 0:
            return self.base_to_rgb(feat)

        # Run all phases up to (but excluding) the active one.
        prev_feat = feat
        prev_to_rgb = self.base_to_rgb
        for s in range(self.current_phase - 1):
            block = self.phases[s]
            prev_feat = block(prev_feat, upsample=True)
            prev_to_rgb = block.to_rgb

        active = self.phases[self.current_phase - 1]
        new_feat = active(prev_feat, upsample=True)
        new_rgb = active.to_rgb(new_feat)

        if self.alpha >= 1.0:
            return new_rgb

        # Fade-in: upsample previous-phase RGB to the new resolution and blend.
        old_rgb = F.interpolate(prev_to_rgb(prev_feat), scale_factor=2, mode="nearest")
        return self.fade_in(old_rgb, new_rgb, self.alpha)
