"""Progressive-growing GAN discriminator.

Mirror of the :class:`~spectramr.models.gans.progressive_gan.ProgressiveGAN`
generator following Karras *et al.* [Karras et al., ICLR 2018]. The
discriminator downsamples symmetrically; a fade-in interpolant blends a
newly-added (higher-resolution) input phase as ``alpha`` is annealed
0 -> 1. A :class:`MinibatchStdDev` channel is appended before the final
4x4 block so the discriminator sees an across-batch variability statistic,
giving it a direct signal for intra-batch mode collapse.

It is registered as the ``progressive_discriminator`` critic, so a YAML
selects it through ``model.discriminator_component.name``. It exposes the
``in_channels`` / ``set_phase`` interface that ``ProgressiveGANStrategy``
requires of the critic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.progressive import FadeIn, MinibatchStdDev
from spectramr.models.registry import register_model


class _ProGANDiscBlock(nn.Module):
    """One discriminator phase: two 3x3 convs then 2x average-pool downsample."""

    def __init__(self, in_ch: int, out_ch: int, in_channels: int) -> None:
        super().__init__()
        self.from_rgb = nn.Conv2d(in_channels, in_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return F.avg_pool2d(x, kernel_size=2)


@register_model(
    role="discriminator",
    name="progressive_discriminator",
    training_mode="gan",
    input_domain="image",
)
class ProgressiveDiscriminator(nn.Module):
    """Progressive-growing GAN discriminator (critic).

    Args:
        in_channels: Channels of the input image (real or generated).
        max_resolution: Final input resolution (power of two, >= 4).
        base_channels: Feature channels of the 4x4 head; per-phase channel
            counts mirror the generator (halved per phase away from 4x4).
        min_channels: Lower bound on per-phase feature channels.
        mbstd_group_size: Group size of the minibatch-stddev layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        max_resolution: int = 256,
        base_channels: int = 512,
        min_channels: int = 16,
        mbstd_group_size: int = 4,
    ) -> None:
        super().__init__()
        if max_resolution < 4 or (max_resolution & (max_resolution - 1)) != 0:
            raise ValueError(f"max_resolution must be a power of two >= 4, got {max_resolution}")
        self.in_channels = in_channels
        self.max_resolution = max_resolution
        self.base_channels = base_channels
        self.num_phases = int(math.log2(max_resolution)) - 1

        # Highest-resolution phase has the fewest channels; the 4x4 head has
        # base_channels. Phase s (0 == 4x4) uses max(base // 2**s, min).
        blocks: list[nn.Module] = []
        for s in range(self.num_phases - 1, 0, -1):
            in_ch = max(base_channels // (2**s), min_channels)
            out_ch = max(base_channels // (2 ** (s - 1)), min_channels)
            blocks.append(_ProGANDiscBlock(in_ch, out_ch, in_channels))
        # ``blocks`` is ordered from the highest resolution down to 8x8.
        self.blocks = nn.ModuleList(blocks)

        # 4x4 head: from_rgb (for phase-0-only input), mbstd, conv, classifier.
        self.from_rgb_base = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.mbstd = MinibatchStdDev(group_size=mbstd_group_size)
        self.head_conv = nn.Conv2d(base_channels + 1, base_channels, kernel_size=3, padding=1)
        self.head_act = nn.LeakyReLU(0.2, inplace=True)
        self.classifier = nn.Linear(base_channels * 4 * 4, 1)

        self.fade_in = FadeIn()

        self.current_phase: int = self.num_phases - 1
        self.alpha: float = 1.0

    def set_phase(self, phase: int, alpha: float = 1.0) -> None:
        """Set the active growth phase and fade-in coefficient (see generator)."""
        if not 0 <= phase < self.num_phases:
            raise ValueError(f"phase must be in [0, {self.num_phases - 1}], got {phase}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.current_phase = phase
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of images.

        Args:
            x: Image tensor ``[B, in_channels, R, R]`` (R should match the
                active phase resolution).

        Returns:
            Real-valued critic scores ``[B, 1]`` (WGAN convention).
        """
        if self.current_phase == 0:
            feat = self.head_act(self.from_rgb_base(x))
        else:
            # The active block consumes the highest-resolution input.
            active_idx = self.num_phases - 1 - self.current_phase
            active = self.blocks[active_idx]
            feat = active.from_rgb(x)
            feat = active(feat)

            if self.alpha < 1.0:
                # Fade-in: downsample the raw input and project it via the
                # next (lower-res) block's from_rgb, then blend.
                down = F.avg_pool2d(x, kernel_size=2)
                if active_idx + 1 < len(self.blocks):
                    old = self.blocks[active_idx + 1].from_rgb(down)
                else:
                    old = self.from_rgb_base(down)
                feat = self.fade_in(old, feat, self.alpha)

            # Remaining lower-resolution blocks down to 8x8.
            for blk in self.blocks[active_idx + 1 :]:
                feat = blk(feat)

            feat = self.from_rgb_base(feat) if feat.shape[1] != self.base_channels else feat

        # 4x4 head.
        feat = self.mbstd(feat)
        feat = self.head_act(self.head_conv(feat))
        feat = feat.reshape(feat.shape[0], -1)
        return self.classifier(feat)
