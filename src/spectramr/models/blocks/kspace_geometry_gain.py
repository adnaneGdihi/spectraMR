"""Per-bin geometry conditioning for complex k-space feature maps, as a gain.

The geometry planes (:mod:`kspace_geometry_encoding`) tell a k-space backbone
which bins are known at this rung and where each bin sits. Appended to the
input as ``(plane, 0)`` channels they enter the k-space data path additively,
and a plane is not a k-space signal: its mean is an impulse at the image
origin, and the planes that are constant along readout (``support``,
``acquired``, ``ky``, ``pe_gap``) become a line through the centre row. A
bias-free network that maps zero k-space to zero then emits a measurement-
independent term, and the real-only planes also break the global-phase
equivariance the complex convolutions keep.

:class:`KSpaceGeometryGain` carries the same information as a modulation
instead: at each U-Net stage every complex channel ``c`` at bin ``k`` is scaled
by ``1 + g_c(planes(k))`` with a real ``g``. Zero stays zero, ``e^{i theta} z``
maps to ``e^{i theta}`` times the output, and the gain is zero-initialised so
the block starts at the identity.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.models.blocks.kspace_attention_kernels import crop_to_level_grid


class KSpaceGeometryGain(nn.Module):
    """One zero-initialised 1x1 map from the planes to a real gain per stage.

    Args:
        num_planes: Geometry planes per sample (``K`` of the ``[N, K, H, W]`` input).
        stage_channels: Complex channel count at each stage the gain is applied
            to, in the order :meth:`forward`'s ``stage`` indexes them.

    Raises:
        ValueError: ``num_planes`` is not positive or ``stage_channels`` is empty.
    """

    def __init__(self, num_planes: int, stage_channels: tuple[int, ...]) -> None:
        super().__init__()
        if num_planes <= 0 or not stage_channels:
            raise ValueError(
                f"KSpaceGeometryGain needs num_planes > 0 and at least one stage; got "
                f"num_planes={num_planes}, stage_channels={stage_channels}."
            )
        self.num_planes = int(num_planes)
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.maps = nn.ModuleList(
            nn.Conv2d(self.num_planes, c, kernel_size=1) for c in self.stage_channels
        )
        for conv in self.maps:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, features: torch.Tensor, planes: torch.Tensor, stage: int) -> torch.Tensor:
        """Scale ``features`` per bin and complex channel.

        Args:
            features: ``[N, 2C, h, w]`` interleaved ``(Re, Im)`` or ``[N, C, h, w]``
                complex, on a centre crop of the planes' grid.
            planes: ``[N, K, H, W]`` real geometry planes on the full input grid.
            stage: Index into ``stage_channels``.

        Returns:
            ``features`` scaled, same shape and dtype.
        """
        if planes.dim() != 4 or planes.shape[1] != self.num_planes:
            raise ValueError(
                f"KSpaceGeometryGain expects [N, {self.num_planes}, H, W] planes, got "
                f"{tuple(planes.shape)}."
            )
        complex_channels = features.shape[1] if features.is_complex() else features.shape[1] // 2
        if complex_channels != self.stage_channels[stage]:
            raise ValueError(
                f"KSpaceGeometryGain stage {stage} was built for "
                f"{self.stage_channels[stage]} complex channels, got {complex_channels}."
            )
        if planes.shape[0] != features.shape[0]:
            raise ValueError(
                f"KSpaceGeometryGain: plane batch {planes.shape[0]} does not match the "
                f"feature batch {features.shape[0]}."
            )
        level = crop_to_level_grid(planes, features.shape[-2], features.shape[-1])
        gain = 1.0 + self.maps[stage](level.to(features.real.dtype))
        if features.is_complex():
            return features * gain
        # Broadcast over the (Re, Im) pair rather than materialise a doubled gain.
        pairs = features.unflatten(1, (complex_channels, 2))
        return (pairs * gain.unsqueeze(2)).flatten(1, 2)


__all__ = ["KSpaceGeometryGain"]
