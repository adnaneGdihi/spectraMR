"""What a non-Cartesian rung actually measured, as one object.

A sample-domain fidelity term needs four things that a grid tensor cannot
carry: the samples, which of them this rung retained, where they sit in
k-space, and the operator that projects an image onto those coordinates.
Bundling them means the consumer either has the whole measurement or has
nothing -- a partially-threaded set of four separate kwargs would let a term
compute a plausible number from an incomplete one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

__all__ = ["SampleMeasurement", "SampleProjector"]


class SampleProjector(Protocol):
    """The one operator method a sample-domain consumer needs."""

    def forward_project(self, image: Tensor, trajectory: Tensor) -> Tensor:
        """Project ``[B, C, H, W]`` complex image onto ``[B, C, N]`` samples."""
        ...


@dataclass(frozen=True)
class SampleMeasurement:
    """The measurement a non-Cartesian forward process produced for one step.

    Args:
        samples: ``y``, complex ``[B, C, N]`` over the FULL trajectory. Stored
            unmasked so a consumer can mask at whichever rung it is scoring.
        mask: ``m_t``, ``[B, N]``, 1 where this rung retained the sample.
        trajectory: ``[2, N]`` or ``[B, 2, N]`` radians in ``[-pi, pi]``.
        projector: the operator that produced ``samples``; reused rather than
            rebuilt, because a ``torchkbnufft`` object allocates interpolation
            tables per construction.
    """

    samples: Tensor
    mask: Tensor
    trajectory: Tensor
    projector: SampleProjector

    def __post_init__(self) -> None:
        if self.samples.dim() != 3:
            raise ValueError(
                f"samples must be [B, C, N], got {tuple(self.samples.shape)}."
            )
        if self.mask.dim() not in (1, 2):
            raise ValueError(f"mask must be [N] or [B, N], got {tuple(self.mask.shape)}.")
        if self.mask.shape[-1] != self.samples.shape[-1]:
            raise ValueError(
                f"mask covers {self.mask.shape[-1]} samples but there are "
                f"{self.samples.shape[-1]}."
            )

    def project(self, image: Tensor) -> Tensor:
        """``A(image)`` on this measurement's trajectory, complex ``[B, C, N]``."""
        return self.projector.forward_project(
            image, self.trajectory.to(image.device)
        )

    def to(self, device: torch.device | str) -> SampleMeasurement:
        """This measurement on ``device``; the projector is device-agnostic."""
        return SampleMeasurement(
            samples=self.samples.to(device),
            mask=self.mask.to(device),
            trajectory=self.trajectory.to(device),
            projector=self.projector,
        )
