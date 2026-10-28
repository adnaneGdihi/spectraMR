"""Nested cold-diffusion masks from a non-Cartesian trajectory (#612).

Cold diffusion assumes ``M_{t+1} subset-of M_t``. A trajectory re-drawn per rung
breaks that: uniform radial angles ``arange(n) * pi / n`` share almost no spokes
between ``n`` and ``n + 1``, and a spiral whose arm offset and turn count follow
the rung moves every sample. Here each family has ONE fixed sample ordering per
grid -- golden-angle spokes, or golden-angle-rotated interleaves of one spiral arm
-- and a rung keeps a prefix of it. A bin is sampled at a rung iff the first
sample to land in it comes before that rung's cut-off, so nesting is structural.

Two refusals replace the silent fallbacks of the binary search this replaced:

* a rung with ``R <= 1`` is the identity and returns every bin, since a rasterised
  trajectory never covers the grid;
* a rung with ``1 < R < 1 / COVERAGE_CEILING`` asks for more of the grid than any
  trajectory covers, and raises when the accelerator is built.
"""

from __future__ import annotations

import math
from typing import Final

import torch

from spectramr.infrastructure.physics.trajectories import TrajectoryFactory

#: Build-time bound on the fraction of the grid a rasterised ordering reaches.
#: Samples lie inside the ellipse inscribed in the grid (pi/4 of it), and a
#: finite ordering on a small grid saturates a little below that; 0.75 is under
#: the measured saturation on every grid from 32x32 up (see the paired test).
#: A grid whose ordering still falls short raises at mask time, naming its ceiling.
COVERAGE_CEILING: Final[float] = 0.75

#: Ordering length, as a multiple of ``max(H, W)`` spokes (radial) or of
#: ``H * W`` samples (spiral). Long enough to saturate the inscribed ellipse.
_RADIAL_SPOKES_PER_SIDE: Final[int] = 4
_SPIRAL_SAMPLES_PER_BIN: Final[int] = 8


def validate_ladder_reachable(accelerator: object, family: str) -> None:
    """Raise if a declared rung needs more of the grid than a trajectory covers.

    Args:
        accelerator: A built ``KSpaceAccelerator`` (reads ``num_timesteps`` and
            ``get_acceleration_factor``).
        family: Family name for the message.

    Raises:
        ValueError: On the first rung with ``1 < R < 1 / COVERAGE_CEILING``.
    """
    floor = 1.0 / COVERAGE_CEILING
    for t in range(int(accelerator.num_timesteps)):  # type: ignore[attr-defined]
        r = float(accelerator.get_acceleration_factor(t))  # type: ignore[attr-defined]
        if 1.0 + 1e-9 < r < floor:
            msg = (
                f"{family}: rung t={t} declares R={r:.4g}, which needs "
                f"{1.0 / r:.1%} of k-space. A rasterised {family} trajectory covers at "
                f"most {COVERAGE_CEILING:.0%} of the grid (the inscribed ellipse), so "
                f"rungs must be R=1 (the identity, fully sampled) or R >= {floor:.4g}."
            )
            raise ValueError(msg)


def _rotated(traj: torch.Tensor, angle_offset: float) -> torch.Tensor:
    """``traj`` turned by ``angle_offset`` radians about k = 0; the identity at 0.

    Rotating every sample is adding the offset to every spoke or interleave
    angle, so a second realisation of the same ordering needs no second generator.
    """
    if angle_offset == 0.0:
        return traj
    c, s = math.cos(angle_offset), math.sin(angle_offset)
    rotation = torch.tensor([[c, -s], [s, c]], dtype=traj.dtype, device=traj.device)
    return rotation @ traj


def golden_radial_ordering(
    height: int, width: int, *, angle_offset: float = 0.0
) -> tuple[torch.Tensor, int]:
    """Golden-angle (111.25 deg) spokes, one fixed set per grid; returns ``(traj, quantum)``.

    The quantum is ``samples_per_spoke``: a rung keeps whole spokes. ``angle_offset``
    rotates the first spoke off the readout axis; the default is the ladder's ordering.
    """
    spokes = _RADIAL_SPOKES_PER_SIDE * max(height, width)
    traj, _ = TrajectoryFactory.get_radial_trajectory(
        im_size=(height, width), num_spokes=spokes, golden_angle=True
    )
    return _rotated(traj, angle_offset), max(height, width)


def golden_spiral_ordering(
    height: int, width: int, *, num_turns: float, samples_per_arm: int, angle_offset: float = 0.0
) -> tuple[torch.Tensor, int]:
    """One spiral arm at a fixed turn count, interleaves rotated by 137.5 deg.

    Samples are ordered by (interleave, readout) and the quantum is one sample, so a
    rung may end mid-arm and every budget is hit exactly. ``angle_offset`` rotates
    every interleave, as for :func:`golden_radial_ordering`.
    """
    arms = max(1, math.ceil(_SPIRAL_SAMPLES_PER_BIN * height * width / samples_per_arm))
    traj, _ = TrajectoryFactory.get_spiral_trajectory(
        im_size=(height, width),
        num_arms=arms,
        num_turns=num_turns,
        samples_per_arm=samples_per_arm,
        golden_angle=True,
    )
    return _rotated(traj, angle_offset), 1


class NestedTrajectoryLadder:
    """First-hit ranking of one ordering on one grid; rungs are prefixes of it.

    Args:
        rows: Grid row of every sample, in acquisition order.
        cols: Grid column of every sample, in acquisition order.
        height: Grid height.
        width: Grid width.
        quantum: A rung keeps a multiple of this many samples (a whole spoke).
        core: ``[H, W]`` boolean ACS core, kept at every rung below the identity.
        family: Family name for messages.
    """

    def __init__(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        height: int,
        width: int,
        quantum: int,
        core: torch.Tensor,
        family: str,
    ) -> None:
        if quantum < 1:
            raise ValueError(f"quantum must be >= 1, got {quantum}")
        self.height, self.width, self.quantum, self.family = height, width, quantum, family
        flat = (rows * width + cols).cpu()
        n = flat.numel()
        first = torch.full((height * width,), n, dtype=torch.long)
        first.scatter_reduce_(0, flat, torch.arange(n, dtype=torch.long), reduce="amin")
        self._first = first.view(height, width)
        self._core = core.cpu().bool().view(height, width)
        outside = self._first[~self._core]
        self._hits = outside[outside < n].sort().values
        self._core_bins = int(self._core.sum())
        self.ceiling = (self._core_bins + self._hits.numel()) / (height * width)

    def sample_count(self, desired_fraction: float) -> int:
        """Smallest whole-quantum prefix whose mask holds ``desired_fraction`` of the grid.

        Raises:
            ValueError: If the whole ordering cannot reach the fraction.
        """
        bins = self.height * self.width
        need = math.ceil(desired_fraction * bins - 1e-9) - self._core_bins
        if need <= 0:
            return 0
        if need > self._hits.numel():
            msg = (
                f"{self.family}: a {self.height}x{self.width} grid needs "
                f"{desired_fraction:.1%} of k-space, but the trajectory covers at most "
                f"{self.ceiling:.1%} of it. Declare R=1 (the identity) or "
                f"R >= {1.0 / self.ceiling:.4g} for this grid."
            )
            raise ValueError(msg)
        samples = int(self._hits[need - 1]) + 1
        return math.ceil(samples / self.quantum) * self.quantum

    def mask(self, samples: int) -> torch.Tensor:
        """``[H, W]`` boolean mask: the core plus every bin hit by the first ``samples``."""
        return (self._first < samples) | self._core
