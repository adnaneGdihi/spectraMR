#!/usr/bin/env python
"""K-space acceleration utilities for the cold diffusion pipelines.

This module provides acceleration patterns for k-space undersampling in
cold diffusion models. It generates different sampling masks for each
timestep, progressively increasing the sampling rate as the diffusion
process advances.

### Mask Generation Logic
```mermaid
flowchart TD
    Start[Generate Mask] --> Config{Check Config}
    Config -->|Seed| Seed[Set Random Seed]
    Config -->|Accel Factor| Scale[Calculate Scale]

    Scale --> Type{Mask Type?}

    Type -->|Cartesian| Lines[Generate Phase Encode Lines]
    Lines --> Center[Add Calibration Region]

    Type -->|Variable Density| PDF[Calculate Probability Density]
    PDF --> Sample[Monte Carlo Sampling]

    Type -->|Radial| RadTraj[Generate Radial Spokes]
    RadTraj --> Grid[Gridding / Rasterization]

    Type -->|Spiral| SpirTraj[Generate Spiral Arms]
    SpirTraj --> Grid

    Center --> Final[Final Binary Mask]
    Sample --> Final
    Grid --> Final
```
"""

import difflib
import importlib
import inspect
import logging
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, cast

import numpy as np
import torch

from spectramr.infrastructure.physics.bounded_cache import BoundedLRUCache
from spectramr.infrastructure.physics.line_axis import line_axis_for

# [PHYSICS] Import TrajectoryFactory
from spectramr.infrastructure.physics.trajectories import TrajectoryFactory
from spectramr.infrastructure.physics.trajectory_ladder import (
    NestedTrajectoryLadder,
    golden_radial_ordering,
    golden_spiral_ordering,
    validate_ladder_reachable,
)

# Get a logger
logger = logging.getLogger(__name__)


# Global configuration for maximum acceleration factor
MAX_ACCELERATION = 64.0

#: Edge-to-DC growth of the variable-density Poisson-disc exclusion radius: the
#: radius at the inscribed ellipse is ``1 + slope`` times the radius at DC, so the
#: density there is ``(1 + slope)**-2`` of the centre's (1/16 at 3).
POISSON_DENSITY_SLOPE = 3.0

#: Rate constant of the ``exponential`` acceleration ramp.
#:
#: Named because it is consumed in BOTH directions -- the forward ramp
#: (:meth:`KSpaceAccelerator._normalized_progress`) and its inverse
#: (:meth:`KSpaceAccelerator.timestep_for_acceleration`) -- and a literal that
#: drifts between the two silently breaks the round-trip the inverse exists to
#: guarantee. ``infrastructure/physics/severity.py`` mirrors the same value on
#: purpose, so an arm that ramps undersampling and degradation severity together
#: gets comparable curves.
_EXPONENTIAL_RATE = 5.0

#: The acceleration schedules this module implements -- the SSOT that
#: ``_normalized_progress`` and ``timestep_for_acceleration`` both dispatch over.
#:
#: Deliberately a literal rather than ``{s.value for s in AccelerationSchedule}``:
#: ``infrastructure/`` must not import ``config/`` schemas, and more importantly a
#: derived set would make a newly-added enum member *look* supported while no
#: branch computed its curve. Adding a member to the enum should break this
#: module until a curve is written for it -- that is the whole point of #789.
_SUPPORTED_SCHEDULES: frozenset[str] = frozenset(
    {"linear", "polynomial", "exponential", "step", "power_law"}
)


def _schedule_name(schedule: object) -> str:
    """Normalise a schedule to its plain string value.

    ``AccelerationSchedule`` is a ``(str, Enum)``, not a ``StrEnum``, so
    ``str(AccelerationSchedule.STEP)`` is ``'AccelerationSchedule.STEP'`` while
    ``AccelerationSchedule.STEP == 'step'`` is ``True``. Code that compares with
    ``==`` is fine and code that goes through ``str()`` is not --
    ``models/diffusion/kspace_process.py`` does the latter. Normalising once at
    construction means neither spelling can reach a comparison.
    """
    return str(getattr(schedule, "value", schedule))


class MaskType(Enum):
    """Types of sampling masks."""

    UNIFORM_CARTESIAN = "uniform_cartesian"
    CARTESIAN_LINES = "cartesian_lines"  # [PHYSICS] Cartesian Line-Based Sampling
    VARIABLE_DENSITY = "variable_density"
    RADIAL = "radial"
    SPIRAL = "spiral"
    # [Experiment 11] Cartesian Mask Expansion
    EQUISPACED = "equispaced"
    GAUSSIAN = "gaussian"
    RANDOM = "random"
    # [Experiment 11 — FDB baseline] Peripheral-to-central Cartesian.
    # Keeps the high-frequency periphery and skips the low-frequency
    # center initially. Used by the Frequency-Decomposed Bridge schedule
    # (FDB) which acquires peripheral first then bridges inward.
    # See TODO/backlog_baseline_replication_experiment_11.md Phase A.3.
    CARTESIAN_PERIPHERAL = "cartesian_peripheral"
    # [Breakthrough 2026] Schramm–Loewner Evolution conformal trajectory.
    # Driven by Brownian motion with rate sqrt(kappa); Hausdorff dimension
    # min(1 + kappa/8, 2). See src/data/transforms/sle_trajectory.py.
    SLE_KAPPA = "sle_kappa"


#: ``MaskType`` members with no accelerator equivalent: they exist only on the
#: static, timestep-free mask path (metrics, dataset transforms). Every OTHER
#: member must be a name ``SamplingPatternRegistry`` also accepts, so a pattern
#: name means one thing regardless of which path a caller reaches (issue #954).
#:
#: Module-level on purpose: a plain assignment inside an ``Enum`` body becomes a
#: member, not a class attribute, which would add a frozenset to ``MaskType``.
#: Enforced by tests/unit/infrastructure/physics/test_masktype_registry_agreement.py.
STATIC_ONLY_MASK_TYPES: frozenset[str] = frozenset(
    {"cartesian_lines", "cartesian_peripheral", "sle_kappa"}
)

#: Canonical accelerator names that have a static, timestep-free equivalent, and
#: which ``MaskType`` renders them as. Lives beside ``MaskType`` because that is
#: the vocabulary being mapped INTO; ``kspace_masks`` used to carry its own copy,
#: which is how the two drifted (issue #954).
#:
#: A canonical name absent from this map is accelerator-only: the static path
#: must raise for it rather than degrade to uniform sampling (pitfall #9).
ACCELERATOR_TO_MASK_TYPE: dict[str, "MaskType"] = {
    "uniform_cartesian": MaskType.UNIFORM_CARTESIAN,
    "equispaced": MaskType.EQUISPACED,
    "variable_density": MaskType.VARIABLE_DENSITY,
    "variable_density_2d_gaussian": MaskType.GAUSSIAN,
    "random_cartesian": MaskType.RANDOM,
    "radial": MaskType.RADIAL,
    "spiral": MaskType.SPIRAL,
    # `linear` has no static pattern of its own and has always rendered as
    # uniform_cartesian here. KSpaceMaskGenerator's default_pattern IS "linear",
    # so dropping this entry breaks the default rather than some exotic arm.
    "linear": MaskType.UNIFORM_CARTESIAN,
}


class MaskGenerator:
    """Central utility for generating sampling masks with reproducible seeds.

    Now consolidated into sampling.py to serve as the unified source of truth for all mask logic.
    """

    def __init__(self, seed: int | None = None):
        """Initialize mask generator with optional seed for reproducibility.

        ``seed`` is stored and used to build *local* RNGs
        (``np.random.RandomState(self.seed)`` / ``torch.Generator().manual_seed``)
        inside the generation methods — so reproducibility does NOT depend on
        the process-global RNG state. Reseeding the global ``random`` /
        ``numpy`` / ``torch`` generators here (the old behaviour) clobbered
        dataloader-shuffle / dropout entropy on every construction; since
        ``MaskGenerator(seed=epoch)`` is built inside the per-step loss path of
        SSL pretraining, that froze the global RNG to ``epoch`` every step
        (audit 2026-06, pitfall: determinism via initialize_accelerator, not
        ad-hoc global reseeds). The global reseed is removed; per-call masks stay
        reproducible via ``self.seed``.
        """
        self.seed = seed

    @staticmethod
    def trajectory_bin_indices(
        trajectory: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grid ``(row, col)`` of every trajectory sample, in sample order.

        The one owner of the rasteriser's index rule, so the nested trajectory ladder
        (:mod:`~spectramr.infrastructure.physics.trajectory_ladder`) bins samples
        exactly as :meth:`_rasterize_trajectory` does. ``[-pi, pi]`` maps to
        ``[0, N]`` with ``k = 0`` on bin ``N // 2``; each sample goes to its nearest
        bin, then clamps to the grid. Flooring sent a sample on the centre line to
        either side by the sign of a float residue (``cos(pi/2) = -4.4e-8``), so a
        0/90-degree spoke pair was not a transpose pair (#2551).
        """
        x_norm = trajectory[0] / (2 * np.pi) + 0.5
        y_norm = trajectory[1] / (2 * np.pi) + 0.5
        x_idx = torch.round(x_norm * width).long().clamp(0, width - 1)
        y_idx = torch.round(y_norm * height).long().clamp(0, height - 1)
        return y_idx, x_idx

    @staticmethod
    def variable_density_poisson_disc(
        allowed: torch.Tensor,
        radius: float,
        generator: torch.Generator,
        *,
        density_slope: float = POISSON_DENSITY_SLOPE,
    ) -> torch.Tensor:
        r"""A saturated variable-density Poisson-disc set on the bins ``allowed`` marks.

        Random sequential adsorption: bins are visited once in a seeded random order
        and a bin is kept unless an earlier kept bin :math:`j` lies closer than
        :math:`r_j = \mathrm{radius}\,(1 + \mathrm{slope}\,\rho_j)`, with
        :math:`\rho` the distance from DC in half-grid units (1 on the inscribed
        ellipse). Visiting every bin is dart throwing run to saturation, so the
        density follows :math:`1/r^2` -- denser at DC -- and is set by ``radius``
        alone. ``PoissonDiskKSpaceAccelerator`` keeps its fixed-radius master
        pattern; that one is a cold-diffusion ladder and is not seeded per call.

        Args:
            allowed: ``(H, W)`` bool; only these bins may be kept.
            radius: Exclusion radius at DC, in bins. Below 1 keeps every visited bin.
            generator: Seeds the visiting order.
            density_slope: How fast the radius grows towards the edge; 0 is uniform.

        Returns:
            ``(H, W)`` bool, a subset of ``allowed``.
        """
        height, width = allowed.shape
        ky = (torch.arange(height, dtype=torch.float64) - height // 2) / (height / 2)
        kx = (torch.arange(width, dtype=torch.float64) - width // 2) / (width / 2)
        rho = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).clamp(max=1.0)
        quantised = torch.round(4.0 * radius * (1.0 + density_slope * rho)).numpy().astype(int)
        blocked = (~allowed.bool()).numpy().copy()
        kept = np.zeros((height, width), dtype=bool)
        discs: dict[int, tuple[int, np.ndarray]] = {}
        flat_blocked = blocked.reshape(-1)
        for index in torch.randperm(height * width, generator=generator).tolist():
            if flat_blocked[index]:
                continue
            y, x = divmod(index, width)
            kept[y, x] = True
            q = int(quantised[y, x])
            if q not in discs:
                reach = max(0, math.ceil(q / 4.0) - 1)
                offsets = np.arange(-reach, reach + 1)
                dist2 = offsets[:, None] ** 2 + offsets[None, :] ** 2
                discs[q] = (reach, 16 * dist2 < q * q)
            reach, disc = discs[q]
            y0, y1 = max(0, y - reach), min(height, y + reach + 1)
            x0, x1 = max(0, x - reach), min(width, x + reach + 1)
            blocked[y0:y1, x0:x1] |= disc[
                y0 - y + reach : y1 - y + reach, x0 - x + reach : x1 - x + reach
            ]
            blocked[y, x] = True
        return torch.from_numpy(kept)

    def _rasterize_trajectory(
        self,
        trajectory: torch.Tensor,
        shape: tuple[int, ...],
        dcf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rasterize continuous trajectory onto Cartesian grid.

        Maps [-pi, pi] coordinates to [0, N-1] indices.
        Uses nearest neighbor regridding for binary masks.

        Args:
            trajectory: (2, N) tensor of k-space coordinates in [-pi, pi]
            shape: (H, W) or (C, H, W)
            dcf: Optional density compensation function (unused for binary mask)

        Returns:
            Binary mask tensor
        """
        if len(shape) == 3:
            H, W = shape[1], shape[2]
            channels = shape[0]
        else:
            H, W = shape[0], shape[1]
            channels = 1

        mask = torch.zeros((channels, H, W), dtype=torch.bool)  # device? will be moved later
        y_idx, x_idx = self.trajectory_bin_indices(trajectory, H, W)

        # Set mask
        if channels == 1:
            mask[0, y_idx, x_idx] = True
        else:
            mask[:, y_idx, x_idx] = True

        if len(shape) == 2:
            return mask[0]
        return mask

    def generate_mask(
        self,
        mask_type: str | MaskType,
        shape: tuple[int, ...],
        acceleration_factor: float = 2.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate sampling mask.

        Args:
            mask_type: Type of mask to generate
            shape: Shape of the mask (height, width) or (height, width, coils)
            acceleration_factor: Undersampling factor
            **kwargs: Additional parameters for specific mask types

        Returns:
            Binary mask tensor

        .. mermaid::

            flowchart TD
                Start[generate_mask] --> CheckType{MaskType?}
                CheckType -->|UNIFORM/CARTESIAN| Cartesian[Cartesian Lines]
                CheckType -->|VARIABLE_DENSITY| VarDensity[Variable Density]
                CheckType -->|RADIAL| Radial[Radial Trajectory]
                CheckType -->|SPIRAL| Spiral[Spiral Trajectory]
                CheckType -->|GAUSSIAN| Gaussian[Gaussian PDF]

                Cartesian --> ACS{Center Fraction?}
                VarDensity --> PDF[Generate PDF]
                Radial --> Traj[TrajectoryFactory]
                Spiral --> Traj

                ACS -->|Yes| SampleCenter
                ACS -->|No| SampleOuter

                PDF --> Sample[Weighted Random Sample]
                Traj --> Raster[Rasterize to Grid]

                SampleCenter --> Output
                SampleOuter --> Output
                Sample --> Output
                Raster --> Output
        """
        if isinstance(mask_type, str):
            mask_type = MaskType(mask_type)

        if mask_type == MaskType.UNIFORM_CARTESIAN:
            return self._generate_cartesian_line_mask(
                shape,
                acceleration_factor,
                **kwargs,
            )
        if mask_type == MaskType.CARTESIAN_LINES:
            return self._generate_cartesian_line_mask(
                shape,
                acceleration_factor,
                **kwargs,
            )
        if mask_type == MaskType.VARIABLE_DENSITY:
            return self._generate_variable_density(shape, acceleration_factor, **kwargs)
        if mask_type == MaskType.RADIAL:
            # [PHYSICS] Use TrajectoryFactory for valid radial spokes
            # acceleration_factor roughly maps to num_spokes / N
            # Nyquist: N spokes. Accel R -> N/R spokes.
            H = shape[-2] if len(shape) >= 2 else shape[0]
            num_spokes = int(H / acceleration_factor)
            traj, _ = TrajectoryFactory.get_radial_trajectory(
                im_size=(H, H),
                num_spokes=num_spokes,
                golden_angle=kwargs.get("golden_angle", False),
            )
            return self._rasterize_trajectory(traj, shape)

        if mask_type == MaskType.SPIRAL:
            # [PHYSICS] Use TrajectoryFactory for valid spiral
            # Accel factor determines num_arms?
            # Standard spiral needs N arms for full sampling?
            # Rough approx: arms = 4 * accel? No, arms decreases with accel.
            # Let's say baseline 48 arms. Accel 4 -> 12 arms.
            H = shape[-2] if len(shape) >= 2 else shape[0]
            base_arms = 48  # heuristic for 256x256
            num_arms = max(1, int(base_arms / acceleration_factor))
            traj, _ = TrajectoryFactory.get_spiral_trajectory(im_size=(H, H), num_arms=num_arms)
            return self._rasterize_trajectory(traj, shape)
        if mask_type == MaskType.EQUISPACED:
            return self._generate_equispaced(shape, acceleration_factor, **kwargs)
        if mask_type == MaskType.GAUSSIAN:
            return self._generate_gaussian(shape, acceleration_factor, **kwargs)
        if mask_type == MaskType.RANDOM:
            return self._generate_random(shape, acceleration_factor, **kwargs)
        if mask_type == MaskType.CARTESIAN_PERIPHERAL:
            return self._generate_cartesian_peripheral(shape, acceleration_factor, **kwargs)
        if mask_type == MaskType.SLE_KAPPA:
            return self._generate_sle_kappa(shape, acceleration_factor, **kwargs)
        raise ValueError(f"Unknown mask type: {mask_type}")

    def _generate_sle_kappa(
        self,
        shape: tuple[int, ...],
        acceleration_factor: float,
        kappa: float = 2.0,
        n_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Schramm–Loewner Evolution trajectory sampling.

        Dispatches to :func:`spectramr.data.transforms.sle_trajectory.build_sle_kspace_mask`.
        ``acceleration_factor`` controls the number of trace points
        (``n_steps``) when not explicitly provided: a higher acceleration
        means fewer points and hence a sparser mask.
        """
        from spectramr.data.transforms.sle_trajectory import build_sle_kspace_mask

        if len(shape) == 3:
            h, w = shape[1], shape[2]
        else:
            h, w = shape[0], shape[1]
        if n_steps is None:
            # Target |M| ≈ (H * W) / acceleration_factor lattice points.
            target = max(2, int(h * w / max(1.0, acceleration_factor)))
            n_steps = min(target, h * w)
        gen = None
        if self.seed is not None:
            gen = torch.Generator().manual_seed(int(self.seed))
        mask_2d = build_sle_kspace_mask(
            kappa=kappa,
            shape=(h, w),
            n_steps=n_steps,
            generator=gen,
        )
        if len(shape) == 3:
            return mask_2d.unsqueeze(0).expand(shape[0], -1, -1).contiguous()
        return mask_2d

    def _generate_cartesian_peripheral(
        self,
        shape: tuple[int, ...],
        acceleration_factor: float,
        center_skip_fraction: float = 0.08,
        **kwargs,
    ) -> torch.Tensor:
        """Peripheral Cartesian sampling — keeps high-frequency lines, skips center.

        Used by the FDB (Frequency-Decomposed Bridge) baseline, which
        acquires the high-frequency periphery first and bridges inward.
        Inverse of the usual ACS-centered pattern.

        Args:
            shape: Mask shape ``(H, W)`` or ``(C, H, W)``.
            acceleration_factor: Target peripheral undersampling rate.
            center_skip_fraction: Fraction of low-frequency center to
                explicitly leave unsampled.

        Returns:
            Binary mask tensor with ``shape`` and ``dtype=torch.float32``.
        """
        if len(shape) == 3:
            _, H, W = shape
        else:
            H, W = shape
        mask_1d = torch.zeros(H, dtype=torch.float32)

        # Identify the central band to skip.
        center_lines = max(1, int(round(center_skip_fraction * H)))
        center_start = (H - center_lines) // 2
        center_end = center_start + center_lines

        # Peripheral indices: everything outside the center band.
        peripheral_idx = [i for i in range(H) if not center_start <= i < center_end]
        # Sample one in every ``acceleration_factor`` peripheral lines.
        step = max(1, int(round(acceleration_factor)))
        for j, idx in enumerate(peripheral_idx):
            if j % step == 0:
                mask_1d[idx] = 1.0

        # Broadcast along the frequency-encode dimension.
        mask_2d = mask_1d.view(H, 1).expand(H, W).clone()
        if len(shape) == 3:
            return mask_2d.unsqueeze(0).expand(shape).clone()
        return mask_2d

    @staticmethod
    def get_gaussian_pdf(
        num_lines: int,
        std_scale: float = 4.0,
    ) -> np.ndarray:
        """Generate Gaussian PDF for sampling.

        Args:
            num_lines: Number of lines in the dimension
            std_scale: Scale factor for standard deviation (higher = narrower PDF)

        Returns:
            Probability density array (normalized to sum to 1)
        """
        # Gaussian PDF: exp(-x^2 / (2*sigma^2))
        # x is distance from center [-1, 1]
        x = np.linspace(-1, 1, num_lines)

        # sigma in [-1, 1] domain.
        sigma = 1.0 / std_scale

        pdf = np.exp(-(x**2) / (2 * sigma**2))

        # Normalize to probability
        if pdf.sum() > 0:
            return pdf / pdf.sum()
        return np.ones_like(pdf) / len(pdf)

    @staticmethod
    def get_variable_density_pdf(
        num_lines: int,
        density_power: float = 2.0,
    ) -> np.ndarray:
        """Generate variable density PDF (polynomial decay).

        Args:
            num_lines: Number of lines
            density_power: Power of decay (higher = more center focused)

        Returns:
            Probability density array (sum=1)
        """
        # Distance from center (normalized [0, 1])
        # We want 0 at center, 1 at edges
        indices = np.arange(num_lines)
        center = num_lines // 2
        distance = np.abs(indices - center) / center

        # Probability weight
        # p(y) ~ (1 - |y|)^p  where y is normalized distance
        # Wait, if distance is 0 (center), weight is 1. If distance is 1 (edge), weight is 0.
        # This matches (1-distance)**power

        weights = (1 - distance) ** density_power

        # Normalize
        if weights.sum() > 0:
            return weights / weights.sum()
        return np.ones_like(weights) / len(weights)

    def _generate_cartesian_line_mask(
        self,
        shape: tuple[int, ...],
        acceleration_factor: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Simulates physical Phase Encoding (PE) undersampling.
        Args:
            shape: (C, H, W) or (H, W). W is Readout (Frequency Encode), H is Phase Encode.
        """
        if len(shape) == 3:
            # (Coils, Height, Width) -> H is PE, W is Readout
            height, width = shape[1], shape[2]
        else:
            height, width = shape[0], shape[1]

        # [Physics] 1. Readout Dimension (Width) is ALWAYS fully sampled.
        # The gradient is on, we get these samples "for free".
        # Mask is invariant along W.

        # 2. Autocalibration Signal (ACS) - Low Frequencies
        # We always sample the center lines for phase reference/contrast.
        num_lines = height
        center_fraction = kwargs.get("center_fraction", 0.08)  # standard 8% ACS
        num_low_freq = int(num_lines * center_fraction)

        # Ensure even ACS
        if num_low_freq % 2 != 0:
            num_low_freq += 1

        center_start = (num_lines - num_low_freq) // 2
        center_end = center_start + num_low_freq

        # 3. Variable Density Sampling for High Frequencies
        # Total lines budget based on acceleration
        total_budget = int(num_lines / acceleration_factor)
        remaining_budget = max(0, total_budget - num_low_freq)

        # Exclude center from random selection
        outer_indices = np.concatenate(
            [np.arange(0, center_start), np.arange(center_end, num_lines)]
        )

        # PDF favors lines closer to center (k-space energy decay)
        # p(y) ~ (1 - |y|)^p
        # Normalize to [-1, 1] across the whole height
        norm_dist = np.abs(np.linspace(-1, 1, num_lines))
        decay_power = kwargs.get("decay_power", 2.0)
        pdf = (1 - norm_dist) ** decay_power  # Power law density

        # Extract probabilities for outer indices
        outer_probs = pdf[outer_indices]

        # Normalize probabilities
        if outer_probs.sum() > 0:
            outer_probs /= outer_probs.sum()
        else:
            outer_probs = np.ones_like(outer_probs) / len(outer_probs)

        # Deterministic Selection based on Seed (Critical for Consistency)
        # Use self.seed if available, else ``seed`` from kwargs, else 42. NOT
        # ``mask_seed``: that is the YAML spelling, translated to ``seed`` by
        # ``models/diffusion/kspace_process.py:208``. This comment claimed otherwise for
        # long enough to be the trail a caller followed into issue #1059.
        seed = self.seed if self.seed is not None else kwargs.get("seed", 42)
        rng = np.random.RandomState(seed)

        # ``rng.choice(replace=False, p=...)`` can only draw as many lines as have
        # NON-ZERO probability, and the power-law density is exactly 0 at the two
        # k-space edges (``norm_dist == 1`` there). Asking for more than that raises
        # "Fewer non-zero entries in p than size" — which is what any acceleration
        # low enough to need every outer line did; R=1 always did. A budget that
        # covers the whole outer band means "sample it all": nothing left to draw.
        n_selectable = int((outer_probs > 0).sum())
        if remaining_budget >= len(outer_indices):
            selected_indices = outer_indices
        else:
            selected_indices = rng.choice(
                outer_indices,
                size=min(remaining_budget, n_selectable),
                replace=False,
                p=outer_probs,
            )

        # Construct Mask
        # Shape: (H, W) -> broadcast along W
        mask_lines = np.zeros(height, dtype=np.float32)
        mask_lines[center_start:center_end] = 1.0  # ACS
        mask_lines[selected_indices] = 1.0  # Outer lines

        # Expand to 2D: (H, 1) * (1, W) -> (H, W)
        mask = torch.from_numpy(mask_lines).view(-1, 1).repeat(1, width)

        return mask

    def _generate_equispaced(
        self, shape: tuple[int, ...], acceleration_factor: float, **kwargs
    ) -> torch.Tensor:
        """Generate regular equispaced mask (SENSE/GRAPPA style)."""
        if len(shape) == 3:
            height, width = shape[1], shape[2]
        else:
            height, width = shape[0], shape[1]

        center_fraction = kwargs.get("center_fraction", 0.08)

        # Calculate ACS
        num_lines = height
        num_low_freq = int(num_lines * center_fraction)
        center_start = (num_lines - num_low_freq) // 2
        center_end = center_start + num_low_freq

        mask = torch.zeros(height, width, dtype=torch.float32)

        # 1. ACS Region
        mask[center_start:center_end, :] = 1.0

        # 2. Equispaced sampling
        spacing = max(1, int(round(acceleration_factor)))
        mask[::spacing, :] = 1.0

        # Ensure ACS overrides grid (union)
        mask[center_start:center_end, :] = 1.0

        return mask

    def _generate_gaussian(
        self, shape: tuple[int, ...], acceleration_factor: float, **kwargs
    ) -> torch.Tensor:
        """Generate Gaussian variable density mask."""
        if len(shape) == 3:
            height, width = shape[1], shape[2]
        else:
            height, width = shape[0], shape[1]

        center_fraction = kwargs.get("center_fraction", 0.08)
        std_scale = kwargs.get("std_scale", 2.5)

        num_lines = height
        num_low_freq = int(num_lines * center_fraction)
        center_start = (num_lines - num_low_freq) // 2
        center_end = center_start + num_low_freq

        total_budget = int(num_lines / acceleration_factor)
        remaining_budget = max(0, total_budget - num_low_freq)

        # Get PDF
        pdf = self.get_gaussian_pdf(num_lines, std_scale)

        # Extract probabilities for outer indices
        outer_indices = np.concatenate(
            [np.arange(0, center_start), np.arange(center_end, num_lines)]
        )

        probs = pdf[outer_indices]

        # Renormalize for the subset
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones_like(probs) / len(probs)

        # Sample
        seed = self.seed if self.seed is not None else kwargs.get("seed", 42)
        rng = np.random.RandomState(seed)

        selected_indices = rng.choice(
            outer_indices,
            size=min(remaining_budget, len(outer_indices)),
            replace=False,
            p=probs,
        )

        mask = torch.zeros(height, width, dtype=torch.float32)
        mask[center_start:center_end, :] = 1.0
        mask[selected_indices, :] = 1.0

        return mask

    def _generate_random(
        self, shape: tuple[int, ...], acceleration_factor: float, **kwargs
    ) -> torch.Tensor:
        """Generate random mask with Poisson-disk minimum-distance constraint.

        Uses rejection sampling with a minimum gap between selected PE lines
        to break coherent aliasing patterns that cause horizontal banding.
        Falls back to uniform random if the budget cannot be filled with
        the gap constraint.

        Args:
            shape: (C, H, W) or (H, W) k-space shape.
            acceleration_factor: Undersampling ratio.
            **kwargs: ``center_fraction`` (default 0.08), ``min_gap`` (default 2),
                ``seed``.

        Returns:
            Binary mask tensor (H, W).
        """
        if len(shape) == 3:
            height, width = shape[1], shape[2]
        else:
            height, width = shape[0], shape[1]

        center_fraction = kwargs.get("center_fraction", 0.08)
        # Minimum gap between selected lines to suppress coherent aliasing
        min_gap: int = kwargs.get("min_gap", 2)

        num_lines = height
        num_low_freq = int(num_lines * center_fraction)
        center_start = (num_lines - num_low_freq) // 2
        center_end = center_start + num_low_freq

        total_budget = int(num_lines / acceleration_factor)
        remaining_budget = max(0, total_budget - num_low_freq)

        # Candidate indices (outside ACS)
        outer_indices = np.concatenate(
            [np.arange(0, center_start), np.arange(center_end, num_lines)]
        )

        seed = self.seed if self.seed is not None else kwargs.get("seed", 42)
        rng = np.random.RandomState(seed)

        # Poisson-disk: shuffle candidates and greedily accept with min gap
        shuffled = outer_indices.copy()
        rng.shuffle(shuffled)

        selected: list[int] = []
        for idx in shuffled:
            if len(selected) >= remaining_budget:
                break
            # Check min distance to all already-selected lines
            if all(abs(int(idx) - int(s)) >= min_gap for s in selected):
                selected.append(int(idx))

        # If gap constraint was too strict, fill remaining with uniform random
        if len(selected) < remaining_budget:
            remaining = set(outer_indices.tolist()) - set(selected)
            extra = rng.choice(
                list(remaining),
                size=min(remaining_budget - len(selected), len(remaining)),
                replace=False,
            )
            selected.extend(extra.tolist())

        selected_indices = np.array(selected, dtype=np.intp)

        mask = torch.zeros(height, width, dtype=torch.float32)
        mask[center_start:center_end, :] = 1.0
        if len(selected_indices) > 0:
            mask[selected_indices, :] = 1.0

        return mask

    def _generate_variable_density(
        self,
        shape: tuple[int, ...],
        acceleration_factor: float,
        **kwargs,
    ) -> torch.Tensor:
        """Generate variable density sampling mask."""
        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 3:
            height, width, _ = shape
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        center_fraction = kwargs.get("center_fraction", 0.1)
        density_power = kwargs.get("density_power", 1.5)

        mask = torch.zeros(height, width, dtype=torch.float32)

        center_size = int(height * center_fraction)
        center_start = height // 2 - center_size // 2
        center_end = center_start + center_size

        mask[center_start:center_end, :] = 1.0

        total_lines = height
        target_lines = int(total_lines / acceleration_factor)

        lines_sampled_so_far = center_end - center_start
        lines_needed = max(0, target_lines - lines_sampled_so_far)

        if lines_needed > 0:
            pdf = self.get_variable_density_pdf(height, density_power)

            selection_mask = np.ones(height, dtype=bool)
            selection_mask[center_start:center_end] = False

            outer_indices = np.where(selection_mask)[0]
            probs = pdf[outer_indices]

            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones_like(probs) / len(probs)

            seed = self.seed if self.seed is not None else kwargs.get("seed", 42)
            rng = np.random.RandomState(seed)

            selected_indices = rng.choice(
                outer_indices,
                size=min(lines_needed, len(outer_indices)),
                replace=False,
                p=probs,
            )

            mask[selected_indices, :] = 1.0

        if mask.sum() == 0:
            mask[height // 2, :] = 1.0

        return mask

    def _generate_radial(
        self,
        shape: tuple[int, ...],
        acceleration_factor: float,
        **kwargs,
    ) -> torch.Tensor:
        """Generate radial sampling mask."""
        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 3:
            height, width, _ = shape
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        num_spokes = kwargs.get("num_spokes", int(height / acceleration_factor))
        golden_angle = kwargs.get("golden_angle", False)

        mask = torch.zeros(height, width, dtype=torch.float32)
        center_y, center_x = height // 2, width // 2

        ga = np.pi * (3 - math.sqrt(5))

        for spoke in range(num_spokes):
            if golden_angle:
                angle = spoke * ga
            else:
                angle = 2 * np.pi * spoke / num_spokes

            for r in range(max(height, width) // 2):
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                if 0 <= y < height and 0 <= x < width:
                    mask[y, x] = 1.0

        center_radius = 3
        y_indices, x_indices = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )
        distances = torch.sqrt(
            (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2,
        )
        mask[distances <= center_radius] = 1.0
        return mask

    def _generate_spiral(
        self,
        shape: tuple[int, ...],
        acceleration_factor: float,
        **kwargs,
    ) -> torch.Tensor:
        """Generate Archimedean spiral sampling mask."""
        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 3:
            height, width, _ = shape
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        base_arms = kwargs.get("base_arms", 48)
        num_arms = max(1, int(base_arms / acceleration_factor))
        if "num_arms" in kwargs:
            num_arms = kwargs["num_arms"]

        turns = kwargs.get("turns", 4.0)
        spiral_width = kwargs.get("spiral_width", 2)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32) - height // 2,
            torch.arange(width, dtype=torch.float32) - width // 2,
            indexing="ij",
        )

        mask = torch.zeros(height, width, dtype=torch.float32)
        max_radius = min(height, width) // 2 * 0.9

        for arm in range(num_arms):
            angle_offset = 2 * np.pi * arm / num_arms
            final_angle = turns * 2 * np.pi
            b = max_radius / final_angle
            a = 0

            theta_range = torch.linspace(0, final_angle, 1000)
            spiral_radii = a + b * theta_range
            spiral_angles = theta_range + angle_offset

            spiral_y = spiral_radii * torch.sin(spiral_angles)
            spiral_x = spiral_radii * torch.cos(spiral_angles)

            for i in range(len(spiral_radii)):
                center_y = int(spiral_y[i] + height // 2)
                center_x = int(spiral_x[i] + width // 2)

                y_start = max(0, center_y - spiral_width // 2)
                y_end = min(height, center_y + spiral_width // 2 + 1)
                x_start = max(0, center_x - spiral_width // 2)
                x_end = min(width, center_x + spiral_width // 2 + 1)

                mask[y_start:y_end, x_start:x_end] = 1.0

        center_radius = max(8, int(spiral_width * 3))
        distances = torch.sqrt(x_coords**2 + y_coords**2)
        center_mask = distances <= center_radius
        mask[center_mask] = 1.0

        return mask

    def get_mask_properties(self, mask: torch.Tensor) -> dict[str, Any]:
        """Get properties of a sampling mask."""
        mask_sum = mask.sum().item()
        total_elements = mask.numel()

        acceleration_factor = float("inf") if mask_sum == 0 else total_elements / mask_sum

        properties = {
            "shape": mask.shape,
            "acceleration_factor": acceleration_factor,
            "sampling_ratio": mask_sum / total_elements,
            "center_sampled": self._is_center_sampled(mask),
        }
        return properties

    def _is_center_sampled(self, mask: torch.Tensor) -> bool:
        """Check if center region is properly sampled."""
        height, width = mask.shape[:2]
        center_y, center_x = height // 2, width // 2
        center_radius = min(8, min(height, width) // 4)

        y_indices, x_indices = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )
        distances = torch.sqrt(
            (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2,
        )
        center_mask = distances <= center_radius

        return (mask * center_mask).sum().item() > 0

    def generate_mask_4d(
        self,
        mask_type: str | MaskType,
        shape_4d: tuple[int, int, int, int],
        acceleration_factor: float = 2.0,
        *,
        per_timepoint: bool = False,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate a 4-D Cartesian mask of shape ``(T, S, H, W)``.

        Extension of :meth:`generate_mask` for spatiotemporal fMRI / MRF
        cohorts (audit_plan_novel_fmri.md fMRI §1 / MRF §1).

        Args:
            mask_type: Any value accepted by :meth:`generate_mask`.
            shape_4d: ``(n_time, n_slice, H, W)``.
            acceleration_factor: Per-(T, S) acceleration.
            per_timepoint: When ``True`` re-draws the in-plane mask
                independently per ``(t, s)`` (breaks temporal aliasing);
                when ``False`` (default) shares the same in-plane mask
                across the full 4-D volume.
            device: Optional output device.
            **kwargs: Passed through to :meth:`generate_mask`.

        Returns:
            Boolean tensor of shape ``(T, S, H, W)``.
        """
        n_time, n_slice, H, W = shape_4d
        if n_time < 1 or n_slice < 1:
            raise ValueError(f"shape_4d must have positive T and S, got {shape_4d}")
        if per_timepoint:
            slabs = []
            for _ in range(n_time):
                inner = []
                for _ in range(n_slice):
                    inner.append(
                        self.generate_mask(
                            mask_type=mask_type,
                            shape=(H, W),
                            acceleration_factor=acceleration_factor,
                            **kwargs,
                        )
                    )
                slabs.append(torch.stack(inner, dim=0))
            mask = torch.stack(slabs, dim=0)
        else:
            base = self.generate_mask(
                mask_type=mask_type,
                shape=(H, W),
                acceleration_factor=acceleration_factor,
                **kwargs,
            )
            mask = base.unsqueeze(0).unsqueeze(0).expand(n_time, n_slice, H, W).contiguous()
        return mask.to(device) if device is not None else mask


def create_mask_generator(seed: int | None = None) -> MaskGenerator:
    """Factory function to create mask generator."""
    return MaskGenerator(seed)


#: Construction kwargs the BASE consumes through ``kwargs.get`` rather than a named
#: parameter, so signature introspection alone cannot see them. Keep this in step with
#: the reads in :meth:`KSpaceAccelerator.__init__` -- a name dropped from here becomes
#: a spurious "unknown kwarg" rejection, and one added but never read re-opens exactly
#: the silent-absorption hole :func:`_reject_unknown_accelerator_kwargs` closes.
_BASE_DYNAMIC_KWARGS: frozenset[str] = frozenset(
    {"acceleration_range", "seed", "base_acceleration", "min_center_fraction"}
)

#: Spellings :class:`ColdDiffusionAccelerator` translates before dispatch. They are
#: accepted at the boundary and never reach a concrete accelerator under these names.
_ACCELERATOR_KWARG_ALIASES: frozenset[str] = frozenset(
    {"mask_direction", "phase_encode_axis", "schedule_type"}
)


def _resolve_accelerator_cls(registry_entry: Any, acceleration_type: str) -> type:
    """The concrete class behind a registry entry: direct, ``{"class": ...}`` or import path.

    Single owner deliberately. :class:`ColdDiffusionAccelerator` resolves a class to
    construct and :func:`_accelerator_kwarg_vocabulary` resolves all of them to read
    their signatures; a second copy of this walk is the duplicate-resolver defect that
    the kwarg check below exists to catch a symptom of.
    """
    cls_ref = registry_entry["class"] if isinstance(registry_entry, dict) else registry_entry

    if isinstance(cls_ref, str):
        module_path, class_name = cls_ref.rsplit(".", 1)
        cls_ref = getattr(importlib.import_module(module_path), class_name)

    if cls_ref is None and acceleration_type == "multi_mask":
        from spectramr.models.utils.multi_mask_accelerator import MultiMaskAccelerator

        cls_ref = MultiMaskAccelerator

    if cls_ref is None:
        raise ValueError(f"Could not resolve class for {acceleration_type}")
    return cast(type, cls_ref)


def _accepted_accelerator_kwargs(accelerator_cls: type) -> set[str]:
    """Every construction kwarg *accelerator_cls* itself consumes.

    All of the registered families declare ``**kwargs``, so Python rejects nothing: the
    union of named parameters across the MRO plus the base's dynamic reads IS what the
    class reads, and anything outside it is dropped on the floor.
    """
    accepted = set(_BASE_DYNAMIC_KWARGS) | set(_ACCELERATOR_KWARG_ALIASES)
    for klass in accelerator_cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for param in inspect.signature(init).parameters.values():
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                accepted.add(param.name)
    accepted.discard("self")
    return accepted


def _accelerator_kwarg_vocabulary() -> set[str]:
    """Every construction kwarg read by ANY registered family.

    Recomputed per call rather than cached at import: the registry is open, and a
    family registered after this module loaded (``test_sampling_registry`` does exactly
    that) must widen the vocabulary rather than be judged against a stale snapshot.
    Nineteen signature walks is a construction-time cost, not a training-loop one.
    """
    vocabulary = set(_BASE_DYNAMIC_KWARGS) | set(_ACCELERATOR_KWARG_ALIASES)
    for name, entry in _ACCELERATOR_REGISTRY.items():
        try:
            vocabulary |= _accepted_accelerator_kwargs(_resolve_accelerator_cls(entry, name))
        except (ImportError, ValueError, AttributeError):
            # An unresolvable entry is a registry defect, but it is not THIS check's to
            # report -- ``test_sampling_reachability`` owns that. Skipping keeps the
            # vocabulary a lower bound, so the worst case is a missed rejection rather
            # than a spurious one.
            continue
    return vocabulary


def family_reads_line_axis(acceleration_type: str) -> bool:
    """Whether the family named ``acceleration_type`` consumes a ``line_axis``.

    ``mask_direction`` reaches a family as ``line_axis``; one that does not consume
    it (2-D point patterns, trajectories) drops it in ``**kwargs`` (#1082). Read off
    the class signature, the same walk the unknown-kwarg gate uses, so there is no
    hand-kept list of line families.
    """
    from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry

    canonical = SamplingPatternRegistry.resolve(acceleration_type)
    entry = _ACCELERATOR_REGISTRY[canonical]
    return "line_axis" in _accepted_accelerator_kwargs(_resolve_accelerator_cls(entry, canonical))


def _reject_unknown_accelerator_kwargs(accelerator_cls: type, kwargs: dict[str, Any]) -> None:
    """Raise on a construction kwarg NO registered family reads.

    Silently absorbing one is not a cosmetic wart. ``mask_seed`` is the spelling every
    experiment YAML uses; the production resolver translates it to ``seed``
    (``models/diffusion/kspace_process.py:208``), but any direct caller that reaches for
    the YAML spelling got ``self.seed = None``, and mask generation then fell back to the
    GLOBAL RNG. That is still reproducible under ``seed_everything`` -- and still wrong,
    because each call draws a fresh permutation instead of truncating one fixed ranking,
    so the cascade stops being nested (25 of 27 transitions re-add k-space). Cold
    diffusion's forward process assumes ``M_{t+1} subset-of M_t``, and a silent fallback
    that breaks it is exactly what non-negotiable #3 forbids. Issue #1059.

    The test is the VOCABULARY, not this family's signature, and the difference is
    deliberate. ``_resolve_process_kwargs`` builds one family-AGNOSTIC dict and hands it
    to whichever family the YAML selected, so ``density_power`` reaching a radial
    accelerator is the dispatch pattern working as designed, not a caller error.
    Per-family strictness would outlaw it. What it cannot be is a name outside the whole
    vocabulary: no family reads it under any dispatch, so it is a typo or a wrong
    spelling every time.

    That leaves knobs a family accepts and discards -- ``random_cartesian`` does 2-D bin
    top-K and structurally has no line axis, yet 36 cohort arms declare
    ``mask_direction: phase``. Real, out of scope here, and measured in issue #1082:
    narrowing that is the per-family schema's job, not construction's.

    Raises:
        TypeError: If any key in *kwargs* is outside the vocabulary, naming near-misses.
    """
    vocabulary = _accelerator_kwarg_vocabulary()
    unknown = sorted(set(kwargs) - vocabulary)
    if not unknown:
        return
    name = getattr(accelerator_cls, "__name__", str(accelerator_cls))
    known = sorted(vocabulary)
    hints = [
        f"{key!r} -> did you mean {close[0]!r}?"
        for key in unknown
        if (close := difflib.get_close_matches(key, known, n=1, cutoff=0.6))
    ]
    detail = (" " + " ".join(hints)) if hints else ""
    raise TypeError(
        f"{unknown!r} is not read by any registered k-space accelerator, so "
        f"{name} would silently discard it.{detail} Known kwargs: {known!r}."
    )


class KSpaceAccelerator(ABC):
    """Abstract base class for k-space acceleration patterns."""

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        acceleration_schedule: str = "linear",
        schedule_power: float = 2.0,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            acceleration_schedule (str): Description.
            schedule_power (float): Description.
        """
        if num_timesteps <= 0:
            msg = "num_timesteps must be a positive integer"
            raise ValueError(msg)
        self.num_timesteps = num_timesteps
        self.max_acceleration = max(1.0, max_acceleration)
        # Normalised to the plain string value and validated HERE rather than at
        # first use: an unsupported schedule is a config error, and surfacing it
        # when the accelerator is built puts it inside `spectramr audit` instead of
        # a few thousand steps into a run (non-negotiable #3 -- an unknown
        # registered-option value raises, it never degrades to a default).
        self.acceleration_schedule = _schedule_name(acceleration_schedule)
        if self.acceleration_schedule not in _SUPPORTED_SCHEDULES:
            msg = (
                f"unknown acceleration_schedule {self.acceleration_schedule!r}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_SCHEDULES))}."
            )
            raise ValueError(msg)
        self.schedule_power = schedule_power
        # Track whether the user explicitly provided acceleration_range.
        # The auto-default [1.0, max_acceleration] silently overrode
        # base_acceleration in the step schedule (returned 1.0 for
        # ratio<0.5 instead of the user's base, e.g., 2.0). Distinguishing
        # explicit vs default lets ``get_acceleration_factor`` route to
        # the base/max binary fallback when no range was declared.
        self._acceleration_range_explicit = "acceleration_range" in kwargs
        self.acceleration_range = kwargs.get("acceleration_range", [1.0, self.max_acceleration])
        # Declared on the base because ``_ensure_sample_budget`` reads it. Every
        # concrete accelerator that randomises overwrites it in its own __init__;
        # the trajectory families leave it None.
        self.seed: int | None = kwargs.get("seed")
        # Likewise for the acceleration floor. ``get_acceleration_factor`` reads it
        # as ``getattr(self, "base_acceleration", 1.0)``, and only two of sixteen
        # accelerators ever stored it -- so for the other fourteen the configured
        # value was swallowed by **kwargs and silently replaced by 1.0, making
        # ``base_acceleration: 4`` mean "fully sampled at t=0" (issue #957).
        # Subclasses that declare it as a named parameter still assign it after
        # this, with the same clamp, so their behaviour is unchanged.
        # FractionalVariableDensityAccelerator exposes ``base_acceleration`` as a
        # read-only property aliasing ``min_acceleration``; that is deliberate, so
        # do not clobber it.
        if not isinstance(getattr(type(self), "base_acceleration", None), property):
            self.base_acceleration: float = max(1.0, float(kwargs.get("base_acceleration", 1.0)))
        # And likewise for the ACS floor, for the same reason (#534). Subclasses
        # that declare it as a named parameter assign it after this call and are
        # unaffected; the rest now have it available to ``_current_center_fraction``
        # instead of swallowing it in ``**kwargs``. ``None`` means "not declared",
        # which the helper reads as "ACS does not shrink".
        if "min_center_fraction" not in vars(self):
            self.min_center_fraction: float | None = kwargs.get("min_center_fraction")
        # The one place a declared floor is checked against the tightest budget, so
        # the answer cannot depend on which family the arm picked (#2061).
        self._validate_center_floor(self.max_acceleration)

    def _current_center_fraction(self, timestep: int, center_fraction: float) -> float:
        """The ACS fraction at ``timestep``, shrinking toward ``min_center_fraction``.

        ``center_fraction`` is the ACS at the LOWEST acceleration and
        ``min_center_fraction`` the floor at the HIGHEST. Holding the ACS at its
        nominal width for the whole ladder is what makes the top rungs
        unrealisable: an always-sampled 8% band IS the entire budget at R=12.5,
        so every nominally-higher rung realises the same ~12x mask (issue #534,
        and the reason ``scripts/ci/ladder_baseline.txt`` exists).

        Returns ``center_fraction`` unchanged unless a floor strictly below it was
        declared, so an arm that does not ask for a shrinking ACS -- which, as of
        this change, is every arm in the corpus -- keeps its masks byte-identical.

        The formula is lifted verbatim from
        :class:`UniformCartesianKSpaceAccelerator`, which has interpolated since
        #534 while its variable-density siblings did not. That includes applying
        ``schedule_power`` a second time on top of ``_normalized_progress`` (which
        already applies it under ``power_law`` / ``polynomial``); the double
        application is
        preserved deliberately so the one family that already shrank keeps its
        exact masks. Changing the curve is a separate, corpus-wide decision.
        """
        floor = self.min_center_fraction
        if floor is None or floor >= center_fraction:
            return center_fraction
        scaled_progress = self._normalized_progress(timestep) ** self.schedule_power
        return center_fraction - scaled_progress * (center_fraction - floor)

    def _guaranteed_core_fraction(self, center_fraction: float) -> float:
        """The centred fraction a *fixed-ranking* family must keep at every rung.

        The sibling helper :meth:`_current_center_fraction` re-derives the ACS
        width per timestep, which is correct only for families that re-rank per
        timestep. A family that ranks k-space ONCE and truncates by budget has
        exactly one guaranteed core for the whole cascade, and its width is
        decided at the HIGHEST acceleration -- where the budget is smallest.
        That is precisely what ``min_center_fraction`` declares, so it, not
        ``center_fraction``, is the right quantity for the core.

        Using ``center_fraction`` instead is what produced the disc collapse in
        #1069: a nominal 8% core against a 1/16 = 6.25% budget IS the entire
        mask, so the top rungs realised a pure low-pass filter with no
        incoherent aliasing at all.

        Returns ``center_fraction`` unchanged when no floor was declared, so an
        arm that does not ask for one keeps its masks byte-identical.
        """
        floor = self.min_center_fraction
        if floor is None:
            return center_fraction
        return max(0.0, min(float(floor), center_fraction))

    def _validate_center_floor(self, max_acceleration: float) -> None:
        """Refuse an explicit ACS floor the tightest budget cannot honour.

        An EXPLICIT floor wider than ``1 / max_acceleration`` is a config error: no
        mask can deliver it, so fail at BUILD rather than silently at the top of the
        ladder (non-negotiable #3). Only when explicitly declared -- the implicit
        default mirrors ``center_fraction``, and rejecting that would refuse the
        library default at every ``max_acceleration`` above ~30.

        Called once, from ``KSpaceAccelerator.__init__``, because the guard is a
        property of the DECLARATION, not of any one family's geometry: every family
        answers the same declaration the same way. Callers must forward an undeclared
        floor as ``None`` (``build_accelerator_kwargs`` does); substituting
        ``center_fraction`` turns it into a declared one.
        """
        if self.min_center_fraction is None:
            return
        min_budget_fraction = 1.0 / max(1.0, float(max_acceleration))
        if self.min_center_fraction > min_budget_fraction:
            msg = (
                f"min_center_fraction ({self.min_center_fraction:.4f}) exceeds the "
                f"sampling budget at max_acceleration={max_acceleration:g} "
                f"({min_budget_fraction:.4f} of k-space). The ACS floor alone would "
                "consume more than the whole budget, so the declared floor is "
                "unreachable. Lower min_center_fraction or lower max_acceleration."
            )
            raise ValueError(msg)

    #: Whether ``_ensure_sample_budget`` may add or drop samples to hit the exact
    #: point budget. Geometric trajectories (radial, spiral) and the contiguous
    #: partial-Fourier block define their own support and must not be topped up:
    #: doing so would put samples off-trajectory. Set False in those subclasses
    #: rather than isinstance-checking them from the base (open/closed).
    enforces_sample_budget: bool = True

    def _make_generator(self, device: torch.device, timestep: int) -> torch.Generator | None:
        """Create a timestep-INVARIANT RNG for mask generation.

        The seed deliberately does not include ``timestep``: cold diffusion needs
        the nested property ``M_{t+1} subset-of M_t``, and re-seeding per timestep
        makes lines twinkle in and out between adjacent steps instead of being
        truncated from one fixed ranking.

        Args:
            device: Device the generator draws on.
            timestep: Accepted and ignored; kept so subclasses can override with
                a timestep-dependent policy if one is ever justified.

        Returns:
            A seeded generator, or None when the accelerator has no seed (callers
            then fall back to the global RNG).
        """
        if self.seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen

    @abstractmethod
    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate acceleration mask for given timestep.

        Args:
            kspace_shape: Shape of k-space data (height, width) or
                (channels, height, width)
            timestep: Current diffusion timestep (0 to num_timesteps-1)
            device: Target device for the mask

        Returns:
            Binary mask tensor of rank 3, ``(channels, height, width)``, where True
            indicates a sampled point. **The returned rank is always 3, whatever the
            rank of ``kspace_shape``** -- a 2-tuple yields ``(1, height, width)``, not
            ``(height, width)``.

        The rank is part of the contract because consumers index it. It used to be
        left unstated, and the two family groups then disagreed: every Cartesian
        family allocated ``(channels, H, W)`` unconditionally, while the trajectory
        families returned ``MaskGenerator._rasterize_trajectory`` verbatim, which
        drops to ``(H, W)`` for a 2-tuple shape. ``ColdDiffusionAccelerator.
        _first_drop_map`` took ``[0]`` of the result, so for a trajectory family under
        a 2-tuple that selected k-space ROW 0 and broadcast it -- turning a declared
        R=7.8 mask into R=128 with every row identical. Subclasses must therefore
        normalise before returning; :meth:`_as_channelled` does it in one call.
        """

    @staticmethod
    def _unpack_shape(kspace_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """_unpack_shape.

        Args:
            kspace_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, int, int]: Description.
        """
        if len(kspace_shape) == 2:
            height, width = kspace_shape
            channels = 1
        elif len(kspace_shape) == 3:
            channels, height, width = kspace_shape
        else:
            msg = f"Unsupported k-space shape: {kspace_shape}"
            raise ValueError(msg)
        return channels, height, width

    @staticmethod
    def _as_channelled(mask: torch.Tensor, channels: int) -> torch.Tensor:
        """Normalise a mask to the rank-3 ``(channels, H, W)`` return contract.

        The trajectory families build their mask through
        :meth:`MaskGenerator._rasterize_trajectory`, which mirrors the rank of the
        shape it was handed and so returns ``(H, W)`` for a 2-tuple. Every consumer
        that indexes the result -- ``_first_drop_map`` most consequently -- reads
        that as a channel axis and silently takes a single k-space row. Normalising
        here rather than at each consumer keeps the fix at the contract boundary:
        a family that forgets to call this is caught by the rank check in
        :meth:`ColdDiffusionAccelerator._first_drop_map` instead of producing a
        plausible-looking mask at the wrong acceleration.

        Args:
            mask: A ``(H, W)`` or ``(C, H, W)`` boolean mask.
            channels: The channel count the caller's ``kspace_shape`` declared.

        Returns:
            The mask as ``(channels, H, W)``.
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            msg = f"acceleration mask must be rank 2 or 3, got shape {tuple(mask.shape)}"
            raise ValueError(msg)
        if mask.shape[0] == channels:
            return mask
        if mask.shape[0] == 1:
            # ``.contiguous()`` because callers write into the result (the ACS patch
            # below does); an expanded view aliases one row of memory per channel and
            # an in-place write to it raises.
            return mask.expand(channels, -1, -1).contiguous()
        msg = (
            f"acceleration mask has {mask.shape[0]} channels but the k-space shape "
            f"declared {channels}"
        )
        raise ValueError(msg)

    def _clamp_timestep(self, timestep: int) -> int:
        """_clamp_timestep.

        Args:
            timestep (int): Description.
        Returns:
            int: Description.
        """
        if self.num_timesteps <= 1:
            return 0
        return max(0, min(int(timestep), self.num_timesteps - 1))

    def _normalized_progress(self, timestep: int) -> float:
        """Shape ``t / (T - 1)`` into normalised progress under the declared schedule.

        The dispatch is EXHAUSTIVE over ``_SUPPORTED_SCHEDULES`` and raises on
        anything else. It used to handle two names and let the rest fall through
        to ``return ratio``, which made ``polynomial`` a byte-identical alias of
        ``linear`` on all 19 registered accelerators while ``schedule_power`` --
        documented as "power parameter for the *polynomial* schedule" -- was the
        one knob it never read (#789). The undersampling ladder is the training
        curriculum for every timestep-conditioned k-space diffusion arm, so that
        was a declared curriculum silently replaced by a different one.

        Two of the five schedules return the raw ratio, and both do so
        deliberately -- they are explicit branches, not the old fall-through:

        * ``linear`` is the identity by definition.
        * ``step`` returns the ratio UNSHAPED because its consumer treats it as a
          ladder *index*, not an interpolation weight:
          ``get_acceleration_factor`` computes ``int(ratio * len(range))``.
          Shaping it here would warp which rung each timestep lands on.

        ``polynomial`` is ``ratio ** schedule_power`` -- the same curve as
        ``power_law``, which is what the name and ``schedule_power``'s own
        description imply, and what the sibling ramp
        :func:`~spectramr.infrastructure.physics.severity.shape_severity_ratio`
        has always computed for it. The two ramps disagreed until this change;
        severity's was the correct reading.

        Args:
            timestep: Diffusion timestep, clamped into ``[0, T - 1]``.

        Returns:
            Normalised progress in ``[0, 1]``: 0 at ``t=0``, 1 at ``t=T-1``.

        Raises:
            ValueError: If ``acceleration_schedule`` is not a supported name.
        """
        if self.num_timesteps <= 1:
            return 1.0
        clamped = self._clamp_timestep(timestep)
        ratio = float(clamped) / float(self.num_timesteps - 1)

        schedule = self.acceleration_schedule
        if schedule in ("linear", "step"):
            return ratio
        if schedule in ("power_law", "polynomial"):
            return ratio**self.schedule_power
        if schedule == "exponential":
            return (math.exp(ratio * _EXPONENTIAL_RATE) - 1.0) / (math.exp(_EXPONENTIAL_RATE) - 1.0)
        msg = (
            f"unknown acceleration_schedule {schedule!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_SCHEDULES))}."
        )
        raise ValueError(msg)

    def get_acceleration_factor(self, timestep: int) -> float:
        """Get acceleration factor for given timestep.

        Args:
            timestep: Current diffusion timestep

        Returns:
            Acceleration factor (higher = more undersampling): ``base`` at
            ``t=0`` and ``max_acceleration`` at ``t=T-1``, the last timestep.
        """
        # ratio goes from 0 (t=0) to 1 (t=T-1)
        ratio = self._normalized_progress(timestep)

        base = getattr(self, "base_acceleration", 1.0)

        if self.acceleration_schedule == "step":
            # Use the explicit `acceleration_range` only if the user
            # actually declared it. The base-class auto-default
            # [1.0, max] would otherwise silently override
            # base_acceleration here.
            range_is_explicit = getattr(self, "_acceleration_range_explicit", False)
            if (
                range_is_explicit
                and hasattr(self, "acceleration_range")
                and isinstance(self.acceleration_range, list)
                and len(self.acceleration_range) > 0
            ):
                steps = len(self.acceleration_range)
                # ratio is exactly 1 at t=T-1, so idx would be `steps` without `min`
                idx = min(int(ratio * steps), steps - 1)
                return float(self.acceleration_range[idx])
            else:
                return base if ratio < 0.5 else self.max_acceleration

        span = max(0.0, self.max_acceleration - base)
        return base + ratio * span

    def timestep_for_acceleration(self, acceleration: float) -> int:
        """Inverse of :meth:`get_acceleration_factor`.

        Returns the timestep ``t`` whose mask realises the requested
        ``acceleration`` under the *currently configured* schedule. This is the
        SSOT for any caller that needs "give me a mask at R=X" — notably the
        validation cascade in
        ``infrastructure/training/strategies/diffusion.py`` — so the
        forward schedule (``get_acceleration_factor``) and the reverse
        request (this method) always agree.

        Why it matters: the cascade used to hard-code
        ``t = T*(R - base)/(max - base)`` (a linear inverse). When the YAML
        declared ``schedule_type: step`` with a non-uniform
        ``acceleration_range`` the cascade still picked linear timesteps —
        ``R=8`` resolved to ``t=200``, which the step schedule then
        decoded as ``R=4``. Routing through this method keeps the two
        sides of the schedule honest.

        Step schedule semantics:
            - When ``acceleration_range`` is explicit, ``acceleration`` MUST
              appear in it (within ``1e-6`` tolerance). Otherwise a
              ``ValueError`` is raised — the step schedule has no
              well-defined inverse for off-grid values, and silently
              snapping to the nearest bucket is exactly the failure mode
              this method exists to prevent.
            - Without an explicit range, the binary step inverse returns 0
              for ``acceleration ≤ base`` and ``T-1`` for everything
              above (matching the binary forward at the ``ratio=0.5``
              boundary).

        Continuous schedules invert in closed form: ``power_law`` and
        ``polynomial`` (which share the ``ratio ** schedule_power`` curve) by
        the reciprocal power, ``exponential`` by log, ``linear`` by identity.

        Args:
            acceleration: Desired acceleration factor.

        Returns:
            Integer timestep in ``[0, num_timesteps - 1]`` such that
            ``get_acceleration_factor(t) ≈ acceleration`` under the
            current ``acceleration_schedule``.

        Raises:
            ValueError: If the schedule cannot represent ``acceleration``
                (e.g. step schedule with an off-grid value).
        """
        if self.num_timesteps <= 1:
            return 0

        base = getattr(self, "base_acceleration", 1.0)
        T_minus_1 = self.num_timesteps - 1

        if self.acceleration_schedule == "step":
            range_is_explicit = getattr(self, "_acceleration_range_explicit", False)
            if (
                range_is_explicit
                and hasattr(self, "acceleration_range")
                and isinstance(self.acceleration_range, list)
                and len(self.acceleration_range) > 0
            ):
                steps = len(self.acceleration_range)
                for idx, r in enumerate(self.acceleration_range):
                    if abs(float(r) - float(acceleration)) < 1e-6:
                        # Pick the centre of bucket idx so a single-int
                        # round-trip lands inside the same bucket even if
                        # ``ratio * steps`` rounds down at the boundary.
                        ratio = (idx + 0.5) / steps
                        timestep = int(round(ratio * T_minus_1))
                        # Being IN the declared list is not the same as being
                        # REALISED by the schedule (issue #1171). The forward
                        # index is ``min(int(t/(T-1) * steps), steps - 1)``, so
                        # when ``steps > num_timesteps`` it cannot take every
                        # value: entries are skipped, and a request for a skipped
                        # rung would otherwise return a timestep that decodes to
                        # a DIFFERENT rung. Silently snapping to a neighbouring
                        # bucket is precisely the failure this method exists to
                        # prevent, so re-ask the forward schedule and refuse if
                        # it disagrees.
                        realised = self.get_acceleration_factor(timestep)
                        if abs(float(realised) - float(acceleration)) > 1e-6:
                            raise ValueError(
                                f"timestep_for_acceleration: R={acceleration!r} "
                                f"is declared in acceleration_range (index "
                                f"{idx} of {steps}) but is realised at NO "
                                f"timestep: t={timestep} decodes to "
                                f"R={realised!r}. With schedule_type='step' the "
                                f"forward index is min(int(ratio * "
                                f"{steps}), {steps - 1}), which skips entries "
                                f"whenever len(acceleration_range) ({steps}) "
                                f"exceeds num_timesteps "
                                f"({self.num_timesteps}). Declare one rung per "
                                f"timestep (or fewer) so every declared level "
                                f"is reachable."
                            )
                        return timestep
                raise ValueError(
                    f"timestep_for_acceleration: schedule_type='step' with "
                    f"acceleration_range={self.acceleration_range!r} cannot "
                    f"represent R={acceleration!r} (must be one of the "
                    f"declared bucket values within 1e-6 tolerance)."
                )
            # Binary fallback: anything > base lands in the upper bucket.
            if acceleration <= base + 1e-6:
                return 0
            return T_minus_1

        # Continuous schedules invert via _normalized_progress.
        span = max(0.0, self.max_acceleration - base)
        if span <= 0.0:
            return 0
        progress = max(0.0, min(1.0, (float(acceleration) - base) / span))

        # Exhaustive, and mirroring `_normalized_progress` branch for branch: an
        # inverse that falls through to linear for a name the forward ramp shapes
        # is exactly the desynchronisation this method exists to prevent -- it is
        # how `polynomial` inverted as linear while ramping as a power (#789).
        if self.acceleration_schedule in ("power_law", "polynomial"):
            power = max(1e-6, float(self.schedule_power))
            ratio = progress ** (1.0 / power)
        elif self.acceleration_schedule == "exponential":
            # progress = (e^{k*ratio} - 1) / (e^k - 1)
            #   →  ratio = log(1 + progress * (e^k - 1)) / k
            ratio = (
                math.log(1.0 + progress * (math.exp(_EXPONENTIAL_RATE) - 1.0)) / _EXPONENTIAL_RATE
            )
        elif self.acceleration_schedule == "linear":
            ratio = progress
        else:
            # `step` returned above; anything else was rejected at construction.
            msg = (
                f"unknown acceleration_schedule {self.acceleration_schedule!r}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_SCHEDULES))}."
            )
            raise ValueError(msg)

        return int(round(ratio * T_minus_1))

    def _target_sampling_fraction(self, timestep: int) -> float:
        """_target_sampling_fraction.

        Args:
            timestep (int): Description.
        Returns:
            float: Description.
        """
        acceleration = max(1.0, self.get_acceleration_factor(timestep))
        # 1.0 at t=0 (R=1) down to 1/max_acceleration at t=T-1.
        return 1.0 / acceleration

    @staticmethod
    def _apply_center_patch(
        mask: torch.Tensor,
        center_fraction: float,
    ) -> int:
        """_apply_center_patch.

        Args:
            mask (torch.Tensor): Description.
            center_fraction (float): Description.
        Returns:
            int: Description.
        """
        _, height, width = mask.shape
        fraction = max(0.0, min(center_fraction, 0.5))
        if fraction <= 0.0:
            center_row = height // 2
            center_col = width // 2
            mask[:, center_row, center_col] = True
            return 1

        side_scale = math.sqrt(fraction)
        center_h = max(1, int(round(height * side_scale)))
        center_w = max(1, int(round(width * side_scale)))
        center_h = min(center_h, height)
        center_w = min(center_w, width)

        start_h = max(0, (height - center_h) // 2)
        start_w = max(0, (width - center_w) // 2)
        end_h = min(height, start_h + center_h)
        end_w = min(width, start_w + center_w)
        mask[:, start_h:end_h, start_w:end_w] = True
        count = (end_h - start_h) * (end_w - start_w)

        center_row = height // 2
        center_col = width // 2
        if not mask[0, center_row, center_col]:
            mask[:, center_row, center_col] = True
            count += 1
        return count

    def _get_permutation(
        self,
        shape: tuple[int, ...],
        seed: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a fixed permutation for the entire diffusion process.

        This ensures nested sampling: mask(t+1) is a subset of mask(t).
        """
        if seed is None:
            # If no seed, we can't guarantee nesting across calls if the object is recreated.
            # But within one object, we could cache it.
            # For now, if no seed, use a deterministic one based on shape to ensure consistency?
            # Or just use a random one but fixed for this instance?
            # The user blueprint implies passing self.seed.
            seed = 0  # Default to 0 if None to ensure stability? Or raise warning?

        gen = torch.Generator(device="cpu")  # Generator must be on CPU for randperm
        gen.manual_seed(seed)  # Fixed seed, independent of timestep
        # Generate permutation on CPU then move to target device
        return torch.randperm(int(np.prod(shape)), generator=gen).to(device)

    def _ensure_sample_budget(
        self,
        mask: torch.Tensor,
        desired_samples: int,
        line_axis: Literal["y", "x"] | None = None,
    ) -> None:
        """Enforce exact sample budget using nested-safe priority ranking.

        [NESTING FIX] Uses a fixed global priority ranking over ALL indices
        (not just sampled/unsampled) so that additions always pick the
        highest-priority unsampled items and removals always drop the
        lowest-priority sampled items. This ensures monotonic budgets
        across timesteps, preserving M_{t+1} ⊂ M_t.
        """
        # Geometric trajectories and the partial-Fourier block define their own
        # support; topping them up would place samples off-trajectory.
        if not self.enforces_sample_budget:
            return

        if desired_samples <= 0:
            return

        _, height, width = mask.shape
        total_points = height * width
        desired_samples = max(0, min(desired_samples, total_points))
        current_samples = int(torch.sum(mask[0]).item())
        sample_diff = desired_samples - current_samples

        if sample_diff == 0:
            return

        # Use fixed permutation for budget adjustment to ensure nesting/stability
        budget_seed = (self.seed if self.seed is not None else 0) + 12345

        if line_axis is not None:
            # Cartesian line-based budget adjustment; operate on whole lines
            axis_y = line_axis == "y"
            num_lines = height if axis_y else width
            points_per_line = width if axis_y else height
            if num_lines == 0 or points_per_line == 0:
                return

            sampled_lines = torch.any(mask[0], dim=1 if axis_y else 0)

            # [NESTING FIX] Generate a fixed global priority ranking over ALL lines.
            # This ranking is invariant across timesteps, so additions/removals
            # always follow the same order → monotonic budget → nesting preserved.
            gen = torch.Generator(device=mask.device)
            gen.manual_seed(budget_seed)
            global_line_priority = torch.randperm(num_lines, generator=gen, device=mask.device)

            if sample_diff > 0:
                # Add lines: pick highest-priority (lowest rank) unsampled lines
                num_lines_to_add = int(math.ceil(sample_diff / points_per_line))
                # Filter to unsampled lines, keeping priority order
                unsampled_mask = ~sampled_lines[global_line_priority]
                unsampled_in_priority = global_line_priority[unsampled_mask]

                if unsampled_in_priority.numel() == 0:
                    return

                num_to_add = min(num_lines_to_add, unsampled_in_priority.numel())
                chosen = unsampled_in_priority[:num_to_add]

                if axis_y:
                    mask[:, chosen, :] = True
                else:
                    mask[:, :, chosen] = True
            else:
                # Remove lines: drop lowest-priority (highest rank) sampled lines
                num_lines_to_remove = int(math.ceil(-sample_diff / points_per_line))
                sampled_mask = sampled_lines[global_line_priority]
                sampled_in_priority = global_line_priority[sampled_mask]

                if sampled_in_priority.numel() <= num_lines_to_remove:
                    return

                # Remove from the END of the priority list (lowest priority)
                chosen = sampled_in_priority[-num_lines_to_remove:]

                if axis_y:
                    mask[:, chosen, :] = False
                else:
                    mask[:, :, chosen] = False
        else:
            # Point-based budget adjustment (non-geometric, non-line-based)
            # [NESTING FIX] Generate a fixed global priority ranking over ALL points.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(budget_seed)
            global_point_priority = torch.randperm(total_points, generator=gen).to(mask.device)

            if sample_diff > 0:
                flat = mask[0].view(-1)
                # Filter to unsampled points in priority order
                unsampled_mask = ~flat[global_point_priority]
                unsampled_in_priority = global_point_priority[unsampled_mask]

                if unsampled_in_priority.numel() == 0:
                    return

                count = min(sample_diff, unsampled_in_priority.numel())
                selected = unsampled_in_priority[:count]

                rows = selected // width
                cols = selected % width
                mask[:, rows, cols] = True

            elif sample_diff < 0:
                num_to_remove = -sample_diff
                flat = mask[0].view(-1)
                # Filter to sampled points in priority order
                sampled_mask = flat[global_point_priority]
                sampled_in_priority = global_point_priority[sampled_mask]

                if sampled_in_priority.numel() <= num_to_remove:
                    return

                # Remove from the END (lowest priority)
                selected_to_remove = sampled_in_priority[-num_to_remove:]
                rows = selected_to_remove // width
                cols = selected_to_remove % width
                mask[:, rows, cols] = False

        # Always ensure the k-space center remains sampled
        center_row = height // 2
        center_col = width // 2
        if height > 0 and width > 0:
            mask[:, center_row, center_col] = True


class UniformCartesianKSpaceAccelerator(KSpaceAccelerator):
    """Structured Cartesian sampling with uniform, periodic undersampling.

    Corresponds to the 'Structured Cartesian' category. Samples every N-th line
    along a specified axis to achieve the target acceleration.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        base_acceleration: float = 1.0,
        center_fraction: float = 0.0325,
        line_axis: Literal["y", "x"] = "y",
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            base_acceleration (float): Description.
            center_fraction (float): Description.
            line_axis (Literal['y', 'x']): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        if line_axis not in ("y", "x"):
            msg = "line_axis must be 'y' or 'x'"
            raise ValueError(msg)
        self.base_acceleration = max(1.0, base_acceleration)
        self.center_fraction = center_fraction
        self.line_axis = line_axis
        self.seed = seed

    # Removed redundant get_acceleration_factor override

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a uniform Cartesian mask with nested sampling."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        # Determine current center fraction dynamically
        progress = self._normalized_progress(timestep)
        current_center_fraction = self.center_fraction
        if self.min_center_fraction is not None and self.min_center_fraction < self.center_fraction:
            val_range = self.center_fraction - self.min_center_fraction
            scaled_progress = progress**self.schedule_power
            current_center_fraction = self.center_fraction - (scaled_progress * val_range)

        # Apply center patch first
        num_lines = height if self.line_axis == "y" else width
        num_center_lines = max(1, int(round(num_lines * current_center_fraction)))
        start_center = (num_lines - num_center_lines) // 2
        end_center = start_center + num_center_lines

        if self.line_axis == "y":
            mask[:, start_center:end_center, :] = True
            sampled_lines = end_center - start_center
        else:
            mask[:, :, start_center:end_center] = True
            sampled_lines = end_center - start_center

        desired_fraction = self._target_sampling_fraction(timestep)

        # Per-batch trace of the chosen sampling fraction. Was previously
        # logged at INFO and flooded the log; demoted to debug so it's
        # opt-in via LOGLEVEL=DEBUG.
        logger.debug(
            f"UniCartesian: t={timestep} | Desired Frac: {desired_fraction:.4f} "
            f"| Base: {self.base_acceleration} -> Max: {self.max_acceleration}"
        )

        effective_fraction = max(desired_fraction, current_center_fraction)
        desired_lines = max(1, int(round(num_lines * effective_fraction)))
        remaining_lines = desired_lines - sampled_lines

        if remaining_lines > 0:
            # Create a set of already sampled center lines
            center_line_indices = set(range(start_center, end_center))

            # Create a pool of available lines for undersampling
            available_indices = [i for i in range(num_lines) if i not in center_line_indices]

            if len(available_indices) > 0:
                # Use fixed permutation for nested sampling of lines
                # We use a derived seed for line permutation
                line_seed = (self.seed if self.seed is not None else 0) + 54321
                gen = torch.Generator(device=device)
                gen.manual_seed(line_seed)

                # Permute available indices to establish a fixed priority order
                perm = torch.randperm(len(available_indices), generator=gen, device=device)

                # Select top N lines based on the fixed order
                num_to_select = min(remaining_lines, len(available_indices))
                # A tensor gather: iterating ``perm`` read one index back to the
                # host per selected line, a sync each on a CUDA mask (#2573).
                pool = torch.tensor(available_indices, dtype=torch.long, device=device)
                selected_indices = pool[perm[:num_to_select]]

                if self.line_axis == "y":
                    mask[:, selected_indices, :] = True
                else:
                    mask[:, :, selected_indices] = True

        # Ensure the final budget is met
        total_points = height * width
        desired_samples = max(1, int(round(total_points * effective_fraction)))
        self._ensure_sample_budget(
            mask, desired_samples, line_axis=cast(Literal["y", "x"], self.line_axis)
        )

        return mask


class PartialFourierKSpaceAccelerator(KSpaceAccelerator):
    """Fractional Fourier (Half Scan) sampling.

    Corresponds to the 'Fractional Fourier' category. Samples a contiguous
    block of k-space, leveraging Hermitian symmetry for reconstruction.
    This implementation is deterministic and does not use a random seed.
    """

    #: The trajectory defines the support; see KSpaceAccelerator.
    enforces_sample_budget = False

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = 2.0,  # PF is typically < 2x
        center_fraction: float = 0.0325,
        pf_fraction: float = 0.75,
        line_axis: Literal["y", "x"] = "y",
        **kwargs: Any,
    ):
        # Partial Fourier has a theoretical max acceleration of 2.0
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            center_fraction (float): Description.
            pf_fraction (float): Description.
            line_axis (Literal['y', 'x']): Description.
        """
        super().__init__(num_timesteps, min(max_acceleration, 2.0), **kwargs)
        if not 0.5 <= pf_fraction <= 1.0:
            msg = "Partial Fourier fraction (pf_fraction) must be between 0.5 and 1.0"
            raise ValueError(msg)
        if line_axis not in ("y", "x"):
            msg = "line_axis must be 'y' or 'x'"
            raise ValueError(msg)
        self.center_fraction = center_fraction  # Kept for API consistency
        self.pf_fraction = pf_fraction
        self.line_axis = line_axis
        # Partial Fourier is a genuinely FIXED acquisition -- a contiguous ~5/8
        # block recovered by Hermitian symmetry -- so it has no ladder to
        # traverse, and a declared one IS discarded.
        #
        # Enforcement deliberately lives in the ladder gate, not in a raise
        # here. `_acceleration_range_explicit` cannot carry the decision:
        # `resolve_undersampling_kwargs` supplies `acceleration_range` from the
        # schema default on EVERY config-driven construction, so the flag means
        # "present in kwargs", not "the user asked for it", and raising on it
        # would reject the ordinary case. What made this undetectable was the
        # dishonest `R_nominal` below, not the missing raise -- see
        # `get_acceleration_factor` (#1160).

    @property
    def realised_acceleration(self) -> float:
        """The one acceleration this pattern actually realises, at every timestep."""
        return 1.0 / self.pf_fraction

    def get_acceleration_factor(self, timestep: int) -> float:
        """The DECLARED acceleration at ``timestep`` -- not the realised one.

        This used to return ``1 / pf_fraction`` at every timestep, which looks
        like honesty (it is what the mask realises) but is what made the pattern
        invisible to its own gate. ``describe_ladder`` reports
        ``(t, R_nominal, R_effective, bins)`` with ``R_nominal`` taken from
        here, so overwriting it produced ``R_nominal == R_effective`` at every
        rung; ``declared_ladder_defects`` saw zero drift and
        ``scripts/ci/check_acceleration_ladder_realisable.py`` passed an arm
        whose whole ladder had been discarded (#1160, pitfall #16).

        Reporting the declared rung makes the divergence measurable: nominal is
        what the config asked for, effective is the ~1.33x this actually
        delivers, and the gate's drift check finally has two different numbers
        to compare. The mask is unaffected --
        :meth:`get_acceleration_mask` reads ``pf_fraction`` directly and never
        routed through this method. Use :attr:`realised_acceleration` for the
        constant this pattern truly delivers.
        """
        return super().get_acceleration_factor(timestep)

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a partial Fourier mask."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        if self.line_axis == "y":
            num_lines = height
            num_sampled_lines = int(round(num_lines * self.pf_fraction))
            start_line = (num_lines - num_sampled_lines) // 2
            end_line = start_line + num_sampled_lines
            mask[:, start_line:end_line, :] = True
        else:  # self.line_axis == "x"
            num_cols = width
            num_sampled_cols = int(round(num_cols * self.pf_fraction))
            start_col = (num_cols - num_sampled_cols) // 2
            end_col = start_col + num_sampled_cols
            mask[:, :, start_col:end_col] = True

        # Also apply a standard center patch to ensure it's always there
        self._apply_center_patch(mask, self.center_fraction)

        return mask


class VDCartesian1DAccelerator(KSpaceAccelerator):
    """Variable-density 1D Cartesian sampling.

    Corresponds to the 'Variable-Density Cartesian' category. Samples lines
    along a single axis with a probability distribution that is highest at the
    center and falls off towards the edges.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        center_fraction: float = 0.0325,
        line_axis: Literal["y", "x"] = "y",
        density_power: float = 2.0,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            center_fraction (float): Description.
            line_axis (Literal['y', 'x']): Description.
            density_power (float): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        if line_axis not in ("y", "x"):
            raise ValueError("line_axis must be 'y' or 'x'")
        self.center_fraction = center_fraction
        self.line_axis = line_axis
        self.density_power = density_power
        self.seed = seed

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a 1D variable-density Cartesian mask with nested sampling."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        desired_fraction = self._target_sampling_fraction(timestep)

        # The ACS shrinks with the budget when a floor was declared; held at its
        # nominal width it clamps this family's ladder at ~1/center_fraction
        # (12.2x for the exp_11 arms, which declare rungs to 32x). See
        # ``_current_center_fraction``.
        current_center_fraction = self._current_center_fraction(timestep, self.center_fraction)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"VarDensity: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        effective_fraction = max(desired_fraction, current_center_fraction)
        axis_y = self.line_axis == "y"
        num_lines = height if axis_y else width
        desired_lines = max(1, int(round(num_lines * effective_fraction)))

        # ONE fixed line ranking, taken top-K at K = ``desired_lines``.
        #
        # The ACS band spans whole lines along the SAME axis the undersampling
        # runs on, so its width is measured in ``num_lines`` -- not in the other
        # dimension, which is what made the band a different size from the lines
        # it was supposed to sit among.
        #
        # Ranking rather than "paint the ACS, then top up" is what lets the ACS
        # shrink WITHOUT breaking nesting. Painting the band first makes the
        # peripheral count ``desired_lines - center_lines``, and once the band
        # shrinks faster than the budget that difference GROWS with t -- lines
        # re-enter the mask at higher acceleration, which cold diffusion's
        # forward process has no way to undo (measured: 3328 re-added samples
        # over a 28-step exp_11 cascade). ``desired_lines`` is monotone in t, so
        # a top-K of a timestep-invariant ranking is nested by construction.
        #
        # The ranking is ACS-first (centre-out, so the band truncates toward DC
        # rather than from one end), then the variable-density permutation over
        # the remaining lines. While the ACS does not shrink -- every arm in the
        # corpus today -- K is always at least the band width, so the realised
        # set is the whole band plus the same peripheral prefix as before: masks
        # are byte-identical.
        nominal_center_lines = max(1, int(round(num_lines * self.center_fraction)))
        nominal_start = (num_lines - nominal_center_lines) // 2
        acs_lines = torch.arange(nominal_start, nominal_start + nominal_center_lines, device=device)
        # Anchored on the DC line (``num_lines // 2``, the centred-FFT
        # convention), NOT on the band's own midpoint: for an even line count
        # those differ by half a line, and ranking by the midpoint puts DC
        # second, so the tightest rungs drop it -- the #581 failure mode in a
        # different family.
        acs_ranked = acs_lines[torch.argsort((acs_lines - num_lines // 2).abs(), stable=True)]

        # [REFACTOR] Use shared logic from MaskGenerator
        pdf_np = MaskGenerator.get_variable_density_pdf(num_lines, self.density_power)
        prob_dist = torch.from_numpy(pdf_np).to(device=device, dtype=torch.float32)
        prob_dist[nominal_start : nominal_start + nominal_center_lines] = 0

        ranking = acs_ranked
        num_available = int((prob_dist > 0).sum().item())
        if num_available > 0:
            # Use fixed seed for nested sampling: a weighted permutation of every
            # non-ACS line, established once and truncated per timestep.
            line_seed = (self.seed if self.seed is not None else 0) + 67890
            gen = torch.Generator(device=device)
            gen.manual_seed(line_seed)
            perm_indices = torch.multinomial(
                prob_dist / prob_dist.sum(),
                num_available,
                replacement=False,
                generator=gen,
            )
            ranking = torch.cat([acs_ranked, perm_indices])

        chosen_indices = ranking[: min(desired_lines, ranking.numel())]
        if axis_y:
            mask[:, chosen_indices, :] = True
        else:
            mask[:, :, chosen_indices] = True

        # Ensure final budget is met across the entire 2D k-space
        total_points = height * width
        desired_samples = max(1, int(round(total_points * effective_fraction)))
        self._ensure_sample_budget(
            mask,
            desired_samples,
            line_axis=cast(Literal["y", "x"], self.line_axis),
        )

        return mask


class VDCartesian2DGaussianAccelerator(KSpaceAccelerator):
    """Variable-density 2D Cartesian sampling with a Gaussian distribution."""

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        center_fraction: float = 0.0325,
        sigma_scaling: float = 0.25,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            center_fraction (float): Description.
            sigma_scaling (float): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.center_fraction = center_fraction
        self.sigma_scaling = sigma_scaling
        self.seed = seed

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a 2D Gaussian-sampled Cartesian mask with nested sampling."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        total_points = height * width
        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"Gaussian2D: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        # See ``_current_center_fraction``: a nominal-width ACS is the entire
        # budget past R=1/center_fraction, which clamped this family at 12.5x
        # while its arms declared rungs to 32x (#534).
        current_center_fraction = self._current_center_fraction(timestep, self.center_fraction)

        effective_fraction = max(desired_fraction, current_center_fraction)
        desired_samples = max(1, int(round(total_points * effective_fraction)))

        # ONE fixed point ranking, taken top-K at K = ``desired_samples`` --
        # the same restructure as VDCartesian1D and for the same reason. Painting
        # the ACS first and topping up to the budget makes the peripheral count
        # ``desired_samples - sampled``, which GROWS with t once the band shrinks
        # faster than the budget, so points re-enter the mask at higher
        # acceleration (measured: 3001 re-added over a 28-step exp_11 cascade,
        # and unlike the 1D family none of it was the budget top-up). Ranking
        # ACS-first-by-radius then the Gaussian permutation is monotone in K, so
        # nesting holds by construction and the ACS truncates toward DC.
        # Create 2D Gaussian probability distribution
        # y = torch.linspace(-1.0, 1.0, height, device=device)
        # x = torch.linspace(-1.0, 1.0, width, device=device)
        # grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Use a fixed sigma for the ranking to ensure nesting
        # We use the sigma corresponding to the middle of the process or a fixed value
        # to define the "ideal" importance of each pixel.
        # Using time-dependent sigma for ranking would violate nesting.
        fixed_sigma = self.sigma_scaling * 0.5 + 0.01
        # prob_map = torch.exp(-(grid_y**2 + grid_x**2) / (2 * fixed_sigma**2))

        # [REFACTOR] Use shared logic from MaskGenerator for consistency
        # MaskGenerator defines std_scale = 1/sigma.
        # Here fixed_sigma is sigma.
        # So std_scale = 1/fixed_sigma
        std_scale = 1.0 / fixed_sigma

        # We need 2D PDF. MaskGenerator provides 1D logic for now (mostly).
        # But wait, MaskGenerator.get_gaussian_pdf is 1D? Yes.
        # _generate_gaussian in MaskGenerator was using 1D logic?
        # Let's check. Ah, MaskGenerator._generate_gaussian creates a 1D PDF and applies it to lines (broadcasts).
        # But VDCartesian2DGaussianAccelerator here seems to be doing full 2D point sampling?
        # "grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')"
        # "prob_map = torch.exp(-(grid_y**2 + grid_x**2) / (2 * fixed_sigma**2))"
        # Yes, this is 2D point sampling.

        # However, a radial 2D gaussian PDF is just outer product of two 1D gaussians (if separable)?
        # exp(-(y^2 + x^2)) = exp(-y^2) * exp(-x^2)
        # Yes. So we can use the 1D function and outer product it.

        pdf_y = MaskGenerator.get_gaussian_pdf(height, std_scale)
        pdf_x = MaskGenerator.get_gaussian_pdf(width, std_scale)

        prob_map = torch.from_numpy(pdf_y[:, None] * pdf_x[None, :]).to(
            device=device, dtype=torch.float32
        )

        # The candidate set is keyed to the NOMINAL ACS, so the Gaussian
        # permutation below stays timestep-invariant even while the realised
        # band shrinks.
        prob_map_flat = prob_map.flatten()
        center_mask_for_prob = torch.zeros_like(prob_map).unsqueeze(0)
        self._apply_center_patch(center_mask_for_prob, self.center_fraction)
        acs_flat = center_mask_for_prob.flatten() > 0
        prob_map_flat[acs_flat] = 0

        # ACS points first, ordered centre-out by radius so a budget too small
        # for the whole band keeps the innermost part of it (DC included) rather
        # than an arbitrary raster prefix.
        rows = torch.arange(height, device=device, dtype=torch.float32) - height // 2
        cols = torch.arange(width, device=device, dtype=torch.float32) - width // 2
        radius_flat = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2).flatten()
        acs_indices = torch.nonzero(acs_flat, as_tuple=False).flatten()
        ranking = acs_indices[torch.argsort(radius_flat[acs_indices], stable=True)]

        num_available = int((prob_map_flat > 0).sum().item())
        if num_available > 0:
            # Use fixed seed for nested sampling
            gauss_seed = (self.seed if self.seed is not None else 0) + 13579
            gen = torch.Generator(device=device)
            gen.manual_seed(gauss_seed)
            perm = torch.multinomial(
                prob_map_flat / prob_map_flat.sum(),
                num_samples=num_available,
                replacement=False,
                generator=gen,
            )
            ranking = torch.cat([ranking, perm])

        mask.view(-1)[ranking[: min(desired_samples, ranking.numel())]] = True

        self._ensure_sample_budget(mask, desired_samples)
        return mask


class VDCartesianCAVAAccelerator(KSpaceAccelerator):
    """CAVA-style variable-density Cartesian sampling.

    Corresponds to the 'Variable-Density Cartesian (CAVA)' category."""

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        center_fraction: float = 0.0325,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            center_fraction (float): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.center_fraction = center_fraction
        self.seed = seed
        # CAVA-specific parameters would go here

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generates a CAVA-style exponential density mask."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"CAVA: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        total_points = height * width
        desired_samples = max(1, int(round(total_points * desired_fraction)))

        # Center patch
        sampled = self._apply_center_patch(mask, self.center_fraction)
        remaining = desired_samples - sampled

        if remaining > 0:
            # Create probability map (Exponential decay for CAVA-style)
            y_coords = torch.linspace(-1.0, 1.0, height, device=device)
            x_coords = torch.linspace(-1.0, 1.0, width, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

            radius = torch.sqrt(grid_y**2 + grid_x**2)

            # CAVA inspired profile: Exponential decay
            # P(r) ~ exp(-alpha * r)
            # alpha determines sharpness. Higher timestep -> potentially sharper decay?
            # Sticking to fixed slope for ensuring nesting if possible.
            slope = 5.0
            prob_map = torch.exp(-slope * radius)

            # Zero out already sampled center
            prob_map[mask[0]] = 0

            # Normalize
            if prob_map.sum() > 0:
                prob_map /= prob_map.sum()

            # Use fixed seed generator to ensure nesting property
            # Ideally we pick top-K from a fixed ranking
            cava_seed = (self.seed if self.seed is not None else 0) + 44444
            gen = torch.Generator(device=device)
            gen.manual_seed(cava_seed)

            # Add small jitter for tie-breaking
            jitter = torch.rand(prob_map.shape, generator=gen, device=device) * 1e-7
            scores = prob_map + jitter

            # Select top-K highest scores
            flat_scores = scores.view(-1)
            # Only consider pixels not yet sampled (prob > 0 or consistent logic)
            # Actually simplest is to rank ALL pixels and take top N until budget is met
            # Mask[0] is already set for center. The center pixels likely have high prob,
            # but we zeroed them. So we pick from the rest.

            _, indices = torch.sort(flat_scores, descending=True)

            # Filter indices to those not in mask?
            # Since we zeroed center probs, they will be at bottom of list?
            # No, exp(-0) is 1. We set them to 0. So they are at bottom.
            # So top scores are the ones just outside center.

            selected_indices = indices[:remaining]

            # Map back to 2D
            rows = selected_indices // width
            cols = selected_indices % width

            mask[:, rows, cols] = True

        self._ensure_sample_budget(mask, desired_samples)
        return mask


class LearnedPatternAccelerator(KSpaceAccelerator):
    """Learned k-space sampling pattern.

    Corresponds to the 'Learned Pattern' category.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        pattern_path: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            pattern_path (str | None): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.pattern_path = pattern_path
        self.seed = seed
        # In a real implementation, we would load and store the learned pattern
        # self.learned_pattern = self._load_pattern(pattern_path)

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generates a mask from a learned pattern or fallback."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"LearnedPattern: t={timestep}")

        # Try to load pattern if path exists
        if self.pattern_path and os.path.exists(self.pattern_path):
            try:
                # Expecting a tensor file or numpy file
                if self.pattern_path.endswith(".pt") or self.pattern_path.endswith(".pth"):
                    loaded_mask = torch.load(self.pattern_path, map_location=device)
                elif self.pattern_path.endswith(".npy"):
                    loaded_np = np.load(self.pattern_path)
                    loaded_mask = torch.from_numpy(loaded_np).to(device)
                else:
                    raise ValueError("Unsupported pattern format")

                # Handle shape mismatch (crop or pad or interpolate)
                # Simple resize via interpolation if float, or just crop/pad
                if loaded_mask.shape[-2:] != (height, width):
                    # For boolean mask, nearest interpolation
                    loaded_mask = (
                        torch.nn.functional.interpolate(
                            loaded_mask.float().unsqueeze(0),
                            size=(height, width),
                            mode="nearest",
                        )
                        .squeeze(0)
                        .bool()
                    )

                # Assign
                mask[:] = loaded_mask

                # Check budget? Learned patterns usually have fixed acceleration.
                # If we need dynamic budget, we might need to prioritize pixels.
                # Assuming learned pattern is for specific acceleration.
                return mask

            except Exception as e:
                # A learned_pattern was explicitly configured AND the file
                # exists, but loading it failed (bad format / corrupt tensor /
                # un-interpolable shape / programming error). Per CLAUDE.md
                # non-negotiable #3/#4 and pitfalls #9/#10, an advertised
                # sampling scheme that cannot be honoured must RAISE rather than
                # silently degrade to variable-density sampling.
                logger.error(f"Failed to load learned pattern from {self.pattern_path}: {e}")
                raise RuntimeError(
                    "LearnedPatternAccelerator: failed to load configured pattern "
                    f"from {self.pattern_path}: {e}"
                ) from e

        # Fallback: Variable Density Sampling
        desired_fraction = self._target_sampling_fraction(timestep)
        total_points = height * width
        desired_samples = max(1, int(round(total_points * desired_fraction)))

        # Standard VDS logic
        self._apply_center_patch(mask, max(0.04, desired_fraction * 0.1))

        # Random filling for remainder lines (fallback for under-specified patterns)
        current_samples = mask.sum().item()
        remaining = desired_samples - current_samples
        if remaining > 0:
            flat_mask = mask[0].flatten()
            unsampled_indices = torch.nonzero(~flat_mask, as_tuple=False).view(-1)

            if unsampled_indices.numel() > 0:
                gen = torch.Generator(device=device)
                if self.seed:
                    gen.manual_seed(self.seed)  # [NESTING FIX] Fixed seed, not seed + timestep

                perm = torch.randperm(unsampled_indices.numel(), generator=gen, device=device)
                chosen = unsampled_indices[perm[:remaining]]

                mask.view(-1)[chosen] = True

        self._ensure_sample_budget(mask, desired_samples)
        return mask


class LowPassAccelerator(KSpaceAccelerator):
    """Low-pass filter acceleration (Central Crop).

    Used for super-resolution learning. The mask is always a central region,
    shrinking as timestep increases (degradation increases).
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        center_fraction: float = 0.0325,  # Minimum center fraction at max degradation
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            center_fraction (float): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.center_fraction = center_fraction

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a low-pass (central crop) mask."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        # ~1.0 at t=0, ~1/max_acceleration at t=T-1 (never below center_fraction).

        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"LowPass: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        # The floor honours ``min_center_fraction`` when one was declared. Without
        # it the disc bottomed out at ``center_fraction``, pinning the realised
        # acceleration at ``1/center_fraction`` no matter what the ladder asked
        # for -- so this class's own docstring promise that the region keeps
        # "shrinking as timestep increases" stopped being true partway up, and on
        # the 28-rung cohort ladder 9 of 28 levels were a plateau (#1159).
        #
        # ``_guaranteed_core_fraction`` (a single constant floor) rather than
        # ``_current_center_fraction`` (a per-timestep shrink): the parameter is
        # documented here as "minimum center fraction at max degradation", i.e. it
        # already plays the floor role, and concentric discs nest whatever the
        # floor is. Returns ``center_fraction`` unchanged when no floor was
        # declared, so existing arms keep byte-identical masks.
        effective_fraction = max(
            desired_fraction, self._guaranteed_core_fraction(self.center_fraction)
        )

        # Apply central crop
        self._apply_center_patch(mask, effective_fraction)

        return mask


class LinearKSpaceAccelerator(KSpaceAccelerator):
    """Linear acceleration pattern sampling more k-space points over time."""

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        base_acceleration: float = 1.0,
        center_fraction: float = 0.0325,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            base_acceleration (float): Description.
            center_fraction (float): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.base_acceleration = max(1.0, base_acceleration)
        self.center_fraction = center_fraction
        self.seed = seed

        # [DEBUG] Verify if configuration is passing acceleration=1.0
        if self.max_acceleration <= 1.01 and self.base_acceleration <= 1.01:
            logger.debug(
                f"⚠️ [LinearAccelerator] WARNING: Max Acceleration is {self.max_acceleration}! Training will be Identity Mapping (Clean->Clean). Check config!"
            )
        else:
            logger.debug(
                f"✅ [LinearAccelerator] Initialized with Range: [{self.base_acceleration} -> {self.max_acceleration}]"
            )

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate linear acceleration mask tracking desired sampling with nested sampling."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros(
            (channels, height, width),
            dtype=torch.bool,
            device=device,
        )

        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"Linear: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        total_points = height * width
        desired_samples = max(1, int(round(total_points * desired_fraction)))

        # `_guaranteed_core_fraction`, not `_current_center_fraction`: the local
        # permutation below is taken over the ACS *complement* and is only nested
        # while that complement stays FIXED across the cascade (see the comment
        # where it is built). A per-timestep shrink would reindex it and trade
        # this class's ceiling defect for a nesting defect. One constant core,
        # sized at the highest rung, lifts the ceiling and keeps the complement
        # fixed. Unchanged when no `min_center_fraction` was declared (#1159).
        sampled = self._apply_center_patch(
            mask, self._guaranteed_core_fraction(self.center_fraction)
        )
        remaining = desired_samples - sampled

        if remaining > 0:
            # Use _get_permutation for nested random sampling of remaining points
            # We need to exclude the center patch from the permutation or just mask it out

            # Simple approach: Get permutation of ALL points, filter out those already sampled, then pick top N
            # This is robust.

            perm = self._get_permutation(mask.shape, self.seed, device)

            # Identify unsampled indices
            flat_mask = mask[0].view(-1)
            # We want to pick `remaining` indices from `perm` that are NOT in `flat_mask`

            # Efficient way:
            # 1. Get indices from perm
            # 2. Check if they are already sampled
            # 3. Take first `remaining` that are not sampled

            # Since center patch is small, we can just iterate or mask
            # Or better:
            # Get indices of unsampled points
            unsampled_indices = torch.nonzero(~flat_mask, as_tuple=False).view(-1)

            if unsampled_indices.numel() > 0:
                # We need a fixed order for these unsampled indices
                # We can use the global permutation to induce an order on them
                # i.e. sort unsampled_indices based on their position in perm?
                # Or just permute them using a fixed seed?
                # If we permute them using a fixed seed, it's nested relative to the set of unsampled points.
                # Since the set of unsampled points (center patch complement) is FIXED (center fraction is const),
                # we can just permute the unsampled points once.

                linear_seed = (self.seed if self.seed is not None else 0) + 99999
                gen = torch.Generator(device="cpu")  # Generator must be on CPU for randperm
                gen.manual_seed(linear_seed)

                local_perm = torch.randperm(unsampled_indices.numel(), generator=gen).to(device)

                num_to_select = min(remaining, unsampled_indices.numel())
                chosen_indices = unsampled_indices[local_perm[:num_to_select]]

                mask.view(-1)[chosen_indices] = True

        return mask


def _nested_trajectory_mask(
    accelerator: KSpaceAccelerator,
    ordering: Callable[[int, int], tuple[torch.Tensor, int]],
    kspace_shape: tuple[int, ...],
    timestep: int,
    device: torch.device,
) -> torch.Tensor:
    """The trajectory families' mask: the identity at ``R <= 1``, else a ladder prefix.

    One owner for radial and spiral, which differ only in their sample ordering
    (see :mod:`~spectramr.infrastructure.physics.trajectory_ladder`). The ACS core is
    fixed for the whole cascade at the size the tightest rung can afford, so the core
    never changes between rungs and the prefix length decides nesting alone.
    """
    channels, height, width = accelerator._unpack_shape(kspace_shape)
    desired_fraction = accelerator._target_sampling_fraction(timestep)
    if desired_fraction >= 1.0:
        return torch.ones((channels, height, width), dtype=torch.bool, device=device)

    ladders: dict[tuple[int, int], NestedTrajectoryLadder] = accelerator._ladders  # type: ignore[attr-defined]
    ladder = ladders.get((height, width))
    if ladder is None:
        traj, quantum = ordering(height, width)
        rows, cols = MaskGenerator.trajectory_bin_indices(traj, height, width)
        core = torch.zeros((1, height, width), dtype=torch.bool)
        accelerator._apply_center_patch(core, accelerator._core_fraction)  # type: ignore[attr-defined]
        family = type(accelerator).__name__
        ladder = NestedTrajectoryLadder(rows, cols, height, width, quantum, core[0], family)
        ladders[(height, width)] = ladder

    # The prefix length is a pure function of (grid, timestep); memoised so the
    # search's host sync runs once per rung rather than once per batch element.
    counts: dict[tuple[int, int, int], int] = accelerator._sample_counts  # type: ignore[attr-defined]
    key = (height, width, int(timestep))
    samples = counts.get(key)
    if samples is None:
        samples = ladder.sample_count(desired_fraction)
        counts[key] = samples
    return accelerator._as_channelled(ladder.mask(samples).to(device), channels)


def _fixed_trajectory_core(accelerator: KSpaceAccelerator, center_fraction: float) -> float:
    """The ACS fraction a trajectory family keeps at every rung below the identity.

    ``_guaranteed_core_fraction`` capped by the tightest declared budget: the class
    default ``center_fraction`` (0.0325) exceeds ``1/R`` above R~30, and a core that
    shrank per rung would let the prefix grow between adjacent rungs.
    """
    tightest = min(
        1.0 / max(1.0, accelerator.get_acceleration_factor(t))
        for t in range(accelerator.num_timesteps)
    )
    return min(accelerator._guaranteed_core_fraction(center_fraction), tightest)


class RadialKSpaceAccelerator(KSpaceAccelerator):
    """Golden-angle radial sampling: each rung keeps the first k spokes of one ordering.

    Spokes advance by the radial golden angle, pi*(sqrt(5) - 1)/2 ~= 111.25 deg
    (Winkelmann 2007), so the first ``k`` spokes are the same set for every rung and
    the cascade nests (#612). ``k`` is derived per rung as the fewest whole spokes
    that, with the ACS core, reach the rung's budget; there is no spoke floor. Serves
    both the ``radial`` and the ``golden_angle`` registry keys.

    The mask approximates the trajectory on the Cartesian grid; true non-Cartesian
    reconstruction (NUFFT) is outside the scope of this accelerator.
    """

    #: The trajectory defines the support; see KSpaceAccelerator.
    enforces_sample_budget = False

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        center_fraction: float = 0.0325,
        **kwargs: Any,
    ):
        """Build the accelerator and refuse a ladder no trajectory can realise.

        Args:
            num_timesteps: Number of rungs.
            max_acceleration: Acceleration at the last rung.
            center_fraction: ACS fraction when no ``min_center_fraction`` floor is
                declared; the core is fixed for the whole cascade.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.center_fraction = max(0.0, min(center_fraction, 0.5))
        validate_ladder_reachable(self, "radial")
        self._core_fraction = _fixed_trajectory_core(self, self.center_fraction)
        self._ladders: dict[tuple[int, int], NestedTrajectoryLadder] = {}
        self._sample_counts: dict[tuple[int, int, int], int] = {}

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """The first ``k(R)`` golden-angle spokes plus the ACS core; all ones at ``R <= 1``."""
        return _nested_trajectory_mask(self, golden_radial_ordering, kspace_shape, timestep, device)


class VariableDensityKSpaceAccelerator(KSpaceAccelerator):
    """Radial variable-density point sampling with a guaranteed centred ACS disc.

    Bins are ranked once by a radially decaying density, with a centred disc of
    ``_guaranteed_core_fraction(center_fraction)`` of k-space placed at the head
    of the ranking; each rung takes the top-K of that ranking, so
    ``M_{t+1} subset-of M_t`` holds exactly. The seeded jitter only breaks ties
    within an equal-radius shell, so the realised mask is close to a disc; a
    weighted random draw is what ``density_nested`` provides.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        density_power: float = 2.0,
        center_fraction: float = 0.0325,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """Initialise the variable-density accelerator.

        Args:
            num_timesteps: Diffusion horizon.
            max_acceleration: Acceleration at the top of the ladder.
            density_power: Exponent of the radial density decay.
            center_fraction: Nominal ACS fraction; the guaranteed core when no
                ``min_center_fraction`` floor is declared.
            seed: Seed for the timestep-invariant tie-breaking jitter.

        Raises:
            ValueError: If a declared ``min_center_fraction`` exceeds the
                sampling budget at ``max_acceleration``.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.density_power = density_power
        self.center_fraction = max(0.0, min(center_fraction, 0.5))
        self.seed = seed
        # Bounded and seed-keyed: dynamic-mask training mutates ``.seed`` per
        # sample on one instance, so a seedless key replayed the first draw for
        # every sample and an unbounded one grew by an entry per sample (#1071).
        self._priority_cache: BoundedLRUCache[tuple[Any, ...], torch.Tensor] = BoundedLRUCache()

    def _priority_cache_key(
        self, channels: int, height: int, width: int, device: torch.device
    ) -> tuple[Any, ...]:
        """Memo key for the ranking: everything it depends on, seed included."""
        return (channels, height, width, str(device), self.seed)

    def _priority_ranking(
        self, channels: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Return the flattened pixel priority order (descending), memoized.

        The ranking is constant across timesteps (the jitter uses a fixed,
        timestep-independent seed), so it is a pure function of shape, device
        and seed.
        """
        cache_key = self._priority_cache_key(channels, height, width, device)
        cached = self._priority_cache.get(cache_key)
        if cached is not None:
            return cached

        # Create probability map (higher in center)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        center_h, center_w = height // 2, width // 2
        distances = torch.sqrt((y_coords - center_h) ** 2 + (x_coords - center_w) ** 2)
        max_distance = torch.sqrt(
            torch.tensor(center_h**2 + center_w**2, device=device),
        )
        probabilities = 1.0 / (1.0 + (distances / (max_distance * 0.3)) ** self.density_power)
        probabilities = probabilities / probabilities.max()

        # [FORENSIC FIX] Fixed jitter (timestep-invariant) so the priority
        # ranking never changes across timesteps — guaranteeing M_t ⊂ M_{t-1}.
        generator = self._make_generator(device, 0)
        if generator is not None:
            jitter = torch.rand(probabilities.shape, generator=generator, device=device) * 1e-6
        else:
            jitter = torch.rand_like(probabilities) * 1e-6

        sorted_indices = torch.argsort((probabilities + jitter).view(-1), descending=True)

        # The ACS disc (equal area to the core fraction, the #581 geometry) goes
        # to the head of the ranking with the density order kept on both sides,
        # so any budget at or above the core fully samples it. The floor's
        # reachability is checked at construction by `_validate_center_floor`.
        core_fraction = self._guaranteed_core_fraction(self.center_fraction)
        core_radius = math.sqrt(max(height * width * core_fraction, 1.0) / math.pi)
        in_core = (distances <= core_radius).view(-1)[sorted_indices]
        sorted_indices = torch.cat((sorted_indices[in_core], sorted_indices[~in_core]))
        self._priority_cache[cache_key] = sorted_indices
        return sorted_indices

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate variable density mask with progressive sampling.

        [FORENSIC FIX] Guarantees Nested Property: M_{t+1} subset of M_t for Cold Diffusion.
        Uses fixed priority scores so that as budget changes, we simply take more/fewer
        of the top-ranked pixels. No random budget adjustment that could break nesting.
        """
        channels, height, width = self._unpack_shape(kspace_shape)

        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"VariableDensity: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        total_points = height * width
        desired_samples = max(1, int(round(total_points * desired_fraction)))

        # Sorted ONCE per shape (memoized) — the ranking is timestep-invariant.
        sorted_indices = self._priority_ranking(channels, height, width, device)

        # [FORENSIC FIX] Select exactly top-K based on current budget
        # This guarantees nesting: budget_t < budget_{t-1} implies M_t ⊂ M_{t-1}
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)
        k = min(desired_samples, total_points)
        if k > 0:
            selected = sorted_indices[:k]
            rows = selected // width
            cols = selected % width
            mask[:, rows, cols] = True

        return mask


class SpiralKSpaceAccelerator(KSpaceAccelerator):
    """Golden-angle spiral sampling: each rung keeps a prefix of one interleave sequence.

    One Archimedean arm at a fixed turn count; interleave ``a`` is that arm rotated by
    ``a`` times the full-circle golden angle (137.5 deg, see
    :data:`~spectramr.infrastructure.physics.trajectories.SPIRAL_GOLDEN_ANGLE`).
    Samples are ordered by (interleave, readout) and a rung may end mid-arm, so every
    rung hits its budget to the bin and the cascade nests (#612).
    """

    #: The trajectory defines the support; see KSpaceAccelerator.
    enforces_sample_budget = False

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        num_turns: float = 4.0,
        samples_per_arm: int = 1024,
        center_fraction: float = 0.0325,
        **kwargs: Any,
    ):
        """Build the accelerator and refuse a ladder no trajectory can realise.

        Args:
            num_timesteps: Number of rungs.
            max_acceleration: Acceleration at the last rung.
            num_turns: Turns of the one arm, fixed for every rung.
            samples_per_arm: Readout samples per interleave.
            center_fraction: ACS fraction when no ``min_center_fraction`` floor is
                declared; the core is fixed for the whole cascade.

        Raises:
            ValueError: On a non-positive ``num_turns`` or ``samples_per_arm``, or a
                rung with ``1 < R < 1 / COVERAGE_CEILING``.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        if num_turns <= 0 or samples_per_arm < 1:
            msg = f"num_turns and samples_per_arm must be positive, got {num_turns}, {samples_per_arm}"
            raise ValueError(msg)
        self.num_turns = float(num_turns)
        self.samples_per_arm = int(samples_per_arm)
        self.center_fraction = max(0.0, min(center_fraction, 0.5))
        validate_ladder_reachable(self, "spiral")
        self._core_fraction = _fixed_trajectory_core(self, self.center_fraction)
        self._ladders: dict[tuple[int, int], NestedTrajectoryLadder] = {}
        self._sample_counts: dict[tuple[int, int, int], int] = {}

    def _ordering(self, height: int, width: int) -> tuple[torch.Tensor, int]:
        return golden_spiral_ordering(
            height, width, num_turns=self.num_turns, samples_per_arm=self.samples_per_arm
        )

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """The first ``n(R)`` spiral samples plus the ACS core; all ones at ``R <= 1``."""
        return _nested_trajectory_mask(self, self._ordering, kspace_shape, timestep, device)


class RandomCartesianKSpaceAccelerator(KSpaceAccelerator):
    """Random point sampling with a graded centred ACS disc and a uniform periphery.

    One timestep-invariant ranking -- a centre-out disc of
    ``_guaranteed_core_fraction(center_fraction)`` of k-space, then i.i.d.
    uniform scores -- is truncated by budget, so ``M_{t+1} subset-of M_t`` holds
    exactly. The peripheral density is flat, not variable-density, and the mask
    selects individual ``(k_y, k_x)`` bins rather than phase-encode lines, so no
    line is fully sampled and the pattern is not acquirable on a 2D Cartesian
    scanner. ``variable_density`` decays radially; ``equispaced`` samples lines.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        min_sampling_rate: float = 0.0,
        max_sampling_rate: float = 1.0,
        center_fraction: float = 0.0325,
        min_center_fraction: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            min_sampling_rate (float): Description.
            max_sampling_rate (float): Description.
            center_fraction: ACS fraction at the LOWEST acceleration -- the radius of
                the graded centre-out priority region.
            min_center_fraction: guaranteed centred-ACS fraction at the HIGHEST
                acceleration. Declared explicitly rather than swallowed by ``**kwargs``
                (issue #581 / pitfall #15). Because the priority scores are graded
                centre-out, the realised ACS is ``min(budget, center_fraction)`` and
                therefore already shrinks with the budget -- this knob asserts a FLOOR
                on that and raises when the floor is unsatisfiable.
            seed (int | None): Description.
        """
        super().__init__(
            num_timesteps, max_acceleration, min_center_fraction=min_center_fraction, **kwargs
        )
        self.min_sampling_rate = max(0.0, min(min_sampling_rate, 1.0))
        self.max_sampling_rate = max(self.min_sampling_rate, min(max_sampling_rate, 1.0))
        self.center_fraction = max(0.0, min(center_fraction, 0.5))
        self.min_center_fraction = (
            self.center_fraction if min_center_fraction is None else float(min_center_fraction)
        )
        self.seed = seed

        if self.min_center_fraction > self.center_fraction:
            msg = (
                f"min_center_fraction ({self.min_center_fraction}) exceeds center_fraction "
                f"({self.center_fraction}). The ACS shrinks as acceleration rises, so the "
                "floor cannot be above the ceiling."
            )
            raise ValueError(msg)


    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate random Cartesian mask with nested sampling.

        [NESTING FIX] Replaced threshold-based random sampling (which broke
        nesting because seed changed per timestep) with deterministic top-K
        selection from fixed priority scores. This guarantees M_{t+1} ⊂ M_t:
        reducing the budget simply removes the lowest-priority points.

        Args:
            kspace_shape (tuple[int, ...]): Description.
            timestep (int): Description.
            device (torch.device): Description.
        Returns:
            torch.Tensor: Description.
        """
        channels, height, width = self._unpack_shape(kspace_shape)

        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"RandomCartesian: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        total_points = height * width
        desired_samples = max(1, int(round(total_points * desired_fraction)))

        # [NESTING FIX] Generate fixed priority scores using timestep-invariant seed.
        # The ranking of pixels never changes across timesteps — reducing the budget
        # simply removes the lowest-priority pixels, guaranteeing nesting.
        generator = self._make_generator(device, timestep)
        random_scores = torch.rand((height, width), device=device, generator=generator)

        # [#581] Graded centre-out ACS priority.
        #
        # A FLAT boost (`scores[acs_box] = 2.0`) makes every ACS bin tie EXACTLY, so
        # `argsort` falls back to flat (row-major) index order and fills the box
        # top-row-first. Once the budget is smaller than the ACS -- i.e.
        # R > 1/center_fraction, R > 12.5 for the exp_11 ladder -- the top-K stops
        # partway down the box and the realised mask is a handful of contiguous
        # horizontal STRIPES whose centroid is above DC, with the DC bin itself
        # UNSAMPLED. Measured on exp_11 attention_none at R=32x: 4 stripes, 0.48% of
        # target k-space energy retained (the equispaced mask kept 98.2%), and the
        # ringing that put every R=32x case at 12-20 dB was that mask's point-spread
        # function.
        #
        # Grading the boost by radius makes the ranking a strict centre-out order: DC is
        # the unique global maximum (always rank 0, therefore always sampled), and the
        # top-K is a centred DISC whose radius grows with the budget. The scores stay
        # timestep-INVARIANT, so the nested property M_{t+1} subset-of M_t that cold
        # diffusion requires still holds exactly.
        #
        # [#1069] The core is sized from `min_center_fraction`, NOT `center_fraction`.
        #
        # An earlier revision of this comment claimed the centre-out grading gave the
        # ACS shrink "for free, which is what min_center_fraction was reaching for".
        # That is half true and the wrong half is load-bearing: the grading does shrink
        # the realised ACS with the budget, but it shrinks it TO the whole budget. The
        # graded band lives in [1.5, 2.0] and every peripheral score in [0, 1) --
        # disjoint ranges -- so the top-K exhausts the entire nominal ACS before taking
        # a single random bin. Once the budget is smaller than the nominal ACS the mask
        # is 100% ACS: a pure low-pass disc. Measured on experiment_11_attention_none
        # (center_fraction 0.08, ladder [2,4,8,10,12,16,32]): Jaccard 0.999 against the
        # ideal disc at R=16 and 0.995 at R=32, acquiring NOTHING beyond 20% / 14% of
        # the Nyquist radius. Eight of its 28 timesteps had no high-frequency data at
        # all, which makes them super-resolution rather than compressed sensing.
        #
        # Sizing the core from the floor fixes it by arithmetic: 2% core against a
        # 3.125% budget at R=32 leaves 1.125% for genuinely random bins, so the
        # aliasing stays incoherent all the way up. `center_fraction` remains the
        # fallback when no floor is declared, so those arms keep byte-identical masks.
        #
        # What this deliberately does NOT do is interpolate the core width per timestep
        # (the `_current_center_fraction` pattern the VD families use). Under a step
        # schedule the budget is CONSTANT across the timesteps of a plateau while an
        # interpolated ACS keeps shrinking, so the peripheral slots grow within the
        # plateau and bins enter M_{t+1} that were absent from M_t. One fixed ranking
        # plus a fixed core cannot do that.
        #
        # Arms wanting a graded central DENSITY rather than a hard core want
        # `density_nested`, which weights the draw radially instead of banding it.
        #
        # The core is a disc of equal AREA to the requested fraction, so the fraction
        # keeps its meaning (fraction of k-space in the core).
        core_fraction = self._guaranteed_core_fraction(self.center_fraction)
        acs_radius = math.sqrt(max(total_points * core_fraction, 1.0) / math.pi)
        rows_grid = torch.arange(height, device=device, dtype=torch.float32) - height // 2
        cols_grid = torch.arange(width, device=device, dtype=torch.float32) - width // 2
        radius = torch.sqrt(rows_grid[:, None] ** 2 + cols_grid[None, :] ** 2)

        # Inside the ACS: 2.0 at DC falling to 1.5 at the rim -- strictly above every
        # random score (which live in [0, 1)), so the ACS is always exhausted before any
        # peripheral bin is taken.
        in_acs = radius <= acs_radius
        graded = 2.0 - 0.5 * (radius / max(acs_radius, 1e-8)).clamp(max=1.0)
        random_scores = torch.where(in_acs, graded, random_scores)

        # Sort ONCE based on fixed scores — ranking is constant across all timesteps
        flat_scores = random_scores.view(-1)
        sorted_indices = torch.argsort(flat_scores, descending=True)

        # Select exactly top-K based on current budget
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)
        k = min(desired_samples, total_points)
        if k > 0:
            selected = sorted_indices[:k]
            rows = selected // width
            cols = selected % width
            mask[:, rows, cols] = True

        return mask


class FractionalVariableDensityAccelerator(VariableDensityKSpaceAccelerator):
    """Variable-density sampling with explicit acceleration range control."""

    def __init__(
        self,
        num_timesteps: int,
        min_acceleration: float = 1.5,
        max_acceleration: float = 8.0,
        density_power: float = 2.0,
        center_fraction: float = 0.0325,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            min_acceleration (float): Description.
            max_acceleration (float): Description.
            density_power (float): Description.
            center_fraction (float): Description.
            seed (int | None): Description.
        """
        min_acceleration = max(1.0, min_acceleration)
        max_acceleration = max(min_acceleration, max_acceleration)
        super().__init__(
            num_timesteps=num_timesteps,
            max_acceleration=max_acceleration,
            density_power=density_power,
            center_fraction=center_fraction,
            seed=seed,
            **kwargs,
        )
        self.min_acceleration = min_acceleration

    # Removed redundant get_acceleration_factor override
    @property
    def base_acceleration(self) -> float:
        """base_acceleration.

        Returns:
            float: Description.
        """
        return self.min_acceleration


class PoissonDiskKSpaceAccelerator(KSpaceAccelerator):
    r"""Blue-noise Poisson disk sampling with progressive density control.

    Refactored for Fixed Rank Sampling to ensure nested properties ($M_{t+1} \subset M_t$).
    Instead of regenerating points at every timestep (which causes 'twinkling'),
    we generate a maximal set of Poisson-disk points once (at max density), assign
    random priorities, and threshold them based on the target sampling fraction.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        min_radius: int = 2,
        max_radius_fraction: float = 0.12,
        min_density: float = 0.0,
        max_density: float = 1.0,
        center_fraction: float = 0.0325,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """__init__.

        Args:
            num_timesteps (int): Description.
            max_acceleration (float): Description.
            min_radius (int): Description.
            max_radius_fraction (float): Description.
            min_density (float): Description.
            max_density (float): Description.
            center_fraction (float): Description.
            seed (int | None): Description.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.min_radius = max(1, min_radius)
        self.max_radius_fraction = max(0.0, min(max_radius_fraction, 0.5))
        self.min_density = max(0.0, min(min_density, 1.0))
        self.max_density = max(self.min_density, min(max_density, 1.0))
        self.center_fraction = max(0.0, min(center_fraction, 0.5))
        self.seed = seed

        # Master patterns keyed by (height, width). ``_generate_master_pattern``
        # returns a host-side ``list[tuple[int, int]]`` of (y, x) indices, so
        # this cache holds NO device memory and cannot contribute to GPU OOM —
        # the rationale that previously disabled it claimed otherwise. Rebuilding
        # one costs 0.58 s at 128², 2.38 s at 256², 4.05 s at 320² (99.9% Python),
        # and ``generate_batch_masks`` requests one per sample per step.
        self._cached_patterns: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def _generate_master_pattern(
        self, height: int, width: int, device: torch.device
    ) -> list[tuple[int, int]]:
        """Generate a maximal density Poisson disk pattern and assign priorities."""
        # Use a local RNG for pattern generation to ensure determinism if seed is set
        # We seed it with self.seed + some hash of shape to be unique per shape but deterministic
        seed_offset = (height * 31 + width) % 100000
        local_seed = (self.seed + seed_offset) if self.seed is not None else None

        # Use numpy RNG for the point generation logic (CPU side)
        rng = np.random.default_rng(local_seed)

        # Determine sampling parameters for the MASTER pattern (densest possible)
        # We use min_radius typically, or a radius derived from max_density
        # But to ensure valid Poisson properties at the base level, we use min_radius.
        radius = self.min_radius

        # Target samples for max density (upper bound approx)
        # We generate a bit more than needed to be safe, then crop to max_density
        total_pixels = height * width

        # Start with center points — _apply_center_patch expects (C, H, W)
        mask = torch.zeros((1, height, width), dtype=torch.bool, device=device)
        if self.center_fraction > 0.0:
            self._apply_center_patch(mask, self.center_fraction)
        mask = mask.squeeze(0)  # back to (H, W) for point extraction

        # Extract initial points
        initial_points = torch.nonzero(mask)  # [N, 2]
        points: list[tuple[int, int]] = [(int(p[0]), int(p[1])) for p in initial_points]

        # Standard Dart Throwing / Bridson-like generation for the rest
        # We do this on CPU for logic simplicity, then convert to Tensor

        # Grid for spatial hashing
        cell_size = radius / math.sqrt(2)
        grid_w = math.ceil(width / cell_size)
        grid_h = math.ceil(height / cell_size)
        grid = {}  # (gx, gy) -> list of points

        def get_grid_coords(p):
            """get_grid_coords.

            Args:
                p (Any): Description.
            Returns:
                Any: Description.
            """
            return int(p[1] / cell_size), int(p[0] / cell_size)

        # Populate grid with initial points
        for p in points:
            gx, gy = get_grid_coords(p)
            if (gx, gy) not in grid:
                grid[(gx, gy)] = []
            grid[(gx, gy)].append(p)

        # Max attempts strategy
        max_total_samples = int(total_pixels * self.max_density)
        max_attempts = max_total_samples * 10

        # Fast candidate generation: Generate a large batch of random coordinates
        # and filter them.
        batch_size = 10000

        # We will collect 'valid' points in order
        new_points = []

        # Pre-generate random candidates
        # Note: Valid Poisson generation is sequential. Parallel is hard.
        # We stick to sequential checks but optimized.

        candidates_y = rng.integers(0, height, size=max_attempts)
        candidates_x = rng.integers(0, width, size=max_attempts)

        # Simple grid check
        idx = 0
        added_count = 0

        # Limit generation to max_density
        target_count = int(total_pixels * self.max_density)
        current_count = len(points)

        while current_count < target_count and idx < max_attempts:
            # Batch process? No, sequential dependency.
            y, x = candidates_y[idx], candidates_x[idx]
            idx += 1

            # Check if valid
            gx, gy = int(x / cell_size), int(y / cell_size)

            # Check neighbors
            valid = True
            for nx in range(gx - 2, gx + 3):
                for ny in range(gy - 2, gy + 3):
                    if (nx, ny) in grid:
                        for other_y, other_x in grid[(nx, ny)]:
                            dist_sq = (y - other_y) ** 2 + (x - other_x) ** 2
                            if dist_sq < radius**2:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                points.append((y, x))
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append((y, x))
                current_count += 1

        return points

    _PATTERN_CACHE_MAX = 8

    def _get_master_pattern(
        self, height: int, width: int, device: torch.device
    ) -> list[tuple[int, int]]:
        """Return the master pattern for ``(height, width)``, building it once.

        The pattern is the maximal-density blue-noise point set; per-timestep
        variation comes from how long a *prefix* of its rank order
        :meth:`get_acceleration_mask` consumes, not from regenerating it. The
        cached value is a host-side list and is never mutated by callers.
        """
        key = (height, width)
        cached = self._cached_patterns.get(key)
        if cached is None:
            cached = self._generate_master_pattern(height, width, device)
            if len(self._cached_patterns) >= self._PATTERN_CACHE_MAX:
                # Bounded: k-space shapes per run are few, so a full clear is
                # simpler than an LRU and keeps the footprint flat.
                self._cached_patterns.clear()
            self._cached_patterns[key] = cached
        return cached

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate Poisson-disk mask using cached master pattern."""
        channels, height, width = self._unpack_shape(kspace_shape)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        desired_fraction = self._target_sampling_fraction(timestep)

        # [DEBUG] Trace internal sampling params
        logger.debug(f"PoissonDisk: t={timestep} | Desired Frac: {desired_fraction:.4f}")

        total_pixels = height * width
        desired_samples = max(1, int(round(total_pixels * desired_fraction)))

        points_list = self._get_master_pattern(height, width, device)
        # Convert to tensor for faster indexing
        if points_list:
            master_points = torch.tensor(points_list, device=device)
        else:
            master_points = torch.empty((0, 2), device=device, dtype=torch.long)

        # Apply center patch first
        sampled = self._apply_center_patch(mask, self.center_fraction)

        # Remaining budget
        remaining = desired_samples - sampled

        if remaining > 0 and master_points.numel() > 0:
            # Filter points already covered by center patch?
            # Or just take points from master list until budget met, ignoring if they overlap center (inefficient but safe if mask handles duplicates)
            # Better: check mask.

            # Since master points are ordered by generation (rank), we iterate
            # But iterating list is slow.
            # Using vector ops:

            # We take first N points from master that are NOT in mask.
            # But we don't know how many we need to check to find N new points.

            # Simple approximation for Fixed Rank:
            # Just verify mask status.

            # Take a chunk of points, check mask, add new ones.
            # Or just trust that master points cover the space well.

            # Let's take 'remaining' * 1.5 candidate points and filter.
            # Actually, `master_points` are the "Blue Noise" points.
            # We should prioritize them in order.

            count = 0
            limit = master_points.shape[0]

            # Vectorized approach:
            # Get Y, X coordinates
            ys = master_points[:, 0].long()
            xs = master_points[:, 1].long()

            # Check which are not yet set
            # mask[0] is the 2D mask
            is_unset = ~mask[0, ys, xs]

            # Get indices of unset points
            valid_indices = torch.nonzero(is_unset, as_tuple=True)[0]

            # Take top 'remaining'
            num_to_take = min(remaining, valid_indices.numel())
            if num_to_take > 0:
                chosen_idx = valid_indices[:num_to_take]
                mask[:, ys[chosen_idx], xs[chosen_idx]] = True

        self._ensure_sample_budget(mask, desired_samples)
        return mask


class NestedKSpaceAccelerator(KSpaceAccelerator):
    """Nested k-space acceleration using fixed permutations.

    Ensures that the set of sampled points at timestep t+1 is always a subset
    of the points sampled at timestep t (nested sampling). This prevents
    'twinkling' artifacts where frequencies appear and disappear randomly.
    """

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        seed: int | None = None,
        center_fraction: float = 0.0,
        **kwargs: Any,
    ):
        """Initialise the nested (fixed-permutation) accelerator.

        Args:
            num_timesteps: Diffusion horizon.
            max_acceleration: Acceleration at the top of the ladder.
            seed: Seed for the single, timestep-invariant permutation.
            center_fraction: Always-sampled ACS fraction. Defaults to 0.0 for
                backward compatibility, but any arm that needs coil-sensitivity
                estimation or a phase reference must set it.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        self.seed = seed
        self.center_fraction = center_fraction
        # Seed-keyed and bounded for the same reason as the variable-density
        # ranking: dynamic-mask training mutates ``.seed`` per sample (#1071).
        self._permutation_cache: BoundedLRUCache[tuple[Any, ...], torch.Tensor] = (
            BoundedLRUCache()
        )
        import logging

        logging.getLogger(__name__).info(f"Initialized NestedKSpaceAccelerator with seed={seed}")

    def _get_fixed_permutation(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Get or generate the fixed permutation for ``shape`` and the current seed."""
        seed = self.seed if self.seed is not None else 0
        shape_key = (tuple(shape), str(device), seed)

        if shape_key not in self._permutation_cache:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

            total_pixels = int(np.prod(shape))
            perm = torch.randperm(total_pixels, generator=gen, device=device)
            self._permutation_cache[shape_key] = perm

        return self._permutation_cache[shape_key]

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate nested acceleration mask."""
        channels, height, width = self._unpack_shape(kspace_shape)
        total_pixels = height * width

        # 1. Determine the budget from the CONFIGURED ladder.
        #
        # This used to interpolate linearly from 100% at t=0 to 1/max_acceleration
        # at t=T, which ignored ``acceleration_schedule``, ``acceleration_range``
        # and ``base_acceleration`` entirely -- on the exp_11 step ladder it kept
        # 57% of k-space where the rung asked for 10%, a +469% budget error. Every
        # other accelerator resolves its budget through ``_target_sampling_fraction``;
        # so does this one now.
        keep_fraction = self._target_sampling_fraction(timestep)
        logger.debug(f"Nested: t={timestep} | keep_fraction: {keep_fraction:.4f}")

        num_keep = int(round(total_pixels * keep_fraction))
        num_keep = max(1, min(num_keep, total_pixels))

        # 3. Build the mask: ACS first, then the highest-priority remaining
        # points from the fixed permutation, so the total lands on the budget.
        #
        # Order matters. Selecting the budget from the permutation and *then*
        # forcing the centre overshoots by however much of the ACS the
        # permutation had not already picked -- at center_fraction 0.08 on the
        # exp_11 ladder that was +92% over the requested fraction.
        mask = torch.zeros((1, height, width), device=device, dtype=torch.bool)
        center_fraction = getattr(self, "center_fraction", 0.0) or 0.0

        # The ACS is not optional physics: a uniform permutation includes the
        # k-space centre only by luck, so without this the family could return a
        # mask with no DC term -- no phase reference, and nothing for
        # coil-sensitivity estimation to fit.
        #
        # Its width comes from `_guaranteed_core_fraction`, so a declared
        # `min_center_fraction` is honoured. Holding it at the full
        # `center_fraction` capped every realised acceleration at
        # `1/center_fraction` -- 12.64x against a ladder declaring 32x -- which
        # made the accelerator *named* for the cold-diffusion nesting property
        # unable to reach the acceleration its own arm declared (#1159). The
        # fixed-ranking variant of the helper is the right one here for the same
        # reason the ranking is cached: one permutation, one core, truncate by
        # budget. Unchanged when no floor was declared.
        kept = self._apply_center_patch(mask, self._guaranteed_core_fraction(center_fraction))

        remaining = num_keep - kept
        if remaining > 0:
            perm = self._get_fixed_permutation((height, width), device)
            flat = mask.view(-1)
            # Keep the permutation's order, drop what the ACS already covers:
            # the ranking stays timestep-invariant, so a smaller budget yields a
            # strict subset and the nested property survives.
            available = perm[~flat[perm]]
            flat[available[:remaining]] = True

        # Expand to channels if needed
        if channels > 1:
            mask = mask.expand(channels, height, width)

        return mask


class DensityNestedKSpaceAccelerator(KSpaceAccelerator):
    """One density, one draw, one truncation: nested variable-density sampling.

    Every other family in this module reaches the nested property
    ``M_{t+1} subset-of M_t`` by ranking k-space with a *deterministic* score and
    taking the top-K. That nests, but it also collapses: the top-K of a
    monotone-decreasing-in-radius score IS a centred disc, so at high
    acceleration the "random" and "variable density" families both realise a
    low-pass filter with no incoherent aliasing at all. Measured on
    ``experiment_11_attention_none`` at its own values (``center_fraction`` 0.08,
    ladder ``[2,4,8,10,12,16,32]``), ``random_cartesian`` realises a mask that is
    100% ACS from R=16 upward -- Jaccard 0.999 against the ideal disc of equal
    cardinality. Compressed sensing needs the opposite: incoherent aliasing.

    The fix is to make the ranking a *weighted random permutation* rather than a
    sort of the density itself. Draw Gumbel keys ``log w(k) + G_k`` once and sort
    descending; the top-K prefix is then a weighted random sample of size K
    without replacement (Efraimidis & Spirakis 2006 -- the Gumbel-top-K form),
    with inclusion probability following ``w``. Prefixes of one permutation are
    nested for free, so the whole cascade is a single draw truncated at each
    rung: build the density, draw the mask at the lowest acceleration, and every
    higher rung is that mask with its lowest-ranked bins removed.

    A contiguous centre-out core of ``min_center_fraction`` (falling back to
    ``center_fraction``) is placed at the head of the ranking, so ESPIRiT keeps a
    contiguous ACS with DC at every rung. The core is a prefix, not a clamp: a
    budget smaller than the core truncates it centre-out rather than overriding
    the requested acceleration.

    The trade-off is inherent to nesting and worth stating: one fixed ranking
    freezes the *shape* of the density across the ladder, and only the budget
    moves. A per-rung redraw would decouple them and is exactly what breaks
    ``M_{t+1} subset-of M_t``.
    """

    #: Exact-K by construction -- the top-up in ``_ensure_sample_budget`` would
    #: be a no-op, and it is never invoked here. That matters: the top-up ADDS
    #: lines from a second, independent permutation, which is one of the two
    #: measured mechanisms that break nesting elsewhere in this module.
    enforces_sample_budget = False

    def __init__(
        self,
        num_timesteps: int,
        max_acceleration: float = MAX_ACCELERATION,
        density_power: float = 2.0,
        center_fraction: float = 0.0325,
        seed: int | None = None,
        line_axis: Literal["y", "x"] | None = None,
        **kwargs: Any,
    ):
        """Initialise the density-nested accelerator.

        Args:
            num_timesteps: Diffusion horizon.
            max_acceleration: Acceleration at the top of the ladder.
            density_power: Exponent of the radial density. Higher concentrates
                the draw toward DC; 0.0 gives a uniform draw.
            center_fraction: Nominal ACS fraction. Used as the guaranteed core
                only when ``min_center_fraction`` is absent -- otherwise the
                bins between the floor and the nominal width are left to the
                density, which already favours them.
            seed: Seed for the single, timestep-invariant Gumbel draw.
            line_axis: ``"y"`` or ``"x"`` to sample whole Cartesian lines (2D
                Cartesian MRI can only skip entire phase-encode lines), or None
                for free 2D point sampling as on a 3D/PROPELLER-style readout.
        """
        super().__init__(num_timesteps, max_acceleration, **kwargs)
        if line_axis is not None and line_axis not in ("y", "x"):
            msg = f"line_axis must be 'y', 'x' or None (got {line_axis!r})"
            raise ValueError(msg)
        if density_power < 0.0:
            msg = f"density_power must be non-negative (got {density_power})"
            raise ValueError(msg)
        self.density_power = density_power
        self.center_fraction = max(0.0, min(center_fraction, 0.5))
        self.seed = seed
        self.line_axis = line_axis
        # BOUNDED, not a dict: the key carries ``seed`` (see ``_ranking``) and
        # ``_generate_batch_masks_dynamic`` mutates ``.seed`` once per sample, so an
        # unbounded memo accrued one never-reread entry per sample for the life of
        # the run. The fixed-seed cascade -- the only reader that ever hits -- needs
        # one entry per (shape, device), which the cap comfortably holds.
        self._ranking_cache: BoundedLRUCache[
            tuple[int, int, str, int | None], tuple[torch.Tensor, int]
        ] = BoundedLRUCache()

    @property
    def core_fraction(self) -> float:
        """Fraction of k-space guaranteed sampled at every rung.

        ``min_center_fraction`` is the ACS *floor* in this module's vocabulary,
        so it is the right quantity for a band that must survive the whole
        ladder. Falls back to ``center_fraction`` when no floor was declared.

        Shared with :class:`RandomCartesianKSpaceAccelerator` via
        :meth:`KSpaceAccelerator._guaranteed_core_fraction` -- both are
        fixed-ranking families, so both need the same core and must not drift.
        """
        return self._guaranteed_core_fraction(self.center_fraction)

    def _ranking(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, int]:
        """The single timestep-invariant ranking, plus the core's length.

        Returned as flat k-space indices when ``line_axis`` is None, or as line
        indices along the chosen axis otherwise.

        Memoized on ``(height, width, device, seed)``. The seed belongs in the
        key: dynamic-mask training reuses ONE accelerator instance and mutates
        ``.seed`` per sample (``kspace_process.py:610``), so a shape-only key
        would hand every sample the first draw and silently make
        ``enable_dynamic_mask`` a no-op.

        Computed ENTIRELY on CPU and moved at the end, so the same ``seed``
        yields the same realised mask whichever device asked for it. It did not:
        ``torch.Generator(device=...)`` is a device-specific stream, so the
        Gumbel draw below differed between CPU and CUDA and 28 of 29 timesteps
        realised different masks for one declared ``mask_seed`` -- identical
        cardinality at every rung, and nested on both, which is exactly why it
        read as reproducible for so long (#1510). Moving only the draw would not
        be enough: ``argsort`` tie-breaking is not guaranteed identical across
        devices either, so the whole ranking is built here and only the finished
        index tensor crosses. This is the idiom three other seeded draws in this
        module already use (``generator=gen`` on CPU, then ``.to(device)``).

        Bit-parity is a promise about CPU realisations, which sim2rank's
        CPU-canonical backend depends on: CPU masks are unchanged by this, and
        GPU masks now match them rather than being a second, undeclared draw.
        """
        cache_key = (height, width, str(device), self.seed)
        cached = self._ranking_cache.get(cache_key)
        if cached is not None:
            return cached

        cpu = torch.device("cpu")
        generator = self._make_generator(cpu, 0)

        if self.line_axis is None:
            rows = torch.arange(height, device=cpu, dtype=torch.float32) - height // 2
            cols = torch.arange(width, device=cpu, dtype=torch.float32) - width // 2
            radius = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2).flatten()
            total = height * width
            # A disc of equal AREA to the requested fraction, so core_fraction
            # keeps meaning "fraction of k-space", matching #581's ACS geometry.
            core_radius = math.sqrt(max(total * self.core_fraction, 1.0) / math.pi)
        else:
            num_lines = height if self.line_axis == "y" else width
            radius = (
                torch.arange(num_lines, device=cpu, dtype=torch.float32) - num_lines // 2
            ).abs()
            total = num_lines
            # A contiguous band of lines, so the fraction is a width not an area.
            core_radius = max(total * self.core_fraction, 1.0) / 2.0

        in_core = radius <= core_radius
        core = torch.nonzero(in_core, as_tuple=False).flatten()
        # Centre-out, so a budget below the core still keeps DC and stays
        # contiguous rather than truncating at an arbitrary raster prefix.
        core = core[torch.argsort(radius[core], stable=True)]

        max_radius = float(radius.max().clamp_min(1.0))
        weights = 1.0 / (1.0 + (radius / (max_radius * 0.3)) ** self.density_power)
        weights = weights.masked_fill(in_core, 0.0)

        # Gumbel-top-K: argsort(log w + G) is a weighted permutation without
        # replacement. Core bins carry w=0 -> key -inf -> they sort last and are
        # sliced off, so `core` below is the only place they appear.
        uniform = torch.rand(total, generator=generator, device=cpu)
        uniform = uniform.clamp(min=torch.finfo(uniform.dtype).tiny)
        keys = torch.log(weights) - torch.log(-torch.log(uniform))
        periphery = torch.argsort(keys, descending=True)[: total - int(core.numel())]

        # The only tensor that crosses. Everything above is device-independent
        # by construction; the cache key keeps ``device`` because what is stored
        # is the moved tensor, not the recipe.
        ranking = torch.cat([core, periphery]).to(device)
        result = (ranking, int(core.numel()))
        self._ranking_cache[cache_key] = result
        return result

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Take the top-K prefix of the fixed weighted permutation."""
        channels, height, width = self._unpack_shape(kspace_shape)
        ranking, _ = self._ranking(height, width, device)

        desired_fraction = self._target_sampling_fraction(timestep)
        mask = torch.zeros((channels, height, width), dtype=torch.bool, device=device)

        if self.line_axis is None:
            k = max(1, round(height * width * desired_fraction))
            selected = ranking[: min(k, ranking.numel())]
            mask.view(channels, -1)[:, selected] = True
        else:
            num_lines = height if self.line_axis == "y" else width
            k = max(1, round(num_lines * desired_fraction))
            selected = ranking[: min(k, ranking.numel())]
            if self.line_axis == "y":
                mask[:, selected, :] = True
            else:
                mask[:, :, selected] = True

        return mask


_ACCELERATOR_REGISTRY: dict[str, type[KSpaceAccelerator] | dict[str, Any]] = {
    "nested": NestedKSpaceAccelerator,
    # Nested by construction AND incoherent -- see the class docstring for why
    # the deterministic-ranking families are only the former (issue #1066).
    "density_nested": DensityNestedKSpaceAccelerator,
    "linear": LinearKSpaceAccelerator,
    "radial": RadialKSpaceAccelerator,
    "variable_density": VariableDensityKSpaceAccelerator,
    "fractional_variable_density": FractionalVariableDensityAccelerator,
    "spiral": SpiralKSpaceAccelerator,
    "golden_angle": RadialKSpaceAccelerator,
    "random_cartesian": RandomCartesianKSpaceAccelerator,
    "poisson_disk": PoissonDiskKSpaceAccelerator,
    # New types from audit file
    "uniform_cartesian": UniformCartesianKSpaceAccelerator,
    "partial_fourier": PartialFourierKSpaceAccelerator,
    "variable_density_1d": VDCartesian1DAccelerator,
    "variable_density_2d_gaussian": VDCartesian2DGaussianAccelerator,
    "variable_density_cava": VDCartesianCAVAAccelerator,
    "learned_pattern": {"class": LearnedPatternAccelerator, "params": {}},
    "multi_mask": {
        "class": "spectramr.models.utils.multi_mask_accelerator.MultiMaskAccelerator",
        "params": {},
    },
    "equispaced": UniformCartesianKSpaceAccelerator,
    # A shrinking centred low-pass disc. Implemented and unit-tested since the
    # sampling expansion, but reachable from no config until now (issue #954).
    "low_pass": LowPassAccelerator,
}

SUPPORTED_ACCELERATION_TYPES: tuple[str, ...] = tuple(
    sorted(_ACCELERATOR_REGISTRY),
)


class ColdDiffusionAccelerator:
    """Primary entry point that instantiates a configured accelerator."""

    def __init__(
        self,
        num_timesteps: int,
        acceleration_type: str = "variable_density",
        enforce_nested: bool = False,
        nested_tolerance: float = 0.5,
        **kwargs: Any,
    ):
        """Wrap a concrete accelerator, optionally enforcing nested sampling.

        Args:
            num_timesteps: Diffusion horizon.
            acceleration_type: Key into the accelerator registry.
            enforce_nested: When True, coerce the cascade so that
                ``M_{t+1} subset-of M_t`` holds for every timestep. Cold
                diffusion's forward process assumes k-space is only ever
                removed as ``t`` grows; several families violate that by
                re-drawing their pattern per timestep instead of truncating one
                ranking (radial, spiral and multi_mask most severely). Defaults
                to False so existing runs are byte-identical.
            nested_tolerance: Minimum share of the family's OWN raw draw at each
                timestep that the coerced mask must retain before this raises.
                Coercion can only remove samples, so a family that re-draws
                heavily collapses towards the ACS; failing loudly beats training
                on a cascade that silently lost most of its k-space.

                The denominator is the raw draw, deliberately -- NOT the
                continuous ``1 / declared_R``. Cartesian families quantise in
                whole k-space lines and so can never match a continuous target
                exactly; measuring against it made this guard fire on sub-line
                rounding and left ``nested_tolerance=1.0`` unsatisfiable even for
                a family that nests perfectly. Against the raw draw, 1.0 is the
                meaningful strict setting: "enforcement must be a no-op".
                Whether the family's raw draw honours its declared R is a
                separate question, answered by
                ``KSpaceUndersamplingProcess.declared_ladder_defects``.

        Raises:
            ValueError: If ``acceleration_type`` is unknown, or if
                ``enforce_nested`` collapses the cascade below
                ``nested_tolerance``.
        """
        self.num_timesteps = num_timesteps
        self.acceleration_type = acceleration_type
        self.enforce_nested = bool(enforce_nested)
        if not 0.0 < float(nested_tolerance) <= 1.0:
            raise ValueError(f"nested_tolerance must lie in (0, 1], got {nested_tolerance!r}.")
        self.nested_tolerance = float(nested_tolerance)
        # (shape, device, seed) -> per-bin first timestep at which the raw
        # cascade drops the bin. ``M_t = first_drop > t`` is then nested by
        # construction, and costs one int16 map rather than T masks.
        # Bounded for the same reason as ``DensityNestedKSpaceAccelerator``'s
        # ranking memo: the key carries ``seed``. Enforcement is currently switched
        # off on the seed-mutating path, so this one does not grow today -- it is
        # bounded so that re-enabling it cannot reintroduce the leak silently.
        self._nested_cache: BoundedLRUCache[tuple[Any, ...], torch.Tensor] = BoundedLRUCache()

        try:
            registry_entry = _ACCELERATOR_REGISTRY[acceleration_type]
        except KeyError as exc:
            available = ", ".join(sorted(_ACCELERATOR_REGISTRY))
            msg = f"Unknown acceleration type: {acceleration_type}. Available types: {available}"
            raise ValueError(msg) from exc

        # Direct class, ``{"class": ...}`` registration and lazy import paths all resolve
        # through the one owner, which ``_accelerator_kwarg_vocabulary`` also uses.
        accelerator_cls = _resolve_accelerator_cls(registry_entry, acceleration_type)

        # [LOGGING] Trace object creation parameters as requested
        logger.info(
            f"Creating Accelerator: {getattr(accelerator_cls, '__name__', str(accelerator_cls))} "
            f"| Timesteps: {num_timesteps} | Params: {kwargs}"
        )

        # ``phase``/``readout`` name a physical axis, so they resolve through the
        # arm's declared phase-encode axis (#2160). The axis is consumed whether
        # or not a direction was declared: every family would swallow it.
        phase_encode_axis = kwargs.pop("phase_encode_axis", None)
        if "mask_direction" in kwargs:
            direction = kwargs.pop("mask_direction")
            resolved_axis = line_axis_for(direction, phase_encode_axis)
            declared_axis = kwargs.get("line_axis")
            if declared_axis is not None and declared_axis != resolved_axis:
                raise ValueError(
                    f"mask_direction={direction!r} resolves to line_axis={resolved_axis!r} "
                    f"but line_axis={declared_axis!r} was also passed; declare one."
                )
            kwargs["line_axis"] = resolved_axis

        # [COMPATIBILITY] Map 'schedule_type' to 'acceleration_schedule' if present
        # If schedule_type is explicitly provided (not None), use it instead of acceleration_schedule
        if "schedule_type" in kwargs:
            schedule_type_value = kwargs.pop("schedule_type")
            if schedule_type_value is not None:
                # schedule_type takes priority over the default acceleration_schedule
                kwargs["acceleration_schedule"] = schedule_type_value

        # Last gate before dispatch, and deliberately AFTER the two alias mappings above
        # so a translated spelling is judged on the name that actually arrives.
        _reject_unknown_accelerator_kwargs(accelerator_cls, kwargs)

        self.accelerator = accelerator_cls(
            num_timesteps=num_timesteps,
            **kwargs,
        )

    @property
    def seed(self) -> int | None:
        """Get seed from underlying accelerator."""
        return getattr(self.accelerator, "seed", None)

    def _first_drop_map(
        self,
        kspace_shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """First timestep at which the raw cascade drops each k-space bin.

        Building this once costs ``num_timesteps`` raw mask evaluations; every
        later query is a comparison. The enforced mask is ``first_drop > t``,
        which is the cumulative intersection ``M_0 and M_1 and ... and M_t`` —
        monotone by construction, so nesting is guaranteed rather than hoped for.
        """
        key = (tuple(kspace_shape), str(device), self.seed)
        cached = self._nested_cache.get(key)
        if cached is not None:
            return cached

        first_drop = torch.full(
            kspace_shape[-2:], self.num_timesteps, dtype=torch.int16, device=device
        )
        shortfalls: list[str] = []
        for t in range(self.num_timesteps):
            raw_mask = self.accelerator.get_acceleration_mask(kspace_shape, t, device=device)
            # Rank is contractual (see ``KSpaceAccelerator.get_acceleration_mask``),
            # but this consumer used to take ``[0]`` unconditionally. A family that
            # returned ``(H, W)`` then had a single k-space ROW broadcast over the
            # whole plane, and the shortfall guard below blamed the family's geometry
            # for what was a shape bug -- so check the rank here and say so.
            if raw_mask.dim() != 3:
                msg = (
                    f"{self.acceleration_type!r} returned a rank-{raw_mask.dim()} "
                    f"mask {tuple(raw_mask.shape)} for kspace_shape={kspace_shape}; "
                    f"get_acceleration_mask must return (channels, height, width). "
                    f"Normalise with KSpaceAccelerator._as_channelled before returning."
                )
                raise ValueError(msg)
            raw = raw_mask[0]
            # A bin drops at the FIRST t where the raw cascade omits it.
            newly = (~raw) & (first_drop == self.num_timesteps)
            first_drop[newly] = t

            # What enforcement COSTS is ``raw_fraction - enforced_fraction``: the
            # bins this timestep's own draw kept and that the cumulative
            # intersection then deleted. Measure against the family's raw draw --
            # never against the continuous ``1 / declared_R``, which this guard
            # used to do.
            #
            # The two are not interchangeable. Every Cartesian family quantises in
            # whole k-space LINES, so ``raw_fraction`` lands on a multiple of 1/H
            # while ``1 / declared_R`` is continuous: they can never be equal, and
            # comparing across that gap made sub-line rounding read as a collapse.
            # On experiment_11_attention_none at 256x256 the worst "shortfall" was
            # 0.018 of ONE line in 256 (t=3 keeps 103 lines where 103.018 were
            # requested). That raised outright at nested_tolerance=1.0, and at the
            # 0.5 default it emitted warnings which contradicted themselves --
            # "realises R=2.23 ... but the schedule declares R=2.23".
            #
            # Family-vs-declared drift is a real defect class, but it belongs to
            # ``KSpaceUndersamplingProcess.declared_ladder_defects``, which measures
            # it directly and carries its own tolerance. Conflating it into this
            # guard made a nesting check unsatisfiable for a family that nests
            # exactly -- the opposite of what it exists to detect.
            #
            # ``enforced ⊆ raw(t)`` holds by construction: ``first_drop > t`` means
            # the bin survived every draw s <= t, this one included. So the ratio is
            # bounded in [0, 1], and ``nested_tolerance=1.0`` means exactly
            # "enforcement must be a no-op" -- which an exactly-nesting family
            # satisfies bit-for-bit rather than approximately.
            enforced_fraction = float((first_drop > t).float().mean())
            raw_fraction = float(raw.float().mean())
            declared_r = max(1.0, self.accelerator.get_acceleration_factor(t))
            realised_r = 1.0 / enforced_fraction if enforced_fraction > 0.0 else float("inf")
            if enforced_fraction < self.nested_tolerance * raw_fraction:
                shortfalls.append(
                    f"t={t}: nesting kept {enforced_fraction:.4f} of k-space where "
                    f"the family's own draw kept {raw_fraction:.4f} "
                    f"(declared R={declared_r:.2f}, realised R={realised_r:.2f})"
                )
            elif enforced_fraction < raw_fraction:
                # Inside the tolerance band. This used to pass in total silence, so
                # an arm could train and report at its DECLARED R while its masks
                # realised up to ``1 / nested_tolerance`` times that -- 2x at the
                # 0.5 default. The band is a deliberate allowance, not a licence to
                # mislabel, so say what enforcement actually deleted.
                logger.warning(
                    "enforce_nested: nesting deleted %.4f of k-space at t=%d that "
                    "%s's own draw had kept (%.4f enforced vs %.4f raw), so the "
                    "masks realise R=%.2f where the schedule declares R=%.2f. This "
                    "is within nested_tolerance=%g so it is allowed, but metrics "
                    "and provenance record the DECLARED value. Lower "
                    "max_acceleration, pick a family that nests exactly, or raise "
                    "nested_tolerance deliberately.",
                    raw_fraction - enforced_fraction,
                    t,
                    self.acceleration_type,
                    enforced_fraction,
                    raw_fraction,
                    realised_r,
                    declared_r,
                    self.nested_tolerance,
                )

        if shortfalls:
            raise ValueError(
                f"enforce_nested collapsed the {self.acceleration_type!r} cascade "
                f"below nested_tolerance={self.nested_tolerance:g} of its own raw "
                f"draw. Nesting can only REMOVE samples, so a family that re-draws "
                f"its pattern per timestep loses everything the redraws disagree "
                f"on. Offending timesteps: {'; '.join(shortfalls[:5])}"
                f"{' ...' if len(shortfalls) > 5 else ''}. Either pick a family "
                f"that already nests, or lower nested_tolerance deliberately."
            )

        self._nested_cache[key] = first_drop
        return first_drop

    def get_acceleration_mask(
        self,
        kspace_shape: tuple[int, ...],
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Get acceleration mask for cold diffusion timestep."""
        if not self.enforce_nested:
            return self.accelerator.get_acceleration_mask(kspace_shape, timestep, device=device)

        # ``_generate_batch_masks_dynamic`` mutates the inner seed per sample to
        # vary the PATTERN at a fixed acceleration. A cascade built for one seed
        # says nothing about another, and rebuilding per sample would cost T mask
        # evaluations per item, so enforcement deliberately applies to the
        # fixed-seed cascade only -- which is the path the reverse trajectory and
        # validation walk, and the only place nesting has to hold.
        channels = kspace_shape[0] if len(kspace_shape) == 3 else 1
        first_drop = self._first_drop_map(kspace_shape, device)
        mask = (first_drop > int(timestep)).unsqueeze(0)
        if channels > 1:
            mask = mask.expand(channels, -1, -1)
        return mask

    def get_acceleration_factor(self, timestep: int) -> float:
        """Get acceleration factor for given timestep."""
        return self.accelerator.get_acceleration_factor(timestep)

    def timestep_for_acceleration(self, acceleration: float) -> int:
        """Delegate the schedule-aware inverse to the wrapped accelerator.

        See :meth:`KSpaceAccelerator.timestep_for_acceleration`. The
        validation cascade in ``infrastructure/training/strategies/diffusion.py``
        resolves the wrapped ``ColdDiffusionAccelerator`` through the strategy's
        ``KSpaceMaskGenerator`` and calls this method to pick a timestep
        whose mask realises the requested ``acceleration`` under the YAML's
        ``acceleration.schedule_type``.
        """
        return self.accelerator.timestep_for_acceleration(acceleration)

    def apply_acceleration(
        self,
        kspace: torch.Tensor,
        timestep: int,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply acceleration to k-space data.

        Args:
            kspace: Full k-space data
            timestep: Current diffusion timestep
            device: Target device

        Returns:
            Tuple of (accelerated_kspace, mask)

        """
        mask = self.get_acceleration_mask(kspace.shape[-2:], timestep, device)

        # Clone the fully sampled k-space and apply mask in-place
        accelerated_kspace = kspace.clone()
        if kspace.dim() == 3:  # (channels, height, width)
            accelerated_kspace.masked_fill_(~mask, 0)
        elif kspace.dim() == 4:  # (batch, channels, height, width)
            accelerated_kspace.masked_fill_(~mask.unsqueeze(0), 0)
        else:
            raise ValueError(f"Unsupported k-space dimensions: {kspace.shape}")

        return accelerated_kspace, mask


# Factory function for creating accelerators
def create_kspace_accelerator(
    acceleration_type: str = "variable_density",
    *,
    num_timesteps: int,
    **kwargs: Any,
) -> ColdDiffusionAccelerator:
    """Factory function for creating k-space accelerators.

    Args:
        acceleration_type: Type of acceleration. Supported values are listed in
            ``SUPPORTED_ACCELERATION_TYPES``.
        num_timesteps: Number of diffusion timesteps; the caller owns the horizon (#2443).
        **kwargs: Additional arguments for accelerator

    Returns:
        Configured ColdDiffusionAccelerator

    """
    return ColdDiffusionAccelerator(
        num_timesteps=num_timesteps,
        acceleration_type=acceleration_type,
        **kwargs,
    )
