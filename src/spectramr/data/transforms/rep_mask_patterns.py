r"""Two-dimensional repetition masks for ``complementary_rep_masking``.

The 1-D transform draws phase-encode lines. This module draws the same pair
:math:`(M_1, M_2)` over the whole ``(H, W)`` grid for the 2-D families, so an
acceleration sweep can compare Cartesian lines against a variable-density
Poisson-disc and golden-angle pseudo-radial or pseudo-spiral sampling at one
meaning of R. M4Raw is a 2-D multi-slice acquisition whose readout is fully
sampled, so a 2-D mask here is a *retrospective* stand-in: it is what a 3-D
Cartesian (two phase-encode axes) or a gridded non-Cartesian acquisition would
have measured, rasterised onto the Cartesian grid. It is not a non-Cartesian
reconstruction; no NUFFT is involved.

**Names and geometry have one owner each.** A pattern name resolves through
:class:`~spectramr.infrastructure.physics.sampling_registry.SamplingPatternRegistry`;
the geometry comes from ``MaskGenerator.variable_density_poisson_disc`` and the
golden-angle orderings of :mod:`~spectramr.infrastructure.physics.trajectory_ladder`,
truncated by :class:`~spectramr.infrastructure.physics.trajectory_ladder.NestedTrajectoryLadder`.

**What R means.** For every pattern, 1-D included,
:math:`|M_1 \cap \mathrm{support}| = \mathrm{round}(|\mathrm{support}|/R)`, where
the 2-D support is the ``H x W`` grid restricted to the acquired phase-encode
columns (M4Raw zero-fills about 24% of them). The Poisson-disc radius is
bisected and its draw trimmed to the count exactly; a trajectory keeps the
fewest whole spokes (radial) or samples (spiral) that reach the count, so its R
is at most the declared one. An achieved R more than 5% off raises.

**Calibration region.** The coil maps of these arms are ESPIRiT maps fitted on
a centred square of ``coil_maps_acs_size`` bins (``estimate_csm_espirit`` with an
integer ``acs_size``, rows and columns from ``H//2 - n//2`` and ``W//2 - n//2``).
Every 2-D mask therefore contains the centred ``n x n`` square with
``n = round(center_fraction * W)`` -- the width of the 1-D ACS band, so a band
of 20 lines at ``center_fraction: 0.08`` becomes a 20 x 20 square -- in both
:math:`M_1` and :math:`M_2`, and the maps stay estimable from the masked input.

**Modes** keep their 1-D meaning: ``independent`` is a second draw (another
Poisson-disc seed; another golden-angle rotation), ``same`` is :math:`M_1`,
``full_target`` the support, and ``complementary`` the calibration region plus
the supported bins :math:`M_1` lacks, subsampled to :math:`|M_1\setminus C|`
above R=2.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import torch

from spectramr.data.transforms.normalization import acs_window
from spectramr.infrastructure.physics.sampling import MaskGenerator
from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry
from spectramr.infrastructure.physics.trajectory_ladder import (
    NestedTrajectoryLadder,
    golden_radial_ordering,
    golden_spiral_ordering,
)

__all__ = [
    "CARTESIAN_1D",
    "R_TOLERANCE",
    "SUPPORTED_PATTERNS",
    "calibration_region",
    "draw_rep_masks_2d",
    "resolve_pattern",
    "seeded_generator",
]

#: The transform's own 1-D family: ACS plus uniformly random phase-encode lines.
#: No registry family has that physics (``random_cartesian`` samples points,
#: ``uniform_cartesian`` is equispaced), so the name is not aliased onto one.
CARTESIAN_1D = "cartesian_1d"

#: Largest relative gap between the declared and the achieved R.
R_TOLERANCE = 0.05

#: Spiral interleave shape, as the cold-diffusion spiral family builds it.
_SPIRAL_TURNS, _SPIRAL_SAMPLES = 4.0, 1024


def seeded_generator(*keys: object) -> torch.Generator:
    """A CPU generator seeded from ``keys`` identically in every process."""
    token = "|".join(str(k) for k in keys).encode()
    return torch.Generator().manual_seed(zlib.crc32(token))


def calibration_region(height: int, width: int, center_fraction: float) -> torch.Tensor:
    """``(H, W)`` bool: the centred ``n x n`` ESPIRiT window, ``n = round(center_fraction * W)``."""
    region = torch.zeros(height, width, dtype=torch.bool)
    rows, cols = acs_window(height, width, center_fraction)
    region[rows, cols] = True
    return region


# --- M1 draws: (support, calibration, count, generator) -> mask --------------------

#: Bisection steps for the Poisson-disc radius; the radius is quantised to 1/4 bin.
_BISECTION_STEPS = 14
_RADIUS_CACHE: dict[tuple[bytes, int, int], float] = {}
_RADIUS_CACHE_MAX = 64


def _poisson_radius(allowed: torch.Tensor, need: int) -> float:
    """Largest DC radius whose saturated set holds at least ``need`` bins.

    Calibrated once per (allowed region, count) with a fixed seed, so the value
    is a pure function of its arguments and the memo changes nothing but time.
    """
    key = (allowed.numpy().tobytes(), allowed.shape[1], need)
    if key not in _RADIUS_CACHE:
        lo, hi = 0.0, float(max(allowed.shape))
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            kept = MaskGenerator.variable_density_poisson_disc(
                allowed, mid, seeded_generator("poisson-radius")
            )
            lo, hi = (mid, hi) if int(kept.sum()) >= need else (lo, mid)
        if len(_RADIUS_CACHE) >= _RADIUS_CACHE_MAX:  # a few supports per corpus
            _RADIUS_CACHE.clear()
        _RADIUS_CACHE[key] = lo
    return _RADIUS_CACHE[key]


def _draw_poisson(
    support: torch.Tensor, calib: torch.Tensor, count: int, gen: torch.Generator
) -> torch.Tensor:
    allowed = support & ~calib
    need = count - int(calib.sum())
    kept = MaskGenerator.variable_density_poisson_disc(allowed, _poisson_radius(allowed, need), gen)
    picked = torch.nonzero(kept.flatten()).flatten()
    if picked.numel() < need:  # a draw short of the calibration draw: top up uniformly
        spare = torch.nonzero((allowed & ~kept).flatten()).flatten()
        extra = spare[torch.randperm(spare.numel(), generator=gen)[: need - picked.numel()]]
        picked = torch.cat([picked, extra])
    picked = picked[torch.randperm(picked.numel(), generator=gen)[:need]]
    mask = calib.clone().flatten()
    mask[picked] = True
    return mask.view_as(calib)


def _trajectory_drawer(
    ordering: Callable[..., tuple[torch.Tensor, int]],
) -> Callable[[torch.Tensor, torch.Tensor, int, torch.Generator], torch.Tensor]:
    def draw(
        support: torch.Tensor, calib: torch.Tensor, count: int, gen: torch.Generator
    ) -> torch.Tensor:
        height, width = support.shape
        offset = float(torch.rand(1, generator=gen, dtype=torch.float64)) * torch.pi
        traj, quantum = ordering(height, width, angle_offset=offset)
        rows, cols = MaskGenerator.trajectory_bin_indices(traj, height, width)
        # Unsupported bins sit in the core so the prefix is counted on the support.
        core = calib | ~support
        ladder = NestedTrajectoryLadder(rows, cols, height, width, quantum, core, ordering.__name__)
        fraction = (count + int((~support).sum())) / (height * width)
        # The prefix that reaches the count, or one quantum (spoke) less if that
        # lands nearer: at R16 one spoke is ~6% of the budget.
        reach = ladder.sample_count(fraction)
        masks = [ladder.mask(s) & support for s in (reach, max(0, reach - quantum))]
        return min(masks, key=lambda m: abs(int(m.sum()) - count))

    return draw


def _spiral(height: int, width: int, *, angle_offset: float) -> tuple[torch.Tensor, int]:
    return golden_spiral_ordering(
        height,
        width,
        num_turns=_SPIRAL_TURNS,
        samples_per_arm=_SPIRAL_SAMPLES,
        angle_offset=angle_offset,
    )


_spiral.__name__ = "golden_spiral_ordering"

#: Canonical registry name -> M1 drawer. ``radial`` and ``golden_angle`` are one
#: accelerator class in the registry and one ordering here.
_DRAWERS: dict[str, Callable[[torch.Tensor, torch.Tensor, int, torch.Generator], torch.Tensor]] = {
    "poisson_disk": _draw_poisson,
    "radial": _trajectory_drawer(golden_radial_ordering),
    "golden_angle": _trajectory_drawer(golden_radial_ordering),
    "spiral": _trajectory_drawer(_spiral),
}

SUPPORTED_PATTERNS: tuple[str, ...] = (CARTESIAN_1D, *sorted(_DRAWERS))


def resolve_pattern(pattern: str) -> str:
    """``cartesian_1d``, or the canonical registry name of a supported 2-D family.

    Raises:
        ValueError: on a name the registry does not know, and on a registry
            family this transform does not draw -- never a fallback to lines.
    """
    if str(pattern).strip().lower() == CARTESIAN_1D:
        return CARTESIAN_1D
    canonical = SamplingPatternRegistry.resolve(pattern)
    if canonical not in _DRAWERS:
        raise ValueError(
            f"complementary_rep_masking: pattern {pattern!r} (canonical {canonical!r}) is a "
            f"sampling family this transform does not draw. Supported: {list(SUPPORTED_PATTERNS)}."
        )
    return canonical


# --- the pair ------------------------------------------------------------------------


def _m2(mode, draw, support, calib, m1, acceleration, count, keys) -> torch.Tensor:
    if mode == "same":
        return m1.clone()
    if mode == "full_target":
        return support.clone()
    if mode == "independent":
        return draw(support, calib, count, seeded_generator(*keys, "m2"))
    rest = torch.nonzero((support & ~m1).flatten()).flatten()
    if acceleration > 2:
        budget = int((m1 & ~calib).sum())
        rest = rest[torch.randperm(rest.numel(), generator=seeded_generator(*keys, "m2"))[:budget]]
    m2 = calib.clone().flatten()
    m2[rest] = True
    return m2.view_as(calib)


def draw_rep_masks_2d(
    pattern: str,
    support: torch.Tensor,
    height: int,
    *,
    acceleration: float,
    center_fraction: float,
    mode: str,
    keys: tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(M1, M2, achieved_R)`` over one slice's ``(H, W)`` grid.

    Args:
        pattern: a canonical name from :func:`resolve_pattern`, not ``cartesian_1d``.
        support: ``(W,)`` bool, the acquired phase-encode columns.
        height: readout length ``H``.
        acceleration, center_fraction, mode, keys: as for the 1-D ``draw_rep_masks``.

    Returns:
        ``M1`` within the support; ``M2`` with the unsupported columns added;
        ``|support| / |M1|``.

    Raises:
        ValueError: when the calibration region alone fills the budget, or the
            achieved R is more than :data:`R_TOLERANCE` off the declared one.
    """
    draw = _DRAWERS[pattern]
    width = support.numel()
    grid = support[None, :].expand(height, width).clone()
    calib = calibration_region(height, width, center_fraction) & grid
    count = round(int(grid.sum()) / acceleration)
    if int(calib.sum()) >= count:
        raise ValueError(
            f"complementary_rep_masking: the {int(calib.sum())}-bin calibration region fills "
            f"the {count}-bin budget of R={acceleration:g} on {int(grid.sum())} supported bins, "
            f"so the mask would be the calibration region alone at R="
            f"{int(grid.sum()) / max(1, int(calib.sum())):.3g}. Lower center_fraction."
        )
    m1 = draw(grid, calib, count, seeded_generator(*keys, "m1"))
    achieved = int(grid.sum()) / int(m1.sum())
    if abs(achieved / acceleration - 1.0) > R_TOLERANCE:
        raise ValueError(
            f"complementary_rep_masking: {pattern} achieved R={achieved:.3f} for a declared "
            f"R={acceleration:g} ({R_TOLERANCE:.0%} tolerance) on a {height}x{width} grid."
        )
    m2 = _m2(mode, draw, grid, calib, m1, acceleration, count, keys)
    return m1, m2 | ~grid, achieved
