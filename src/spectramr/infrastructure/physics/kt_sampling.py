"""k-t undersampling masks for dynamic, velocity-encoded Cartesian acquisitions.

A phase-contrast cine samples a phase-encode grid (``ky`` for a 2D acquisition,
``(ky, kz)`` for 3D; readout is always fully encoded) once per velocity encoding
and cardiac frame. Drawing a different pattern per ``(encoding, frame)`` makes the
aliasing incoherent along time, which is what k-t reconstructions exploit.

These masks are indexed by ``(E, T)`` before the grid, a shape the 2D
``KSpaceAccelerator`` family (a timestep-progressive degradation schedule with a
fixed seed) cannot express, so they live here with their own name table, kept
disjoint from :class:`~spectramr.infrastructure.physics.sampling_registry.SamplingPatternRegistry`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _gaussian_log_density(grid: tuple[int, ...], sigma: float) -> torch.Tensor:
    axes = [torch.linspace(-1.0, 1.0, n) for n in grid]
    r2 = sum(g**2 for g in torch.meshgrid(*axes, indexing="ij"))
    return -r2 / (2 * sigma**2)


def _centre_box(grid: tuple[int, ...], fraction: float) -> torch.Tensor:
    box = torch.zeros(grid, dtype=torch.bool)
    index = []
    for n in grid:
        width = max(1, round(n * fraction))
        start = n // 2 - width // 2
        index.append(slice(start, start + width))
    box[tuple(index)] = True
    return box


def kt_variable_density_mask(
    shape: tuple[int, ...],
    acceleration: float,
    *,
    grid_ndim: int = 2,
    center_fraction: float = 0.08,
    sigma: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Gaussian variable-density k-t mask with a fully sampled centre.

    Every grid frame (one per leading index, e.g. ``(E, T)``) gets exactly
    ``round(N / acceleration)`` samples, ``N`` the grid size: the centre box, then
    points drawn without replacement with probability proportional to a Gaussian
    in normalised coordinates (Gumbel top-k), independently per frame.

    Args:
        shape: ``(*leading, *grid)``; the last ``grid_ndim`` axes are the phase-encode grid.
        acceleration: ``R >= 1``, the same in every frame.
        grid_ndim: 1 (``ky``) or 2 (``ky, kz``).
        center_fraction: centre box width per grid axis, as a fraction of that axis.
        sigma: Gaussian width in normalised ``[-1, 1]`` coordinates.
        generator: CPU RNG for reproducible masks.

    Returns:
        Bool tensor of ``shape``.
    """
    if grid_ndim not in (1, 2) or len(shape) < grid_ndim:
        raise ValueError(f"grid_ndim must be 1 or 2 and fit in shape {shape}; got {grid_ndim}")
    if acceleration < 1.0:
        raise ValueError(f"acceleration must be >= 1, got {acceleration}")
    if sigma <= 0.0 or not 0.0 <= center_fraction < 1.0:
        raise ValueError(
            f"need sigma > 0 and 0 <= center_fraction < 1; got {sigma}, {center_fraction}"
        )
    grid = tuple(shape[len(shape) - grid_ndim :])
    leading = tuple(shape[: len(shape) - grid_ndim])
    n_points = 1
    for n in grid:
        n_points *= n
    budget = round(n_points / acceleration)
    centre = _centre_box(grid, center_fraction).flatten()
    n_centre = int(centre.sum())
    if n_centre > budget:
        raise ValueError(
            f"the centre box holds {n_centre} points but R={acceleration} allows {budget} "
            f"per frame on a {grid} grid; lower center_fraction or acceleration"
        )

    frames = 1
    for n in leading:
        frames *= n
    uniform = torch.rand((frames, n_points), generator=generator).clamp_(1e-12, 1.0 - 1e-7)
    keys = _gaussian_log_density(grid, sigma).flatten() - torch.log(-torch.log(uniform))
    keys = keys.masked_fill(centre, float("inf"))
    chosen = keys.topk(budget, dim=1).indices
    mask = torch.zeros((frames, n_points), dtype=torch.bool)
    mask.scatter_(1, chosen, True)
    return mask.reshape(shape)


#: Pattern name -> sampler. Names must not collide with the 2D registry's.
KT_SAMPLING_PATTERNS: dict[str, Callable[..., torch.Tensor]] = {
    "kt_vd_gaussian": kt_variable_density_mask,
}


def kt_sampling_pattern(name: str) -> Callable[..., torch.Tensor]:
    """Resolve a k-t pattern name; raises on an unknown one."""
    try:
        return KT_SAMPLING_PATTERNS[name]
    except KeyError:
        raise ValueError(
            f"unknown k-t sampling pattern {name!r}; known: {sorted(KT_SAMPLING_PATTERNS)}"
        ) from None


__all__ = ["KT_SAMPLING_PATTERNS", "kt_sampling_pattern", "kt_variable_density_mask"]
