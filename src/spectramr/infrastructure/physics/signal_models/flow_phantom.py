"""Analytic 4D-flow phantom: pulsatile Poiseuille flow in tilted tubes through static tissue.

Every field is a closed-form function of position, so one readout slice is
evaluated without building the volume. Positions are millimetres on the
``(X, Y, Z)`` grid, centred on the volume; ``X`` is the readout axis, and velocity
component ``i`` is the velocity along axis ``i`` (the convention the flow metrics
read). Each tube carries ``v(p, t) = v_peak w(t) (1 - r^2 / R^2)_+ d``, with ``r``
the distance from its straight axis ``d``: the profile is constant along ``d``, so
the field is exactly divergence-free, and quadratic inside the lumen, so central
differences reproduce that away from the wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from spectramr.infrastructure.physics.signal_models.flow_background import polynomial_basis

#: Largest tube tilt off the readout axis. With the slot offset below it keeps
#: tubes apart and every X-plane's lumen a whole cross-section inside the volume.
_MAX_TILT_RAD = math.radians(15.0)


@dataclass(frozen=True)
class FlowPhantomSpec:
    """Grid and physics knobs shared by every subject drawn from it."""

    grid: tuple[int, int, int]
    voxel_size_mm: tuple[float, float, float]
    frames: int
    coils: int
    vessels: int
    peak_velocity: float
    background_order: int
    background_amplitude: float


@dataclass(frozen=True)
class FlowPhantom:
    """One subject's drawn parameters."""

    centres: torch.Tensor  # [V, 3] mm
    directions: torch.Tensor  # [V, 3] unit
    radii: torch.Tensor  # [V] mm
    peaks: torch.Tensor  # [V] m/s
    waveform: torch.Tensor  # [T] in (0, 1]
    texture: torch.Tensor  # [K, 4]: wave vector (cycles over the FOV) and phase
    background: torch.Tensor  # [3, M] coefficients, one row per encoded direction
    common_phase: torch.Tensor  # [M] coefficients shared by every encoding
    coil_centres: torch.Tensor  # [C, 3] mm
    coil_phase: torch.Tensor  # [C]


def _extent(spec: FlowPhantomSpec) -> torch.Tensor:
    return torch.tensor([n * h for n, h in zip(spec.grid, spec.voxel_size_mm, strict=True)])


def _uniform(generator: torch.Generator, *shape: int, low: float = 0.0, high: float = 1.0):
    return low + (high - low) * torch.rand(shape, generator=generator)


def draw_flow_phantom(spec: FlowPhantomSpec, generator: torch.Generator) -> FlowPhantom:
    """Draw one subject. Tubes sit in distinct angular slots, so they do not overlap."""
    extent = _extent(spec)
    n_terms = polynomial_basis((2, 2, 2), spec.background_order).shape[1]
    slots = torch.arange(spec.vessels) * (2 * math.pi / spec.vessels)
    angle = slots + _uniform(generator, spec.vessels, low=-0.3, high=0.3)
    offset = 0.25 * extent[1:].min() * torch.stack([angle.cos(), angle.sin()], dim=-1)
    centres = torch.cat([torch.zeros(spec.vessels, 1), offset], dim=-1)
    tilt = _uniform(generator, spec.vessels, high=_MAX_TILT_RAD)
    azimuth = _uniform(generator, spec.vessels, high=2 * math.pi)
    directions = torch.stack(
        [tilt.cos(), tilt.sin() * azimuth.cos(), tilt.sin() * azimuth.sin()], dim=-1
    )
    min_voxel = min(spec.voxel_size_mm[1:])
    phase = torch.arange(spec.frames) / spec.frames
    coil_angle = torch.arange(spec.coils) * (2 * math.pi / max(spec.coils, 1))
    ring = 0.6 * extent[1:].max()
    return FlowPhantom(
        centres=centres,
        directions=directions,
        radii=min_voxel * _uniform(generator, spec.vessels, low=2.5, high=4.0),
        peaks=spec.peak_velocity * _uniform(generator, spec.vessels, low=0.6, high=1.0),
        waveform=0.15 + 0.85 * torch.exp(-(((phase - 0.2) / 0.1) ** 2)),
        texture=torch.cat(
            [_uniform(generator, 4, 3, low=-3.0, high=3.0), _uniform(generator, 4, 1, high=6.3)],
            dim=-1,
        ),
        background=spec.background_amplitude * torch.randn((3, n_terms), generator=generator),
        common_phase=math.pi * torch.randn(n_terms, generator=generator) / n_terms,
        coil_centres=torch.stack(
            [
                _uniform(generator, spec.coils, low=-0.3, high=0.3) * extent[0],
                ring * coil_angle.cos(),
                ring * coil_angle.sin(),
            ],
            dim=-1,
        ),
        coil_phase=_uniform(generator, spec.coils, high=2 * math.pi),
    )


def grid_basis_for(spec: FlowPhantomSpec) -> torch.Tensor:
    """Background-phase monomials over the whole grid, ``[X, Y, Z, M]``."""
    return polynomial_basis(spec.grid, spec.background_order).reshape(*spec.grid, -1)


def _slice_points(spec: FlowPhantomSpec, x_index: int) -> torch.Tensor:
    """``[Y, Z, 3]`` centred millimetre positions of readout slice ``x_index``."""
    centred = [
        (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * h
        for n, h in zip(spec.grid, spec.voxel_size_mm, strict=True)
    ]
    yy, zz = torch.meshgrid(centred[1], centred[2], indexing="ij")
    return torch.stack([torch.full_like(yy, float(centred[0][x_index])), yy, zz], dim=-1)


def phantom_slice(
    spec: FlowPhantomSpec,
    phantom: FlowPhantom,
    x_index: int,
    grid_basis: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Every field of readout slice ``x_index``.

    ``grid_basis`` is :func:`grid_basis_for` of ``spec``; a caller evaluating many
    slices builds it once rather than per slice.

    Returns:
        ``velocity`` ``[3, T, Y, Z]`` m/s; ``magnitude`` ``[Y, Z]``; ``lumen`` and
        ``static`` ``[Y, Z]`` bool; ``background`` ``[3, Y, Z]`` rad (per encoded
        direction); ``common_phase`` ``[Y, Z]`` rad; ``coil_maps`` ``[C, Y, Z]``
        complex, unit root-sum-of-squares.
    """
    points = _slice_points(spec, x_index)
    rel = points[None] - phantom.centres[:, None, None]  # [V, Y, Z, 3]
    along = (rel * phantom.directions[:, None, None]).sum(dim=-1, keepdim=True)
    r2 = (rel - along * phantom.directions[:, None, None]).pow(2).sum(dim=-1)  # [V, Y, Z]
    profile = (1.0 - r2 / phantom.radii[:, None, None] ** 2).clamp_min(0.0)
    lumen = (profile > 0).any(dim=0)
    speed = phantom.peaks[:, None, None] * profile  # [V, Y, Z]
    velocity = torch.einsum("vyz,vi,t->ityz", speed, phantom.directions, phantom.waveform)

    extent = _extent(spec)
    unit = points / extent  # [-1/2, 1/2] per axis
    body = (unit.pow(2) / 0.42**2).sum(dim=-1) <= 1.0
    texture = torch.cos(2 * math.pi * unit @ phantom.texture[:, :3].T + phantom.texture[:, 3])
    tissue = 0.5 * (1.0 + 0.15 * texture.sum(dim=-1))
    magnitude = torch.where(lumen, torch.ones_like(tissue), tissue * body)

    basis = (grid_basis if grid_basis is not None else grid_basis_for(spec))[x_index]  # [Y, Z, M]
    dist2 = (points[None] - phantom.coil_centres[:, None, None]).pow(2).sum(dim=-1)
    width = 0.7 * extent.max()
    raw = torch.exp(-dist2 / (2 * width**2)) * torch.exp(
        1j
        * (phantom.coil_phase[:, None, None] + 0.5 * unit[None, ..., 1] - 0.3 * unit[None, ..., 2])
    )
    coil_maps = (raw / raw.abs().pow(2).sum(dim=0).sqrt()).to(torch.complex64)
    return {
        "velocity": velocity,
        "magnitude": magnitude,
        "lumen": lumen,
        "static": body & ~lumen,
        "background": (basis @ phantom.background.T).permute(2, 0, 1),
        "common_phase": basis @ phantom.common_phase,
        "coil_maps": coil_maps,
    }


__all__ = ["FlowPhantom", "FlowPhantomSpec", "draw_flow_phantom", "grid_basis_for", "phantom_slice"]
