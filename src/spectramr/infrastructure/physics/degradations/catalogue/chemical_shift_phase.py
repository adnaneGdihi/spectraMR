"""chemical_shift_phase: type-2 chemical-shift interference of fat and water in a GRE voxel.

Fat and water in one voxel add as ``|W + F exp(i 2 pi df TE)|`` with ``df = -sigma gamma/2pi B0``,
``sigma`` the scanner profile's ``chi_ppm`` (3.5 ppm, -447 Hz at 3 T), W = (1 - f) x, F = f x
(Dixon, Radiology 153:189, 1984; Bernstein, King & Zhou, Handbook of MRI Pulse Sequences,
2004, ch. 8). Severity moves TE from the first in-phase echo ``1/|df|`` to the first opposed-phase echo ``1.5/|df|``. Without a ``fat_map``
the fat fraction is a seeded smooth map with peak ``fat_fraction``. The input phase is kept.
"""

# ruff: noqa: N803 -- b0_T is the scanner profile field name
from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _generator
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import smooth_random_field
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.digital_twin_simulator import (
    DEFAULT_CHI_PPM,
    chemical_shift_pixels,
)

__all__ = [
    "chemical_shift_phase_condition",
    "chemical_shift_phase_volume",
    "degrade_chemical_shift_phase",
    "fat_offset_hz",
    "swept_te_ms",
]


def fat_offset_hz(b0_T: float, sigma_ppm: float | None = None) -> float:
    """Fat frequency relative to water, ``-sigma gamma/2pi B0`` (Hz), via the one gamma owner."""
    sigma = DEFAULT_CHI_PPM if sigma_ppm is None else sigma_ppm
    return -chemical_shift_pixels(b0_T, 1.0, sigma)


def swept_te_ms(theta: float, b0_T: float, sigma_ppm: float | None = None) -> float:
    """``TE_in + theta (TE_opp - TE_in)`` in ms, ``TE_in = 1/|df|`` and ``TE_opp = 1.5/|df|``."""
    te_in = 1.0 / abs(fat_offset_hz(b0_T, sigma_ppm))
    return 1e3 * te_in * (1.0 + 0.5 * float(theta))


def chemical_shift_phase_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    b0_T: float = 3.0,
    fat_fraction: float = 0.1,
    fat_map: Tensor | None = None,
    sigma_ppm: float | None = None,
) -> Tensor:
    """Voxel-wise fat-water interference of ``x`` ``(B, C, D, H, W)`` in any mode."""
    check_input(x, geometry, "chemical_shift_phase")
    sigma_ppm = DEFAULT_CHI_PPM if sigma_ppm is None else sigma_ppm
    if not (b0_T > 0 and sigma_ppm > 0):
        raise ValueError(
            f"chemical_shift_phase: b0_T and sigma_ppm must be > 0; got {b0_T}, {sigma_ppm}"
        )
    if not 0.0 <= fat_fraction <= 1.0:
        raise ValueError(
            f"chemical_shift_phase: fat_fraction must be in [0, 1]; got {fat_fraction}"
        )
    if float(theta) <= 0.0:
        return x
    b, _, d, h, w = x.shape
    dt = x.real.dtype
    if fat_map is None:
        field = smooth_random_field(
            b, (d, h, w), generator=_generator(x.device, seed), device=x.device, dtype=dt
        )
        lo = field.amin(dim=(1, 2, 3, 4), keepdim=True)
        hi = field.amax(dim=(1, 2, 3, 4), keepdim=True)
        f = fat_fraction * (field - lo) / (hi - lo).clamp_min(torch.finfo(dt).tiny)
    else:
        if fat_map.dim() != 5 or fat_map.shape[1] != 1 or fat_map.shape[0] not in (1, b):
            raise ValueError(
                f"chemical_shift_phase: fat_map must be (B or 1, 1, D, H, W); got {tuple(fat_map.shape)}"
            )
        if tuple(fat_map.shape[-3:]) != (d, h, w):
            raise ValueError(
                f"chemical_shift_phase: fat_map {tuple(fat_map.shape[-3:])} != {(d, h, w)}"
            )
        f = fat_map.to(device=x.device, dtype=dt)
        # One host sync, only for a caller-supplied map: an out-of-range fraction must raise.
        if bool((f.amin() < 0) | (f.amax() > 1)):
            raise ValueError("chemical_shift_phase: fat_map must lie in [0, 1]")
    phi = (
        2.0 * math.pi * fat_offset_hz(b0_T, sigma_ppm) * swept_te_ms(theta, b0_T, sigma_ppm) * 1e-3
    )
    # |(1 - f) + f e^{i phi}| by hypot: stable where the two terms cancel (opposed phase).
    gain = torch.hypot(1.0 - f + f * math.cos(phi), f * math.sin(phi))
    return x * gain


def degrade_chemical_shift_phase(x: Tensor, theta: float, seed: int = 0, **kwargs) -> Tensor:
    """2-D kernel: ``x`` ``(B, C, H, W)`` as a one-slice volume."""
    fat_map = kwargs.pop("fat_map", None)
    if fat_map is not None:
        kwargs["fat_map"] = fat_map.unsqueeze(2)
    return run_as_volume(chemical_shift_phase_volume, x, theta, seed, None, **kwargs)


def chemical_shift_phase_condition(
    profile, theta: float, geometry: AcquisitionGeometry, shape
) -> dict:
    """The profile's B0, fat fraction and fat-water shift ``chi_ppm`` (as ``chemical_shift`` reads it)."""
    return {
        "b0_T": profile.b0_T,
        "fat_fraction": profile.fat_fraction,
        "sigma_ppm": profile.chi_ppm,
    }
