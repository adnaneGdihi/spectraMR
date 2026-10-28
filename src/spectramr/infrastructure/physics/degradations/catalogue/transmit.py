"""Transmit-field axes: ``b1_transmit`` (smooth B1+ map) and ``dielectric_b1`` (standing wave).

Both produce a B1+ scale map ``kappa(r)`` and hand it to :func:`transmit_signal.apply_transmit`.
``kappa`` is a physical position field, so multislice and 3-D samples of it agree and one 5-D
implementation serves every mode; a 2-D slice runs as a one-slice volume.

``dielectric_b1``: the RF wavelength in tissue is ``lambda = c / (f0 sqrt(eps_r))`` with
``f0 = gamma_bar B0`` (Hoult, JMRI 12:46, 2000; Collins et al., MRM 40:847, 1998), and the
central brightening / ring pattern is modelled as ``1 + A cos(2 pi |r - r0| / lambda)``.
Both seeded maps are referenced to the prescan transmit calibration (mean flip angle over the
object is nominal), so a long wavelength at low field leaves only a vanishing pattern.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _generator
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import (
    mm_grid,
    smooth_random_field,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_signal import (
    DEFAULT_SEQUENCE,
    DEFAULT_T1_MS,
    apply_transmit,
    calibrated_kappa,
    check_signal_args,
    lift_maps,
    object_support,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import field_coords
from spectramr.infrastructure.physics.digital_twin_simulator import GAMMA_BAR_HZ_PER_T

__all__ = [
    "DIELECTRIC_AMPLITUDE",
    "KAPPA_PEAK",
    "b1_transmit_volume",
    "centre_bright_shape",
    "degrade_b1_transmit",
    "degrade_dielectric_b1",
    "dielectric_b1_volume",
    "dielectric_kappa",
    "rf_wavelength_mm",
]

#: Peak |kappa - 1| of the smooth map at theta = 1.
KAPPA_PEAK = 0.4
#: Standing-wave amplitude A at theta = 1.
DIELECTRIC_AMPLITUDE = 0.5
_C_MM_PER_S = 299_792_458.0e3
_DEFAULT_B0_T = 3.0
_DEFAULT_EPS_R = 50.0
#: Share of the centre-bright term in the map shape; the rest is a seeded smooth field.
_CENTRE_WEIGHT = 0.7
_CENTRE_SIGMA = 0.5
_MAPS = ("kappa_map",)


def centre_bright_shape(
    batch: int,
    shape: tuple[int, int, int],
    voxel_mm: tuple[float, float, float] | None,
    seed: int,
    device: torch.device,
) -> Tensor:
    """``[B, 1, D, H, W]`` unit-peak shape: centre-bright (3 T and above) plus a seeded field."""
    z, y, x = field_coords(shape, voxel_mm, device)
    centre = 2.0 * torch.exp(-(x**2 + y**2 + z**2) / (2.0 * _CENTRE_SIGMA**2)) - 1.0
    rand = smooth_random_field(batch, shape, generator=_generator(device, seed), device=device)
    s = _CENTRE_WEIGHT * centre + (1.0 - _CENTRE_WEIGHT) * rand
    return s / s.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-12)


def _signal_kwargs(
    sequence: str, fa_deg: float | None, tr_ms: float | None, t1_ms: Any
) -> dict[str, Any]:
    check_signal_args(sequence, fa_deg, tr_ms, t1_ms)
    return {"sequence": sequence, "fa_deg": fa_deg, "tr_ms": tr_ms, "t1_ms": t1_ms}


def b1_transmit_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    sequence: str = DEFAULT_SEQUENCE,
    fa_deg: float | None = None,
    tr_ms: float | None = None,
    t1_ms: Any = DEFAULT_T1_MS,
    kappa_map: Tensor | None = None,
) -> Tensor:
    """``x`` times the signal ratio of the calibrated ``kappa_full = 1 + 0.4 s(r)``.

    The seeded map is referenced to the transmit calibration (:func:`calibrated_kappa`); a measured
    ``kappa_map`` is already relative to nominal: ``kappa = 1 + theta (kappa_map - 1)``.
    """
    check_input(x, geometry, "b1_transmit")
    signal = _signal_kwargs(sequence, fa_deg, tr_ms, t1_ms)
    if theta <= 0.0:
        return x
    if kappa_map is None:
        s = centre_bright_shape(x.shape[0], tuple(x.shape[-3:]), geometry.voxel_mm, seed, x.device)
        kappa = calibrated_kappa(1.0 + KAPPA_PEAK * s, object_support(x), theta)
    else:
        kappa = 1.0 + theta * (torch.as_tensor(kappa_map, device=x.device) - 1.0)
    return apply_transmit(x, kappa, **signal)


def rf_wavelength_mm(b0_T: float, eps_r: float = _DEFAULT_EPS_R) -> float:  # noqa: N803
    """RF wavelength in tissue in mm: ``c / (gamma_bar B0 sqrt(eps_r))``."""
    if not (b0_T > 0.0 and eps_r > 0.0):
        raise ValueError(f"b0_T and eps_r must be positive, got {b0_T!r}, {eps_r!r}")
    return _C_MM_PER_S / (GAMMA_BAR_HZ_PER_T * b0_T * math.sqrt(eps_r))


def dielectric_kappa(
    shape: tuple[int, int, int],
    voxel_mm: tuple[float, float, float] | None,
    *,
    b0_T: float,  # noqa: N803
    eps_r: float = _DEFAULT_EPS_R,
    amplitude: float = DIELECTRIC_AMPLITUDE,
    device: torch.device | None = None,
) -> Tensor:
    """``[D, H, W]`` standing-wave scale ``1 + A cos(2 pi |r| / lambda)``, ``r`` in mm."""
    z, y, x = mm_grid(shape, voxel_mm_or_unit(voxel_mm), device=device)
    r = torch.sqrt(x**2 + y**2 + z**2)
    return 1.0 + amplitude * torch.cos(2.0 * math.pi * r / rf_wavelength_mm(b0_T, eps_r))


def dielectric_b1_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    b0_T: float = _DEFAULT_B0_T,  # noqa: N803
    eps_r: float = _DEFAULT_EPS_R,
    sequence: str = DEFAULT_SEQUENCE,
    fa_deg: float | None = None,
    tr_ms: float | None = None,
    t1_ms: Any = DEFAULT_T1_MS,
) -> Tensor:
    """Calibrated standing-wave scale of ``1 + 0.5 cos(2 pi |r| / lambda)`` inside the support."""
    check_input(x, geometry, "dielectric_b1")
    signal = _signal_kwargs(sequence, fa_deg, tr_ms, t1_ms)
    rf_wavelength_mm(b0_T, eps_r)  # validates b0_T and eps_r before the early return
    if theta <= 0.0:
        return x
    full = dielectric_kappa(
        tuple(x.shape[-3:]), geometry.voxel_mm, b0_T=b0_T, eps_r=eps_r, device=x.device
    )
    kappa = calibrated_kappa(full, object_support(x), theta)
    return apply_transmit(x, kappa, **signal)


def degrade_b1_transmit(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kwargs: Any
) -> Tensor:
    """2-D ``b1_transmit`` on ``(B, C, H, W)``; ``voxel_mm`` comes from the slice geometry."""
    return run_as_volume(b1_transmit_volume, x, theta, seed, voxel_mm, **lift_maps(kwargs, _MAPS))


def degrade_dielectric_b1(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kwargs: Any
) -> Tensor:
    """2-D ``dielectric_b1`` on ``(B, C, H, W)``; ``voxel_mm`` comes from the slice geometry."""
    return run_as_volume(dielectric_b1_volume, x, theta, seed, voxel_mm, **kwargs)
