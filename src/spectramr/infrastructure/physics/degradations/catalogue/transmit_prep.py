"""Magnetisation-preparation axes: ``imperfect_fat_sat`` (CHESS) and ``imperfect_inversion`` (IR).

Both add a residual that the ideal preparation would have removed, in phase with the local image
(a complex image keeps the phase of the voxel it perturbs; the water-fat phase evolution with TE
is not modelled). Both are position-dependent fields, so one 5-D implementation serves every mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from spectramr.core.quantile import robust_quantile
from spectramr.infrastructure.physics.degradations._common import _generator
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import smooth_random_field
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_signal import (
    lift_maps,
    object_support,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.degradations.spec import MissingInput, Requirement

__all__ = [
    "BAND_HZ",
    "MAX_DB0_HZ",
    "MAX_INEFFICIENCY",
    "degrade_imperfect_fat_sat",
    "degrade_imperfect_inversion",
    "imperfect_fat_sat_volume",
    "imperfect_inversion_volume",
    "inversion_signal",
]

#: Peak |dB0| at theta = 1, and the half-width of the fat saturation band.
MAX_DB0_HZ = 300.0
BAND_HZ = 100.0
#: 1 - kappa of the inversion pulse at theta = 1.
MAX_INEFFICIENCY = 0.3
_DEFAULT_FAT_FRACTION = 0.1  # ScannerProfile's default
_QMAP_KEYS = ("t1", "pd")


def _fat_seed(seed: int) -> int:
    """The seed of the fat map: a fixed odd multiplier keeps it apart from the field-map seed."""
    return (int(seed) * 2_654_435_761 + 1) % (2**31 - 1)


def _in_phase(x: Tensor) -> Tensor:
    """Unit phasor of ``x`` (1 where ``x`` is 0); the sign of ``x`` (1 at 0) for a real image."""
    if x.is_complex():
        return torch.polar(torch.ones_like(x.real), torch.angle(x))
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))


def _level(x: Tensor) -> Tensor:
    """Per sample and channel 99th-percentile magnitude, ``[B, C, 1, 1, 1]``."""
    b, c = x.shape[:2]
    flat = x.detach().abs().reshape(b * c, -1).float()
    return robust_quantile(flat, 0.99, dim=1).clamp(min=1e-12).view(b, c, 1, 1, 1)


def imperfect_fat_sat_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    fat_fraction: float = _DEFAULT_FAT_FRACTION,
    fat_map: Tensor | None = None,
    b0_map_hz: Tensor | None = None,
    band_hz: float = BAND_HZ,
) -> Tensor:
    """``x`` (the water image) plus residual fat ``F (1 - eta)``, ``eta = exp(-(dB0 / w)^2)``.

    ``dB0 = 300 Hz theta s(r)`` for a seeded unit-peak smooth ``s``, or ``theta b0_map_hz``;
    ``F`` is ``fat_map`` or ``fat_fraction`` times the sample's 99th-percentile signal times a
    seeded smooth map in [0, 1] inside the object support (CHESS: Haase et al., J Magn Reson
    67:258, 1986; the Gaussian band is the model of the task, not a pulse simulation).
    """
    check_input(x, geometry, "imperfect_fat_sat")
    if not band_hz > 0.0:
        raise ValueError(f"band_hz must be positive, got {band_hz!r}")
    if not 0.0 <= fat_fraction <= 1.0:
        raise ValueError(f"fat_fraction must be in [0, 1], got {fat_fraction!r}")
    if theta <= 0.0:
        return x
    shape = tuple(x.shape[-3:])
    # Two generators (the second seed derived from the first): each field is drawn sample by sample.
    batch = x.shape[0]
    s_b0 = smooth_random_field(batch, shape, generator=_generator(x.device, seed), device=x.device)
    s_fat = smooth_random_field(
        batch, shape, generator=_generator(x.device, _fat_seed(seed)), device=x.device
    )
    if b0_map_hz is None:
        db0 = theta * MAX_DB0_HZ * s_b0
    else:
        db0 = theta * torch.as_tensor(b0_map_hz, device=x.device)
    if fat_map is None:
        fat = fat_fraction * _level(x) * (0.5 * (1.0 + s_fat)) * object_support(x)
    else:
        fat = torch.as_tensor(fat_map, device=x.device)
    return x + fat * (1.0 - torch.exp(-((db0 / band_hz) ** 2))) * _in_phase(x)


def inversion_signal(kappa: float | Tensor, ti_ms: float, t1_ms: Tensor) -> Tensor:
    """Longitudinal magnetisation at TI after an inversion of efficiency ``kappa``, TR >> T1.

    ``Mz = 1 - (1 + kappa) exp(-TI / T1)`` (Bernstein, Handbook of MRI Pulse Sequences, 2004,
    ch. 6); it vanishes at ``TI = T1 ln(1 + kappa)``.
    """
    return 1.0 - (1.0 + kappa) * torch.exp(-ti_ms / t1_ms)


def _t1_pd(qmaps: Mapping[str, Any] | None) -> tuple[Any, Any]:
    if not isinstance(qmaps, Mapping) or not set(_QMAP_KEYS) <= set(qmaps):
        raise MissingInput("imperfect_inversion", Requirement.QMAPS)
    return qmaps["t1"], qmaps["pd"]


def imperfect_inversion_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    qmaps: Mapping[str, Any] | None = None,
    ti_ms: float | None = None,
) -> Tensor:
    """``x`` (the ideal-inversion IR image) plus ``pd (|Mz(kappa)| - |Mz(1)|)``.

    ``kappa = 1 - 0.3 theta``.

    ``qmaps`` holds ``t1`` (ms) and ``pd`` (the equilibrium image, in the units of ``x``); TI has
    no default. For ``x = pd |Mz(1)|`` the result is exactly ``pd |Mz(kappa)|``, so the null moves
    from ``T1 ln 2`` to ``T1 ln(1 + kappa)`` and no division by the (zero) ideal null is needed.
    """
    check_input(x, geometry, "imperfect_inversion")
    t1, pd = _t1_pd(qmaps)
    if ti_ms is None or not ti_ms > 0.0:
        raise ValueError(f"imperfect_inversion needs a positive ti_ms (no default), got {ti_ms!r}")
    if theta <= 0.0:
        return x
    t1 = torch.as_tensor(t1, device=x.device)
    pd = torch.as_tensor(pd, device=x.device)
    kappa = 1.0 - MAX_INEFFICIENCY * theta
    delta = pd * (inversion_signal(kappa, ti_ms, t1).abs() - inversion_signal(1.0, ti_ms, t1).abs())
    return x + delta * _in_phase(x)


def degrade_imperfect_fat_sat(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kwargs: Any
) -> Tensor:
    """2-D ``imperfect_fat_sat`` on ``(B, C, H, W)``."""
    kw = lift_maps(kwargs, ("fat_map", "b0_map_hz"))
    return run_as_volume(imperfect_fat_sat_volume, x, theta, seed, voxel_mm, **kw)


def degrade_imperfect_inversion(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kwargs: Any
) -> Tensor:
    """2-D ``imperfect_inversion`` on ``(B, C, H, W)``."""
    qmaps = kwargs.get("qmaps")
    if isinstance(qmaps, Mapping):
        kwargs = {**kwargs, "qmaps": lift_maps(qmaps, _QMAP_KEYS)}
    return run_as_volume(imperfect_inversion_volume, x, theta, seed, voxel_mm, **kwargs)
