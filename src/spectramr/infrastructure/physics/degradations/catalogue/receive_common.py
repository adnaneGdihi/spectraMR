"""Shared plumbing of the receive-chain axes: mode encoding, acquisition-ordered lines, conditions.

Every axis is written once on the canonical ``(B, C, D, H, W)`` layout. A 4-D slice gains a unit
slice axis, so SLICE_2D, MULTISLICE_2D and VOLUME_3D share one implementation whose encoded dims
(2-D k-space per slice, or one 3-D k-space) and line timing follow the acquisition mode.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    resolve_geometry,
)
from spectramr.infrastructure.physics.per_sample_rng import per_sample_normal_pair

__all__ = [
    "ENCODED_DIMS",
    "acquisition_ranks",
    "check_n_coils",
    "coil_covariance_condition",
    "complex_white",
    "control_generator",
    "n_coils_condition",
    "positive",
    "receive_op",
    "subject_peak",
    "two_d",
]

#: k-space axes encoded in each mode: a 2-D k-space per slice, or one 3-D k-space.
ENCODED_DIMS: dict[AcquisitionMode, tuple[int, ...]] = {
    AcquisitionMode.SLICE_2D: (-2, -1),
    AcquisitionMode.MULTISLICE_2D: (-2, -1),
    AcquisitionMode.VOLUME_3D: (-3, -2, -1),
}

_CONTROL_OFFSET = 0x5EED


def receive_op(core: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Adapt ``core(x5, theta, seed, geometry, dims, **kw)`` to the mode-impl signature."""

    @functools.wraps(core)
    def impl(x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, **kw) -> Tensor:
        if x.dim() not in (4, 5) or (x.dim() == 4) != (geometry.mode is AcquisitionMode.SLICE_2D):
            raise ValueError(
                f"{core.__name__}: {x.dim()}-D input does not match acquisition mode {geometry.mode}"
            )
        x5 = x.unsqueeze(2) if x.dim() == 4 else x
        out = core(x5, float(theta), int(seed), geometry, ENCODED_DIMS[geometry.mode], **kw)
        return out.squeeze(2) if x.dim() == 4 else out

    return impl


def two_d(impl: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """The bare 2-D kernel ``kernel(x, theta, seed=0, **kw)`` of a mode implementation."""

    def kernel(x: Tensor, theta: float, seed: int = 0, **kw) -> Tensor:
        return impl(x, theta, seed, resolve_geometry(x, None), **kw)

    kernel.__name__ = f"{impl.__name__}_2d"
    return kernel


def control_generator(device: torch.device, seed: int) -> torch.Generator:
    """A second stream for the artefact's control draws (block start, step time, dropped coils).

    Kept apart from the noise stream so a sample's noise does not depend on how many control
    values the batch drew first.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + _CONTROL_OFFSET)
    return g


def acquisition_ranks(
    geometry: AcquisitionGeometry, d: int, h: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``(rank, tau)`` of every ``(slice or kz, ky)`` line, both ``[D, H]``.

    ``rank`` is the line's position in acquisition order (0 = first); ``tau`` its acquisition
    time as a fraction of the scan, in ``[0, 1)``. Timing comes from ``line_times`` in every mode.
    """
    times = line_times(geometry, d, h, tr_s=1.0)
    order = torch.argsort(times.flatten(), stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel())
    return rank.view(d, h).to(device), (times / (times.max() + 1.0)).to(device)


def complex_white(shape: tuple[int, ...], g: torch.Generator, k: Tensor) -> Tensor:
    """Circular complex Gaussian ``w`` of shape ``shape`` with ``E|w|^2 = 1``, one draw per sample."""
    re, im = per_sample_normal_pair(*shape, generator=g, device=k.device, dtype=k.real.dtype)
    return torch.complex(re, im) * (0.5**0.5)


def subject_peak(k: Tensor) -> Tensor:
    """Peak ``|Re|`` or ``|Im|`` of each subject's k-space over coils and every line, ``[B,1,1,1,1]``."""
    peak = torch.maximum(k.real.abs().amax(dim=(1, 2, 3, 4)), k.imag.abs().amax(dim=(1, 2, 3, 4)))
    return peak.clamp(min=torch.finfo(peak.dtype).tiny).view(-1, 1, 1, 1, 1)


def positive(name: str, value: float) -> float:
    """``value`` as a float, raising unless it is finite and strictly positive."""
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return v


def check_n_coils(x: Tensor, n_coils: int | None) -> None:
    """Raise when the profile's coil count disagrees with the input's channel count."""
    if n_coils is not None and x.shape[1] != n_coils:
        raise ValueError(
            f"the scanner profile declares n_coils={n_coils} but the input has {x.shape[1]} channels"
        )


def n_coils_condition(
    profile, theta: float, geometry: AcquisitionGeometry, shape: tuple[int, ...]
) -> dict[str, object]:
    """The profile's coil count, which the kernel checks against the input."""
    return {"n_coils": profile.n_coils}


def coil_covariance_condition(
    profile, theta: float, geometry: AcquisitionGeometry, shape: tuple[int, ...]
) -> dict[str, object]:
    """The profile's coil count and, when it has one, its inter-coil noise covariance.

    A profile without a covariance leaves the key out so the caller may supply one; with neither,
    the kernel uses its declared synthetic fallback.
    """
    cond: dict[str, object] = {"n_coils": profile.n_coils}
    if profile.coil_covariance is not None:
        cond["coil_covariance"] = profile.coil_covariance
    return cond
