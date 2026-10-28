"""Native readout-motion schedules: translation_drift (D29), sudden_motion (D30), random_motion (D31).

Each keeps its declared readout displacement (Ruling 32) and makes it a function of each line's
acquisition time from ``acquisition_timing.line_times``: lines are ``(slice, ky)`` for a
sequential multislice stack (each slice its own 2-D k-space) and ``(kz, ky)`` for a 3-D volume.
The shift of line ``l`` is the exact Fourier phase ``exp(-2 pi i f_x dx_l)`` along kx. Only the
order of the times matters here; they use the 2-D motion kernels' TR of 0.5 s.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import check_volume
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "ENCODING",
    "ENCODING_DIMS",
    "MOTION_TR_S",
    "random_motion_volume",
    "readout_line_shift",
    "sudden_motion_volume",
    "translation_drift_volume",
]

# The 2-D motion kernels' TR (pulsatile_motion's tr_s).
MOTION_TR_S = 0.5
# A multislice stack is D 2-D k-spaces (lines (slice, ky)); a volume is one 3-D k-space; a 2-D
# slice is a one-slice stack.
ENCODING = {
    AcquisitionMode.SLICE_2D: "slice",
    AcquisitionMode.MULTISLICE_2D: "slice",
    AcquisitionMode.VOLUME_3D: "volume",
}
ENCODING_DIMS = {
    AcquisitionMode.SLICE_2D: (-2, -1),
    AcquisitionMode.MULTISLICE_2D: (-2, -1),
    AcquisitionMode.VOLUME_3D: (-3, -2, -1),
}


def readout_line_shift(x: Tensor, dx: Tensor, geometry: AcquisitionGeometry) -> Tensor:
    """Shift line ``(d, ky)`` by ``dx[b, d, ky]`` px along readout, ``dx`` of shape ``[B or 1, D, H]``."""
    dims = ENCODING_DIMS[geometry.mode]
    k = fftnc(x, dims=dims)
    rdt = x.real.dtype
    fx = centred_freqs(x.shape[-1], x.device, dtype=rdt).view(1, 1, 1, 1, -1)
    d = dx.to(device=x.device, dtype=rdt).unsqueeze(1).unsqueeze(-1)
    return ifftnc(k * torch.exp(-2j * math.pi * fx * d), dims=dims)


def _times(x: Tensor, geometry: AcquisitionGeometry) -> Tensor:
    return line_times(geometry, x.shape[-3], x.shape[-2], tr_s=MOTION_TR_S).to(torch.float64)


def translation_drift_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, max_drift_px: float = 10.0
) -> Tensor:
    """D29: ``dx = theta * max_drift_px * (2 tau - 1)``, ``tau`` the line's normalised time."""
    check_volume(x, geometry, "translation_drift")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    t = _times(x, geometry)
    tau = (t - t.min()) / max(float(t.max() - t.min()), 1e-30)
    dx = float(theta) * max_drift_px * (2.0 * tau - 1.0)
    return readout_line_shift(x, dx[None], geometry)


def sudden_motion_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, max_shift_px: float = 12.0
) -> Tensor:
    """D30: one step of ``theta * max_shift_px`` at the median acquisition time (Ruling 33).

    Lines acquired at ``t >= t_median`` are shifted, earlier ones are not. ``t_median`` is the
    time of line ``n // 2`` in acquisition order, which in 2-D linear order is the old
    centre-line time ``ky = H // 2``; in centric 3-D order the step leaves a genuine ghost.
    """
    check_volume(x, geometry, "sudden_motion")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    t = _times(x, geometry)
    t_median = t.flatten().sort().values[t.numel() // 2]
    dx = (t >= t_median).to(torch.float64) * (float(theta) * max_shift_px)
    return readout_line_shift(x, dx[None], geometry)


def random_motion_volume(
    x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, max_shift_px: float = 3.0
) -> Tensor:
    """D31: i.i.d. ``U(-1, 1) * theta * max_shift_px`` per ``(slice or kz, ky)`` line, per sample."""
    check_volume(x, geometry, "random_motion")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    b, _, d, h, _ = x.shape
    u = per_sample_rand(b, d, h, generator=_generator(x.device, seed), device=x.device)
    return readout_line_shift(x, (u * 2.0 - 1.0) * (float(theta) * max_shift_px), geometry)
