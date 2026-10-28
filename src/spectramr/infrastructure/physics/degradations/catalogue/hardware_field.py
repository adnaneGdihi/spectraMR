"""Hardware field axes: ``b0_drift`` and ``gradient_delay`` (EPI readout-polarity ghost).

Both act per k-space line on the multislice / 3-D line grid ``(slice or kz, ky)``; a 2-D slice is
a one-slice stack. Timing comes from ``acquisition_timing``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    readout_offset_px,
)
from spectramr.infrastructure.physics.degradations.catalogue.hardware_common import (
    line_taus,
    shift_lines,
    shot_taus,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    ENCODING_DIMS,
    readout_line_shift,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

__all__ = [
    "B0_DRIFT_MAX_HZ",
    "DEFAULT_BW_HZ_PER_PX",
    "GRADIENT_DELAY_MAX_DWELL",
    "b0_drift_2d",
    "b0_drift_volume",
    "gradient_delay_2d",
    "gradient_delay_volume",
    "off_resonance_lines",
]

# Declared severity ends: the offset reached at the last line, and the echo displacement (dwell).
B0_DRIFT_MAX_HZ = 50.0
GRADIENT_DELAY_MAX_DWELL = 0.5
# Used when no scanner profile sets the readout bandwidth (Hz per pixel).
DEFAULT_BW_HZ_PER_PX = 250.0
_SEQUENCES = ("spin_warp", "epi")


def off_resonance_lines(
    x: Tensor,
    df_hz: Tensor,
    geometry: AcquisitionGeometry,
    *,
    bw_hz_per_px: float,
    sequence: str,
    esp_s: float | None = None,
) -> Tensor:
    """Line ``(d, ky)`` at offset ``df_hz[d, ky]`` (Hz): content moves ``-df / BW_px`` px.

    A spin-warp line at off-resonance ``df`` is displaced ``df / BW_px`` px along the frequency
    axis (Haacke et al., MRI: Physical Principles and Sequence Design, 1999, ch. 15). An EPI
    line is displaced along phase-encode by ``df / BW_pe`` px, ``BW_pe = 1 / (N_pe ESP)``
    (Jezzard & Balaban, MRM 34:65, 1995), i.e. ``df N_pe ESP`` px. The sign is the one of
    :func:`readout_offset_px`, shared with b0_geometric_distortion and metal_artifact.
    """
    if sequence == "spin_warp":
        return readout_line_shift(x, -readout_offset_px(df_hz[None], bw_hz_per_px), geometry)
    if esp_s is None:
        raise ValueError("an EPI B0 drift needs the echo spacing: set profile.esp_ms (or esp_s=)")
    return shift_lines(x, geometry, pe=-df_hz[None] * (x.shape[-2] * esp_s))


def _check_scan(bw_hz_per_px: float, sequence: str, esp_s: float | None) -> None:
    if not bw_hz_per_px > 0:
        raise ValueError(f"bw_hz_per_px must be > 0, got {bw_hz_per_px}")
    if sequence not in _SEQUENCES:
        raise ValueError(f"sequence must be one of {_SEQUENCES}, got {sequence!r}")
    if esp_s is not None and not esp_s > 0:
        raise ValueError(f"esp_s must be > 0, got {esp_s}")
    if sequence == "epi" and esp_s is None:
        raise ValueError("an EPI B0 drift needs the echo spacing: set profile.esp_ms (or esp_s=)")


def b0_drift_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    max_offset_hz: float = B0_DRIFT_MAX_HZ,
    bw_hz_per_px: float = DEFAULT_BW_HZ_PER_PX,
    sequence: str = "spin_warp",
    esp_s: float | None = None,
) -> Tensor:
    """Linear main-field drift ``df(t) = theta max_offset_hz t / T_scan`` (Hz).

    Spin-warp evaluates it per line at the line's acquisition time over that of the last line.
    EPI evaluates it once per shot (the slice, or the kz partition) at the shot's mean line time
    over the last shot's: an echo train lasts tens of ms, so within it the drift is constant and
    the shot is a pure PE shift ``df N_pe ESP``. The offset reaches ``theta * max_offset_hz`` at
    the end of the scan (a linear drift of the main field, e.g. from gradient or shim heating).
    """
    check_input(x, geometry, "b0_drift")
    _check_scan(bw_hz_per_px, sequence, esp_s)
    if not max_offset_hz >= 0:
        raise ValueError(f"max_offset_hz must be >= 0, got {max_offset_hz}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    timing = shot_taus if sequence == "epi" else line_taus
    tau = timing(geometry, x.shape[-3], x.shape[-2], x.device)
    df = float(theta) * max_offset_hz * tau
    return off_resonance_lines(
        x, df, geometry, bw_hz_per_px=bw_hz_per_px, sequence=sequence, esp_s=esp_s
    )


def gradient_delay_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    max_delay_dwell: float = GRADIENT_DELAY_MAX_DWELL,
) -> Tensor:
    """EPI readout-gradient delay ``tau = theta max_delay_dwell`` dwell times (N/2 ghost).

    The odd/even echoes sample k_x at ``k +/- tau/2`` samples; reconstructing on the nominal grid
    multiplies the hybrid ``(ky, x)`` line by ``exp(-/+ i pi tau x / N_ro)`` (odd/even ky index,
    ``x`` centred at ``N_ro // 2``): the shift theorem in k_x (Bruder et al., MRM 23:311, 1992).
    ``tau`` in microseconds is ``tau * dwell``, ``dwell = 1 / (BW_px N_ro)``.
    """
    check_input(x, geometry, "gradient_delay")
    if not max_delay_dwell >= 0:
        raise ValueError(f"max_delay_dwell must be >= 0, got {max_delay_dwell}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    h, w = x.shape[-2:]
    rdt = x.real.dtype
    tau = float(theta) * max_delay_dwell
    sign = (torch.arange(h, device=x.device) % 2 * 2 - 1).to(rdt).view(-1, 1)
    xc = (torch.arange(w, device=x.device) - w // 2).to(rdt).view(1, -1)
    phase = torch.exp(-1j * math.pi * tau * sign * xc / w)
    dims = ENCODING_DIMS[geometry.mode][:-1]
    return ifftnc(fftnc(x, dims=dims) * phase, dims=dims)


def b0_drift_2d(x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kw: Any) -> Tensor:
    """2-D ``b0_drift`` on ``(B, C, H, W)``: a one-slice ``SLICE_2D`` volume."""
    return run_as_volume(b0_drift_volume, x, theta, seed, voxel_mm, **kw)


def gradient_delay_2d(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kw: Any
) -> Tensor:
    """2-D ``gradient_delay`` on ``(B, C, H, W)``: a one-slice ``SLICE_2D`` volume."""
    return run_as_volume(gradient_delay_volume, x, theta, seed, voxel_mm, **kw)
