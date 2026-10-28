"""Native multislice and 3-D pulsatile_motion (D7): cardiac phase per line from its acquisition time.

Each ``(slice, ky)`` (multislice) or ``(kz, ky)`` (3-D) line sees the cardiac phase at its own
time ``t`` from ``acquisition_timing.line_times``, quantised into the 2-D kernel's 16 phase bins.
One pose is rendered per used bin and streamed through ``assemble_from_states``, so ghosts fall
along every phase-encode axis of the encoding. The deformation is the 2-D radial envelope in
3-D, isotropic in mm (Ruling 30): peak ``|d| = theta * max_amplitude_px * voxel_mm[1]`` mm.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import mm_to_px, warp_volume
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import (
    check_volume,
    field_coords,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    ENCODING,
    ENCODING_DIMS,
)
from spectramr.infrastructure.physics.fft_ops import ifftnc

__all__ = ["cardiac_bins", "pulsatile_volume", "pulsation_field_mm"]

_Voxel = tuple[float, float, float]


def pulsation_field_mm(
    shape: tuple[int, int, int],
    voxel_mm: _Voxel,
    peak_mm: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[1, 3, D, H, W]`` ``(dz, dy, dx)`` radial field in mm at full pulse, peak ``|d| = peak_mm``.

    ``d = peak_mm * exp(-r^2 / 0.5) / (0.5 e^-0.5) * r_vec``, with ``r_vec`` the position in mm
    over the largest half-extent, so ``|d| = peak_mm`` on the sphere ``r = 0.5``.
    """
    z, y, x = field_coords(shape, voxel_mm, device, dtype)
    envelope = torch.exp(-(z**2 + y**2 + x**2) / 0.5) / (0.5 * math.exp(-0.5))
    return (peak_mm * envelope * torch.stack([z, y, x])).unsqueeze(0)


def cardiac_bins(times_s: Tensor, base_freq_hz: float, n_phase_bins: int, seed: int) -> Tensor:
    """Cardiac phase bin of every line, ``[D, H]`` long on the CPU.

    The phase origin is one draw per call from a CPU generator (as the 2-D kernel draws one per
    call), so the schedule is a static CPU tensor and needs no device sync.
    """
    phase0 = torch.rand(1, generator=_generator(torch.device("cpu"), seed), dtype=torch.float64)
    phase = (2.0 * math.pi * base_freq_hz * times_s.to(torch.float64) + phase0) % (2.0 * math.pi)
    return (phase / (2.0 * math.pi) * n_phase_bins).long().clamp(0, n_phase_bins - 1)


def pulsatile_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    max_amplitude_px: float = 9.0,
    base_freq_hz: float = 1.2,
    tr_s: float = 0.5,
    n_phase_bins: int = 16,
) -> Tensor:
    """D7 in multislice / 3-D: pose of bin ``p`` = ``sin(bin centre) * pulsation_field``."""
    check_volume(x, geometry, "pulsatile_motion")
    if n_phase_bins < 1:
        raise ValueError(f"pulsatile_motion: n_phase_bins must be >= 1; got {n_phase_bins}")
    if max_amplitude_px < 0:
        raise ValueError(f"pulsatile_motion: max_amplitude_px must be >= 0; got {max_amplitude_px}")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    d, h = x.shape[-3], x.shape[-2]
    bins = cardiac_bins(line_times(geometry, d, h, tr_s=tr_s), base_freq_hz, n_phase_bins, seed)
    # Only the bins some line falls in are rendered.
    used, state = torch.unique(bins, return_inverse=True)
    pulse = [math.sin((p + 0.5) * 2.0 * math.pi / n_phase_bins) for p in used.tolist()]

    vox = voxel_mm_or_unit(geometry.voxel_mm)
    peak_mm = float(theta) * max_amplitude_px * vox[1]
    unit = mm_to_px(
        pulsation_field_mm(tuple(x.shape[-3:]), vox, peak_mm, x.device, x.real.dtype), vox
    )
    batch = (x.shape[0], -1, -1, -1, -1)
    k = assemble_from_states(
        len(pulse),
        state.view(d, h),
        lambda s: warp_volume(x, (pulse[s] * unit).expand(*batch)),
        encoding=ENCODING[geometry.mode],
    )
    return ifftnc(k, dims=ENCODING_DIMS[geometry.mode])
