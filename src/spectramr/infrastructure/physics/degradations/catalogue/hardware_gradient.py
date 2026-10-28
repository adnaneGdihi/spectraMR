"""Gradient-hardware axes: ``gradient_scaling_drift`` (FOV calibration) and ``mechanical_vibration``."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.hardware_common import (
    shift_lines,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    ENCODING,
    ENCODING_DIMS,
)
from spectramr.infrastructure.physics.fft_ops import ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "GRADIENT_SCALE_MAX",
    "VIBRATION_MAX_MM",
    "gradient_scaling_drift_2d",
    "gradient_scaling_drift_volume",
    "mechanical_vibration_2d",
    "mechanical_vibration_volume",
]

# Declared severity ends: peak |epsilon| reached at the last line, and the vibration amplitude (mm).
GRADIENT_SCALE_MAX = 0.02
VIBRATION_MAX_MM = 1.0
# Quantisation of the drift into rendered scale states (as rigid_motion's 8 shots).
N_SCALE_STATES = 8
# Vibration repetition time when no profile sets one: 10 ms keeps 20 Hz un-aliased (0.2 cycle/line).
DEFAULT_VIBRATION_TR_S = 0.010
_VOXEL_INDEX = {"kz": 0, "pe": 1, "ro": 2}
_VIBRATION_AXES = {
    AcquisitionMode.SLICE_2D: ("pe", "ro"),
    AcquisitionMode.MULTISLICE_2D: ("pe", "ro"),
    AcquisitionMode.VOLUME_3D: ("kz", "pe", "ro"),
}


def gradient_scaling_drift_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    max_scale: float = GRADIENT_SCALE_MAX,
    static_fraction: float = 0.5,
    n_states: int = N_SCALE_STATES,
) -> Tensor:
    """Gradient calibration error: FOV scale ``1 + eps_a(t)`` per axis, static plus linear drift.

    ``eps_a(tau) = theta max_scale s_a (f + (1 - f) tau)`` with a seeded sign ``s_a`` per axis and
    sample, ``f`` the static fraction and ``tau`` the line's normalised acquisition time; the
    peak ``|eps|`` is ``theta max_scale`` at the last line. A gradient too weak by ``1 / (1 + eps)``
    scales the FOV by ``1 + eps`` (Bernstein et al., Handbook of MRI Pulse Sequences, 2004,
    10.3), so the object is reconstructed as the pull-back ``out(r) = in((1 + eps) r)`` about the
    centre ``N // 2``; the sign is seeded, so both directions occur. The drift is rendered in
    ``n_states`` scale states, each state's lines gathered into k-space (as in rigid motion).
    Multislice scales the in-plane axes only.
    """
    check_input(x, geometry, "gradient_scaling_drift")
    if (
        not max_scale >= 0
        or not 0.0 <= static_fraction <= 1.0
        or not isinstance(n_states, int)
        or n_states < 1
    ):
        raise ValueError(
            f"need max_scale >= 0, static_fraction in [0, 1], n_states >= 1; got "
            f"{max_scale}, {static_fraction}, {n_states}"
        )
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    b, _, d, h, w = x.shape
    rdt = x.real.dtype
    u = per_sample_rand(b, 3, generator=_generator(x.device, seed), device=x.device, dtype=rdt)
    sign = (u >= 0.5).to(rdt) * 2.0 - 1.0  # +/-1 per (sample, axis); kz only in 3-D
    kz_on = float(geometry.mode is AcquisitionMode.VOLUME_3D)
    sign = torch.cat([sign[:, :1] * kz_on, sign[:, 1:]], dim=1)
    n = 1 if static_fraction >= 1.0 else n_states
    shots = shot_index(line_times(geometry, d, h, tr_s=1.0), n)
    coords = [(torch.arange(s, device=x.device) - s // 2).to(rdt) for s in (d, h, w)]
    views = ((1, d, 1, 1), (1, 1, h, 1), (1, 1, 1, w))

    def render(p: int) -> Tensor:
        tau = p / (n - 1) if n > 1 else 0.0
        eps = float(theta) * max_scale * (static_fraction + (1.0 - static_fraction) * tau) * sign
        disp = torch.stack(
            [
                (eps[:, a].view(b, 1, 1, 1) * coords[a].view(views[a])).expand(b, d, h, w)
                for a in range(3)
            ],
            dim=1,
        )
        return warp_volume(x, disp, padding_mode="zeros")

    k = assemble_from_states(n, shots, render, encoding=ENCODING[geometry.mode])
    return ifftnc(k, dims=ENCODING_DIMS[geometry.mode])


def mechanical_vibration_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    max_amplitude_mm: float = VIBRATION_MAX_MM,
    frequency_hz: float = 20.0,
    tr_s: float = DEFAULT_VIBRATION_TR_S,
    axis: str | None = None,
) -> Tensor:
    """Sinusoidal table / coil vibration ``d(t) = A sin(2 pi f_v t + phi)`` along a seeded axis.

    ``A = theta max_amplitude_mm`` is converted to voxels of the axis (1 mm voxels without
    ``voxel_mm``); line ``(d, ky)`` at time ``t`` from ``acquisition_timing`` (repetition time
    ``tr_s``) gets the shift-theorem phase ``exp(-2 pi i f_axis d(t))``. The line phase
    ``exp(-i z sin(a))`` expands by Jacobi-Anger into sidebands ``J_m(z)`` at
    ``m * frac_fold(f_v TR) * N_pe`` pixels along phase-encode (consecutive ky lines), with
    ``z = 2 pi f A / voxel`` for an object frequency ``f`` along the axis (Abramowitz & Stegun
    9.1.41). Lines are TR apart, so the vibration is sampled stroboscopically and aliases:
    ``frac_fold(u)`` is ``|u - round(u)|`` in [0, 1/2], and an integer ``f_v TR`` leaves no
    ghost, only the static displacement ``A sin(phi)``. The axis (``pe`` / ``ro``, plus ``kz``
    in 3-D) and the phase ``phi`` are drawn per sample unless ``axis`` fixes the axis.
    """
    check_input(x, geometry, "mechanical_vibration")
    names = _VIBRATION_AXES[geometry.mode]
    if axis is not None and axis not in names:
        raise ValueError(f"axis must be one of {names} in {geometry.mode.value}, got {axis!r}")
    if not max_amplitude_mm >= 0 or not frequency_hz >= 0 or not tr_s > 0:
        raise ValueError(
            f"need max_amplitude_mm >= 0, frequency_hz >= 0, tr_s > 0; got "
            f"{max_amplitude_mm}, {frequency_hz}, {tr_s}"
        )
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    b, _, d, h, _ = x.shape
    rdt = x.real.dtype
    u = per_sample_rand(b, 2, generator=_generator(x.device, seed), device=x.device)
    if axis is None:
        pick = (u[:, 0] * len(names)).floor().long().clamp(max=len(names) - 1)
    else:
        pick = torch.full((b,), names.index(axis), device=x.device)
    t = line_times(geometry, d, h, tr_s=tr_s).to(device=x.device, dtype=torch.float64)
    phase = 2.0 * math.pi * u[:, 1].to(torch.float64)
    wave = torch.sin(2.0 * math.pi * frequency_hz * t[None] + phase.view(b, 1, 1))
    vox = voxel_mm_or_unit(geometry.voxel_mm)
    shifts = {
        name: (
            (pick == a).view(b, 1, 1)
            * (float(theta) * max_amplitude_mm / vox[_VOXEL_INDEX[name]])
            * wave
        ).to(rdt)
        for a, name in enumerate(names)
    }
    return shift_lines(x, geometry, **shifts)


def gradient_scaling_drift_2d(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kw: Any
) -> Tensor:
    """2-D ``gradient_scaling_drift`` on ``(B, C, H, W)``: a one-slice ``SLICE_2D`` volume."""
    return run_as_volume(gradient_scaling_drift_volume, x, theta, seed, voxel_mm, **kw)


def mechanical_vibration_2d(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kw: Any
) -> Tensor:
    """2-D ``mechanical_vibration`` on ``(B, C, H, W)``: a one-slice ``SLICE_2D`` volume."""
    return run_as_volume(mechanical_vibration_volume, x, theta, seed, voxel_mm, **kw)
