"""respiratory_motion: free-breathing SI translation and AP chest-wall scaling, per acquired line.

Each line sees the breathing state ``u(t)`` in ``[0, 1]`` at its own acquisition time ``t``
(``acquisition_timing.line_times``): relative to the reference state ``u_ref`` the object is
scaled about the image centre along AP by ``1 + s (u - u_ref)`` and translated along SI by
``A (u - u_ref)`` mm, so ``A`` and ``s`` are the peak-to-peak excursions. The default reference
is the time-averaged state (the position a free-breathing image depicts, as ``translation_drift``
centres its ramp); ``reference="exhale"`` takes end-expiration (``u = 0``). A periodic ``u`` sampled once per TR modulates
k-space periodically along the line index, which puts ghosts along PE at multiples of
``N_pe TR / T_resp`` px (Wood & Henkelman, "MR image artifacts from periodic motion", Med Phys
12:143, 1985).

A translation along an encoded axis is the exact per-line Fourier phase ``exp(-2 pi i f d)``.
The scaling, and a translation through a multislice stack's (excited, not encoded) slice axis,
are rendered on ``n_states`` quantised levels of ``u`` and streamed through
``assemble_from_states``, so memory stays O(volume).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    AXIS_INDEX,
    MOTION_TR_S,
    check_input,
    encoded_axes,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.degradations.volume.motion_rigid import line_phase
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    ENCODING,
    ENCODING_DIMS,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc
from spectramr.infrastructure.physics.motion_simulation import (
    respiratory_trace,
    respiratory_trace_peak,
)

__all__ = [
    "PEAK_AP_SCALE",
    "PEAK_SI_MM",
    "REFERENCES",
    "TRACES",
    "breathing_state",
    "degrade_respiratory_motion",
    "resolve_axes",
    "respiratory_motion_volume",
]

_M = AcquisitionMode
#: Peak SI excursion at theta = 1: free-breathing diaphragm motion is 10-20 mm.
PEAK_SI_MM = 20.0
#: Peak relative AP scale at theta = 1: about 4 % chest-wall expansion.
PEAK_AP_SCALE = 0.04
# Normal adult breathing: 15 breaths/min.
_PERIOD_S = 4.0
_N_STATES = 16

_DEFAULT_AXES = {
    _M.SLICE_2D: ("pe", "ro"),
    _M.MULTISLICE_2D: ("slice", "pe"),
    _M.VOLUME_3D: ("slice", "pe"),
}


def _breathing(t: Tensor, period_s: float) -> Tensor:
    # One breath per period (the FFD simulator's default trace has two).
    return respiratory_trace(t, period_s, breaths_per_cycle=1) / respiratory_trace_peak(
        breaths_per_cycle=1
    )


def _sinusoid(t: Tensor, period_s: float) -> Tensor:
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * t / period_s))


def _constant(t: Tensor, period_s: float) -> Tensor:
    return torch.ones_like(t)


#: Breathing state ``u(t)`` in ``[0, 1]`` by name; ``breathing`` is the hysteresis trace.
TRACES: dict[str, Callable[[Tensor, float], Tensor]] = {
    "breathing": _breathing,
    "sinusoid": _sinusoid,
    "constant": _constant,
}
#: The breathing state the undisplaced object depicts, from the lines' states ``u``.
REFERENCES: dict[str, Callable[[Tensor], float]] = {
    "mean": lambda u: float(u.mean()),
    "exhale": lambda u: 0.0,
}


def resolve_axes(
    mode: AcquisitionMode, si_axis: str | None, ap_axis: str | None
) -> tuple[str, str]:
    """The (SI, AP) axis names for ``mode``, defaulted per mode and validated."""
    default_si, default_ap = _DEFAULT_AXES[mode]
    si = default_si if si_axis is None else si_axis
    ap = default_ap if ap_axis is None else ap_axis
    for key, value in (("si_axis", si), ("ap_axis", ap)):
        if value not in AXIS_INDEX:
            raise ValueError(
                f"respiratory_motion: {key} must be one of {sorted(AXIS_INDEX)}; got {value!r}"
            )
        if mode is _M.SLICE_2D and value == "slice":
            raise ValueError(
                f"respiratory_motion: {key}='slice' needs a volume; a 2-D slice has none"
            )
    if si == ap:
        raise ValueError(f"respiratory_motion: si_axis and ap_axis must be distinct; both {si!r}")
    return si, ap


def breathing_state(times_s: Tensor, trace: str, period_s: float, seed: int) -> Tensor:
    """``u`` of every line, ``[D, H]`` float64 on the CPU, from a seeded phase origin.

    The origin is one CPU draw per call (as ``pulsatile_motion``'s cardiac phase), so the
    schedule is static, shared by the batch and needs no device sync.
    """
    t0 = torch.rand(1, generator=_generator(torch.device("cpu"), seed), dtype=torch.float64)
    return TRACES[trace](times_s.to(torch.float64) + t0 * period_s, period_s).clamp(0.0, 1.0)


def _validate(trace, reference, peak_si_mm, peak_ap_scale, period_s, tr_s, n_states) -> None:
    if trace not in TRACES:
        raise ValueError(
            f"respiratory_motion: trace must be one of {sorted(TRACES)}; got {trace!r}"
        )
    if reference not in REFERENCES:
        raise ValueError(
            f"respiratory_motion: reference must be one of {sorted(REFERENCES)}; got {reference!r}"
        )
    if not peak_si_mm >= 0:
        raise ValueError(f"respiratory_motion: peak_si_mm must be >= 0; got {peak_si_mm}")
    if not 0 <= peak_ap_scale < 1:
        raise ValueError(
            f"respiratory_motion: peak_ap_scale must be in [0, 1); got {peak_ap_scale}"
        )
    if not period_s > 0:
        raise ValueError(f"respiratory_motion: period_s must be > 0; got {period_s}")
    if not tr_s > 0:
        raise ValueError(f"respiratory_motion: tr_s must be > 0; got {tr_s}")
    if not (isinstance(n_states, int) and n_states >= 2):
        raise ValueError(f"respiratory_motion: n_states must be an int >= 2; got {n_states!r}")


def _render(x: Tensor, level: float, ap: str, scale: float, slice_shift_px: float) -> Tensor:
    """The object ``level`` from the reference: AP scale ``1 + scale level``, slice shift if any."""
    if level == 0.0:
        return x
    b, _, *grid = x.shape
    disp = torch.zeros((1, 3, *grid), device=x.device, dtype=x.real.dtype)
    a = AXIS_INDEX[ap]
    view = [1, 1, 1]
    view[a] = grid[a]
    r = torch.arange(grid[a], device=x.device, dtype=x.real.dtype) - grid[a] // 2
    # Pull-back of an expansion about the FFT centre: out(r) = x(c + (r - c) / (1 + s)).
    disp[:, a] = (r * (1.0 / (1.0 + scale * level) - 1.0)).view(view)
    # A shift of +d slices samples x(r - d); slices moved past the stack receive no signal.
    disp[:, 0] = disp[:, 0] - level * slice_shift_px
    return warp_volume(x, disp.expand(b, -1, -1, -1, -1), padding_mode="zeros")


def respiratory_motion_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    si_axis: str | None = None,
    ap_axis: str | None = None,
    peak_si_mm: float = PEAK_SI_MM,
    peak_ap_scale: float = PEAK_AP_SCALE,
    trace: str = "breathing",
    reference: str = "mean",
    period_s: float = _PERIOD_S,
    tr_s: float = MOTION_TR_S,
    n_states: int = _N_STATES,
) -> Tensor:
    """Respiratory motion of ``x`` ``(B, C, D, H, W)`` in any mode (``D == 1`` for a slice)."""
    check_input(x, geometry, "respiratory_motion")
    mode = geometry.mode
    si, ap = resolve_axes(mode, si_axis, ap_axis)
    _validate(trace, reference, peak_si_mm, peak_ap_scale, period_s, tr_s, n_states)
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    d, h = x.shape[-3], x.shape[-2]
    u = breathing_state(line_times(geometry, d, h, tr_s=tr_s), trace, period_s, seed)
    u_ref = REFERENCES[reference](u)
    vox = voxel_mm_or_unit(geometry.voxel_mm)
    shift_px = float(theta) * peak_si_mm / vox[AXIS_INDEX[si]]
    scale = float(theta) * peak_ap_scale
    encoded = si in encoded_axes(mode)
    dims = ENCODING_DIMS[mode]
    if scale > 0.0 or not encoded:
        index = torch.round(u * (n_states - 1)).long()
        used, state = torch.unique(index, return_inverse=True)
        levels = (used.to(torch.float64) / (n_states - 1) - u_ref).tolist()
        slice_shift = 0.0 if encoded else shift_px
        k = assemble_from_states(
            len(levels),
            state.view(d, h),
            lambda p: _render(x, levels[p], ap, scale, slice_shift),
            encoding=ENCODING[mode],
        )
    else:
        k = fftnc(x, dims=dims)
    if encoded and shift_px > 0.0:
        shifts = torch.zeros((1, len(dims), d, h), dtype=torch.float64)
        shifts[:, AXIS_INDEX[si] - (3 - len(dims))] = (u - u_ref) * shift_px
        k = k * line_phase(shifts, x.shape, x.device, x.real.dtype)
    return ifftnc(k, dims=dims)


def degrade_respiratory_motion(
    x: Tensor,
    theta: float,
    seed: int = 0,
    voxel_mm: tuple[float, float, float] | None = None,
    **kwargs,
) -> Tensor:
    """The 2-D kernel: a ``(B, C, H, W)`` slice, lines timed per ``ky`` (1 mm pixels if unknown)."""
    return run_as_volume(respiratory_motion_volume, x, theta, seed, voxel_mm, **kwargs)
