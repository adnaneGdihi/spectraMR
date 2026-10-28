"""The hardware catalogue family: b0_drift, gradient_delay, gradient_scaling_drift, mechanical_vibration.

Every axis is native in all three modes: a slice is a one-slice stack, and the stack and the
volume time each ``(slice or kz, ky)`` line through ``acquisition_timing``.
"""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.hardware_field import (
    b0_drift_2d,
    b0_drift_volume,
    gradient_delay_2d,
    gradient_delay_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.hardware_gradient import (
    gradient_scaling_drift_2d,
    gradient_scaling_drift_volume,
    mechanical_vibration_2d,
    mechanical_vibration_volume,
)
from spectramr.infrastructure.physics.degradations.conditions import b0_shift_sequence
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import MS, VOL
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import (
    ActivationGate,
    Realisation,
    VoxelSliceKernel,
    entry,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_SLICE = AcquisitionMode.SLICE_2D


def _modes(kernel_2d, volume_impl):
    impls = {_SLICE: VoxelSliceKernel(kernel_2d), MS: volume_impl, VOL: volume_impl}
    return impls, dict.fromkeys(impls, Realisation.NATIVE)


def b0_drift_condition(profile, theta, geometry, shape) -> dict[str, float | str]:
    """The readout bandwidth, the shift model (spin-warp or EPI) and, for EPI, the echo spacing."""
    cond: dict[str, float | str] = {
        "bw_hz_per_px": float(profile.readout_bw_hz_per_px),
        "sequence": b0_shift_sequence(profile.sequence),
    }
    if profile.esp_ms is not None:
        cond["esp_s"] = float(profile.esp_ms) * 1e-3
    return cond


def vibration_condition(profile, theta, geometry, shape) -> dict[str, float]:
    """The repetition time ``tr_ms`` of the profile when it has one."""
    return {} if profile.tr_ms is None else {"tr_s": float(profile.tr_ms) * 1e-3}


_B0_DRIFT = _modes(b0_drift_2d, b0_drift_volume)
_DELAY = _modes(gradient_delay_2d, gradient_delay_volume)
_SCALING = _modes(gradient_scaling_drift_2d, gradient_scaling_drift_volume)
_VIBRATION = _modes(mechanical_vibration_2d, mechanical_vibration_volume)

ENTRIES = (
    entry(
        code="D40",
        name="b0_drift",
        family="field",
        description="Linear main-field drift over the scan",
        severity=SeveritySpec(
            (_P("max_frequency_offset", "Hz", 0.0, 50.0),),
            notes="df(t) = theta 50 Hz t / T_scan, t the line's acquisition time (reaches the "
            "maximum at the last line). A spin-warp line shifts df / BW_px px along readout; an "
            "EPI line shifts df N_pe ESP px along phase-encode (profile.esp_ms, required for "
            "EPI). EPI is evaluated once per shot (slice or kz partition), spin-warp per line. "
            "BW_px = profile.readout_bw_hz_per_px, 250 Hz/px without a profile. The peak shift "
            "is 0.2 px at 250 Hz/px, so this is a weak axis by design.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        implementations=_B0_DRIFT[0],
        realisation=_B0_DRIFT[1],
        requires=frozenset(),
        condition=b0_drift_condition,
        active_when=frozenset(),
    ),
    entry(
        code="D41",
        name="gradient_delay",
        family="field",
        description="EPI readout-gradient delay (N/2 ghost)",
        severity=SeveritySpec(
            (_P("echo_displacement", "dwell", 0.0, 0.5),),
            notes="odd/even echoes sample k_x at +/- tau/2, tau = theta 0.5 dwell; in time tau "
            "= theta 0.5 / (BW_px N_ro) (7.8 us at 250 Hz/px, N_ro = 256; "
            "tau_us = theta 0.5e6 / (BW_px N_ro)). Odd/even is the ky index parity. "
            "Active only for EPI protocols.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        implementations=_DELAY[0],
        realisation=_DELAY[1],
        requires=frozenset(),
        condition=None,
        active_when=frozenset({ActivationGate.SEQUENCE_EPI}),
    ),
    entry(
        code="D42",
        name="gradient_scaling_drift",
        family="geometric",
        description="Gradient calibration error with linear drift (FOV scaling)",
        severity=SeveritySpec(
            (_P("peak_fov_scale_error", "fraction", 0.0, 0.02),),
            notes="eps_a(t) = theta 2% s_a (0.5 + 0.5 t / T_scan), s_a = +/-1 seeded per axis "
            "and sample; the peak |eps| is reached at the last line. In-plane axes only in 2-D "
            "and multislice, all three in 3-D. The drift is rendered in 8 scale states.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        implementations=_SCALING[0],
        realisation=_SCALING[1],
        requires=frozenset(),
        condition=None,
        active_when=frozenset(),
    ),
    entry(
        code="D43",
        name="mechanical_vibration",
        family="motion",
        description="Table or gradient-coil vibration (PE sidebands)",
        severity=SeveritySpec(
            (_P("vibration_amplitude", "mm", 0.0, 1.0),),
            notes="d(t) = A sin(2 pi f_v t + phi), f_v = 20 Hz, A = theta 1 mm, along a seeded "
            "axis (1 mm voxels without voxel_mm). Lines are TR apart, so the sampling is "
            "stroboscopic: sidebands sit at +/- frac_fold(f_v TR) N_pe px along phase-encode "
            "(consecutive ky lines; frac_fold(u) = |u - round(u)|), and an integer f_v TR "
            "leaves no ghost, only the static displacement A sin(phi). Other 3-D line orders "
            "have other spacings. TR = profile.tr_ms when set, else 10 ms.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        implementations=_VIBRATION[0],
        realisation=_VIBRATION[1],
        requires=frozenset(),
        condition=vibration_condition,
        active_when=frozenset(),
    ),
)
