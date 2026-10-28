"""The catalogue transmit family: B1+ maps, fat saturation, inversion and SAR-limited flip."""

from __future__ import annotations

from collections.abc import Callable

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.transmit import (
    b1_transmit_volume,
    degrade_b1_transmit,
    degrade_dielectric_b1,
    dielectric_b1_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_conditions import (
    dielectric_condition,
    fat_sat_condition,
    inversion_condition,
    sar_condition,
    transmit_condition,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_prep import (
    degrade_imperfect_fat_sat,
    degrade_imperfect_inversion,
    imperfect_fat_sat_volume,
    imperfect_inversion_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_sar import (
    degrade_sar_limited_flip,
    sar_limited_flip_volume,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import (
    ActivationGate,
    DegradationSpec,
    ModeImpl,
    Realisation,
    Requirement,
    VoxelSliceKernel,
    entry,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_SIGNAL_NOTE = (
    " The signal response is keyed by the sequence: GRE / spoiled GRE sin(kappa a) / sin(a) (the "
    "Ernst steady state when TR is given, with T1 = t1_ms, default 1000 ms), SE / TSE "
    "sin(kappa 90) sin^2(kappa 90) with a and TR not read; without a profile the declared default "
    "is a spin echo (a GRE must be given fa_deg), and an unmodelled sequence raises."
)


def _native(
    *,
    code: str,
    name: str,
    kernel: Callable,
    volume: ModeImpl,
    description: str,
    severity: SeveritySpec,
    affects: frozenset[str],
    origin: ArtifactOrigin,
    condition: Callable | None,
    requires: frozenset[Requirement] = frozenset(),
    active_when: frozenset[ActivationGate] = frozenset(),
) -> DegradationSpec:
    """All three modes native: the 2-D kernel for a slice, one 5-D implementation otherwise."""
    return entry(
        code=code,
        name=name,
        family="field",
        description=description,
        severity=severity,
        affects=affects,
        origin=origin,
        implementations={
            AcquisitionMode.SLICE_2D: VoxelSliceKernel(kernel),
            MS: volume,
            VOL: volume,
        },
        realisation=dict.fromkeys((AcquisitionMode.SLICE_2D, MS, VOL), Realisation.NATIVE),
        requires=requires,
        condition=condition,
        active_when=active_when,
    )


ENTRIES: tuple[DegradationSpec, ...] = (
    _native(
        code="D51",
        name="b1_transmit",
        kernel=degrade_b1_transmit,
        volume=b1_transmit_volume,
        description="Smooth B1+ transmit scale map (flip-angle error)",
        severity=SeveritySpec(
            (_P("peak_b1_scale_error", "fraction of nominal B1+", 0.0, 0.4),),
            notes="a seeded smooth, centre-bright B1+ scale map k(r) = 1 + 0.4 s(r), s of unit "
            "peak, referenced to the transmit calibration: kappa = 1 + theta (k / <k> - 1) inside "
            "the object support (mean over the support; magnitude above 5% of the sample peak), so "
            "the actual flip angle is kappa a. kwarg kappa_map (already relative to nominal) "
            "replaces it by kappa = 1 + theta (kappa_map - 1)." + _SIGNAL_NOTE,
        ),
        affects=MAG,
        origin=ArtifactOrigin.SCANNER,
        condition=transmit_condition,
    ),
    _native(
        code="D52",
        name="dielectric_b1",
        kernel=degrade_dielectric_b1,
        volume=dielectric_b1_volume,
        description="Dielectric standing-wave B1+ pattern",
        severity=SeveritySpec(
            (_P("standing_wave_amplitude", "fraction of nominal B1+", 0.0, 0.5),),
            notes="k(r) = 1 + A cos(2 pi |r - r0| / lambda), A = 0.5, referenced to the transmit "
            "calibration: kappa = 1 + theta (k / <k> - 1) inside the object support (magnitude "
            "above 5% of the sample peak), 1 outside, so the pattern vanishes as lambda grows at "
            "low field; r in mm from the FOV centre "
            "(voxel_mm; 1 mm without it) and lambda = c / (42.577 MHz/T B0 sqrt(eps_r)), eps_r = "
            "50 (kwarg): 332 mm at 3 T, 664 mm at 1.5 T. B0 is b0_T (default 3 T) or the profile's."
            + _SIGNAL_NOTE,
        ),
        affects=MAG,
        origin=ArtifactOrigin.PATIENT,
        condition=dielectric_condition,
    ),
    _native(
        code="D53",
        name="imperfect_fat_sat",
        kernel=degrade_imperfect_fat_sat,
        volume=imperfect_fat_sat_volume,
        description="Imperfect CHESS fat saturation (residual fat)",
        severity=SeveritySpec(
            (_P("peak_abs_db0", "Hz", 0.0, 300.0),),
            notes="the input is the fat-saturated (water) image; residual fat F (1 - eta) is added "
            "in phase with it, eta = exp(-(dB0 / w)^2), w = 100 Hz. dB0 = 300 Hz theta s(r) for a "
            "seeded smooth s of unit peak, or theta b0_map_hz. F is the kwarg fat_map, or the "
            "profile's fat_fraction (default 0.1) times the sample's 99th-percentile signal times "
            "a seeded smooth map in [0, 1], inside the object support.",
        ),
        affects=MAG,
        origin=ArtifactOrigin.ACQUISITION,
        condition=fat_sat_condition,
        active_when=frozenset({ActivationGate.FAT_PRESENT}),
    ),
    _native(
        code="D54",
        name="imperfect_inversion",
        kernel=degrade_imperfect_inversion,
        volume=imperfect_inversion_volume,
        description="Imperfect inversion pulse efficiency (IR contrast, shifted null)",
        severity=SeveritySpec(
            (_P("inversion_inefficiency", "1 - kappa", 0.0, 0.3),),
            notes="Mz(TI) = 1 - (1 + kappa) exp(-TI / T1), kappa = 1 - 0.3 theta, TR >> T1; the "
            "null moves from T1 ln 2 to T1 ln(1 + kappa). Requires qmaps = {t1 (ms), pd (the "
            "equilibrium image, in the units of x)} and TI (ti_ms or the profile's, no default): "
            "the input is the ideal-inversion image and pd (|Mz(kappa)| - |Mz(1)|) is added in "
            "phase.",
        ),
        affects=MAG,
        origin=ArtifactOrigin.ACQUISITION,
        condition=inversion_condition,
        requires=frozenset({Requirement.QMAPS}),
    ),
    _native(
        code="D55",
        name="sar_limited_flip",
        kernel=degrade_sar_limited_flip,
        volume=sar_limited_flip_volume,
        description="SAR-limited TSE refocusing angle (reweighted echo train)",
        severity=SeveritySpec(
            (_P("refocusing_flip_reduction", "degrees", 0.0, 120.0),),
            notes="refocusing angle 180 - 120 theta degrees, down to 60 (90 degree excitation); "
            "each k-space line is scaled by the EPG echo amplitude of its echo over the 180 degree "
            "train's, so the axis carries the PSF change and the signal-level change at TE_eff. "
            "It is NOT monotone in "
            "theta everywhere: the rms error of the 32x32 test phantom rises to 0.0069 at theta "
            "0.20, falls to 0.0045 at 0.30 (the stimulated echo recovers from just below 180 "
            "degrees), then rises to 0.116 at 1. "
            "Lines map to echoes in linear order: ranked by acquisition time and cut into ETL "
            "groups (a ky train per slice in 2-D and multislice, the (kz, ky) ranking in 3-D), "
            "cyclically shifted so the k-space centre is at the effective echo nearest the "
            "profile's TE (the middle of the train without one). ETL and ESP are the profile's "
            "(defaults 16 and 10 ms), T1 = 1000 ms and T2 = 80 ms (kwargs).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        condition=sar_condition,
    ),
)
