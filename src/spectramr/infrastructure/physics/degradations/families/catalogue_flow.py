"""The flow axes (family "motion": moving spins): flow_ghosts, flow_voids, inflow_enhancement."""

from __future__ import annotations

from typing import Any

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.flow import (
    FLOW_GHOST_MAX_DEPTH,
    FLOW_VOID_MAX_CM_S,
    INFLOW_MAX_CM_S,
    flow_ghosts,
    flow_ghosts_2d,
    flow_voids,
    flow_voids_2d,
    inflow_enhancement,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL, native_2d
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.spec import Realisation, VoxelSliceKernel, entry
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
M = AcquisitionMode
_NO_SLICE = "a single slice has no entry depth, so there is no inflow gradient to simulate"
_NATIVE = Realisation.NATIVE
_FALLBACK = (
    ". Without a vessel_map it draws 1-3 seeded vessel discs per sample (declared, deterministic)"
)
_VESSELS = (
    " The vessels are the optional vessel_map kwarg (a weight map of the input's rank, clamped to [0, 1]) or, "
    "when it is absent, 1-3 discs per sample (cylinders along D in a stack or volume) drawn "
    "deterministically from the seed."
)


def _known(**values: Any) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _ghosts_condition(profile: Any, theta: float, geom: AcquisitionGeometry, shape: tuple) -> dict:
    return _known(tr_ms=profile.tr_ms)


def _voids_condition(profile: Any, theta: float, geom: AcquisitionGeometry, shape: tuple) -> dict:
    return _known(te_ms=profile.te_ms)


def _inflow_condition(profile: Any, theta: float, geom: AcquisitionGeometry, shape: tuple) -> dict:
    return _known(tr_ms=profile.tr_ms, fa_deg=profile.fa_deg)


ENTRIES = (
    native_2d(
        code="D37",
        name="flow_ghosts",
        family="motion",
        fn=flow_ghosts_2d,
        description="Pulsatile-flow ghosts: vessel signal modulated line by line" + _FALLBACK,
        severity=SeveritySpec(
            (_P("modulation_depth", "dimensionless", 0.0, FLOW_GHOST_MAX_DEPTH),),
            notes="vessel signal times 1 + a cos(2 pi t_n / T_card + phi0) per phase-encode line "
            "(T_card = 1000 ms, TR = 50 ms, or the profile's TR); discrete ghosts of the vessel "
            "alone at +/- N_pe TR / T_card px along phase-encode. Line time: ky (slice); "
            "(slice, ky) by slice_order (stack); (kz, ky) by line_order (volume, ghosts along "
            "both phase-encode axes)." + _VESSELS,
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes={MS: (flow_ghosts, _NATIVE), VOL: (flow_ghosts, _NATIVE)},
        condition=_ghosts_condition,
    ),
    native_2d(
        code="D38",
        name="flow_voids",
        family="motion",
        fn=VoxelSliceKernel(flow_voids_2d),
        description="Flow voids: spin-echo washout of the vessel signal" + _FALLBACK,
        severity=SeveritySpec(
            (_P("through_plane_velocity", "cm/s", 0.0, FLOW_VOID_MAX_CM_S),),
            notes="vessel signal times max(0, 1 - v (TE / 2) / thk), TE = 20 ms or the profile's; "
            "thk is voxel_mm[0] for a slice; slice_thickness_mm, else voxel_mm[0] - slice_gap_mm "
            "for a stack; 5 mm for either without them; the slab D x voxel_mm[0] for a volume "
            "(1 mm voxels without voxel_mm); zero at "
            "v >= 2 thk / TE, which the range reaches at theta = 1 for the defaults (a longer "
            "TE or thinner slice saturates earlier, a thicker slab never)." + _VESSELS,
        ),
        affects=MAG,
        origin=ArtifactOrigin.PATIENT,
        modes={MS: (flow_voids, _NATIVE), VOL: (flow_voids, _NATIVE)},
        condition=_voids_condition,
    ),
    entry(
        code="D39",
        name="inflow_enhancement",
        family="motion",
        description="Inflow enhancement: fresh unsaturated spins brighten the vessel" + _FALLBACK,
        severity=SeveritySpec(
            (_P("through_plane_velocity", "cm/s", 0.0, INFLOW_MAX_CM_S),),
            notes="vessel signal times M_n / M_ss, the spoiled-GRE recursion M_{n+1} = M_n cos(a) "
            "E1 + (1 - E1) from M_0 = 1 after n = depth / (v TR) pulses (T1 = 1500 ms, "
            "TR = 30 ms, a = 30 deg, or the profile's TR and flip angle; a <= 90 deg). Depth is "
            "(k + 1/2) voxel_mm[0] at the slice centre (1 mm without voxel_mm), entry at index 0, "
            "so the gain tends to 1 as v -> 0. SLICE_2D is unsupported: a single slice has no entry depth, so there is "
            "no inflow gradient to simulate and the call raises UnsupportedAcquisitionMode."
            + _VESSELS,
        ),
        affects=MAG,
        origin=ArtifactOrigin.PATIENT,
        implementations={MS: inflow_enhancement, VOL: inflow_enhancement},
        realisation={MS: _NATIVE, VOL: _NATIVE},
        requires=frozenset(),
        unsupported_detail={M.SLICE_2D: _NO_SLICE},
        condition=_inflow_condition,
        active_when=frozenset(),
    ),
)
