"""The motion family: rigid, pulsatile, drift, sudden and random motion, and the catalogue's
respiratory and local intermittent motion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.adapters import Fold, per_slice
from spectramr.infrastructure.physics.degradations.catalogue.motion_local import (
    degrade_local_intermittent_motion,
    local_intermittent_motion_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_respiratory import (
    degrade_respiratory_motion,
    respiratory_motion_volume,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import MS, VOL, ModeDecl, native_2d
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry, SliceOrder
from spectramr.infrastructure.physics.degradations.slice2d.motion import (
    _delegate_rigid_motion,
    degrade_pulsatile_motion,
    degrade_random_motion,
    degrade_sudden_motion,
    degrade_translation_drift,
)
from spectramr.infrastructure.physics.degradations.spec import (
    ModeImpl,
    Realisation,
    VoxelSliceKernel,
)
from spectramr.infrastructure.physics.degradations.volume.motion_pulsatile import pulsatile_volume
from spectramr.infrastructure.physics.degradations.volume.motion_rigid import (
    rigid_motion_multislice,
    rigid_motion_volume,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import (
    random_motion_volume,
    sudden_motion_volume,
    translation_drift_volume,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam


def _multislice(kernel: Callable[..., Tensor], sequential: ModeImpl) -> ModeDecl:
    """Multislice readout motion, by slice order.

    Interleaved: PE line ky of every slice is acquired in the same TR, so every slice sees the
    same displacement d(ky) -- the 2-D kernel run once over all slices. Sequential: each slice
    is acquired after the previous one, so the schedule runs over the whole stack's line times.
    """
    by_order = {
        SliceOrder.INTERLEAVED: per_slice(kernel, Fold.SHARED),
        SliceOrder.SEQUENTIAL: sequential,
    }

    def impl(
        x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, **kwargs: Any
    ) -> Tensor:
        return by_order[geometry.slice_order](x, theta, seed, geometry, **kwargs)

    impl.__name__ = f"multislice_{getattr(kernel, '__name__', 'kernel')}"
    return impl, Realisation.NATIVE


def _schedule_modes(kernel: Callable[..., Tensor], native: ModeImpl) -> dict:
    return {MS: _multislice(kernel, native), VOL: (native, Realisation.NATIVE)}


_SCHEDULE_NOTE = (
    " Volume modes keep the readout displacement and time it by acquisition: sequential "
    "multislice over the whole stack's (slice, ky) line times, 3-D over the (kz, ky) line "
    "order; interleaved multislice shares d(ky) across slices, as in 2-D."
)


ENTRIES = (
    # Two knobs move together. This is a severity ramp, not an ablation -- but the
    # coupling is declared rather than hidden behind whichever one got reported.
    native_2d(
        code="D6",
        name="rigid_motion",
        family="motion",
        fn=_delegate_rigid_motion,
        description="Rigid motion (delegated)",
        severity=SeveritySpec(
            (
                _P("max_translation", "px", 0.0, 10.0),
                _P("max_rotation", "rad", 0.0, 0.1),
            ),
            notes="per-PE-line SE(2) pose; translation and rotation ramp together. Volume "
            "modes: 8 shots of consecutive acquisition time, each with its own rotation (three "
            "angles in 3-D; in-plane plus a through-plane dz in multislice), and "
            "each (kz or slice, ky) line its own translation; translations are isotropic in mm (declared "
            "px x in-plane voxel size per axis, px without voxel_mm) and rotations rigid in mm.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes={
            MS: (rigid_motion_multislice, Realisation.NATIVE),
            VOL: (rigid_motion_volume, Realisation.NATIVE),
        },
    ),
    native_2d(
        code="D7",
        name="pulsatile_motion",
        family="motion",
        fn=degrade_pulsatile_motion,
        description="Pulsatile non-rigid deformation",
        severity=SeveritySpec(
            (_P("peak_displacement", "px", 0.0, 9.0),),
            notes="radial non-rigid pulsation at 1.2 Hz, TR = 0.5 s; each PE line sees the "
            "cardiac phase at its own acquisition time (aliased at the TR, as in an ungated "
            "scan), quantised into 16 rendered poses. Volume modes time each (slice or kz, "
            "ky) line by its acquisition order and pulse a 3-D radial field isotropic in mm, "
            "peak = declared px x in-plane voxel size (1 mm voxels without voxel_mm).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes={
            MS: (pulsatile_volume, Realisation.NATIVE),
            VOL: (pulsatile_volume, Realisation.NATIVE),
        },
    ),
    native_2d(
        code="D29",
        name="translation_drift",
        family="motion",
        fn=degrade_translation_drift,
        description="Coherent translational drift",
        severity=SeveritySpec(
            (_P("max_drift", "px", 0.0, 10.0),),
            notes="coherent linear bulk drift along readout across the acquisition: a "
            "single low-order k_y phase gradient -> directional smear, distinct from the "
            "broadband jitter of rigid_motion."
            + _SCHEDULE_NOTE,
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes=_schedule_modes(degrade_translation_drift, translation_drift_volume),
    ),
    native_2d(
        code="D30",
        name="sudden_motion",
        family="motion",
        fn=degrade_sudden_motion,
        description="Single abrupt bulk shift",
        severity=SeveritySpec(
            (_P("shift", "px", 0.0, 12.0),),
            notes="single abrupt bulk shift at the k-space centre (a mid-scan jerk); the "
            "step in k_y yields a discrete double-image ghost."
            + _SCHEDULE_NOTE
            + " The volume step sits at the median acquisition time (the 2-D centre line in linear order).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes=_schedule_modes(degrade_sudden_motion, sudden_motion_volume),
    ),
    native_2d(
        code="D31",
        name="random_motion",
        family="motion",
        fn=degrade_random_motion,
        description="Random per-PE-line jitter",
        severity=SeveritySpec(
            (_P("max_shift", "px", 0.0, 3.0),),
            notes="i.i.d. per-PE-line random translational jitter -> broadband ghosting; "
            "the pure-translation sibling of rigid_motion (no rotation)."
            + _SCHEDULE_NOTE,
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes=_schedule_modes(degrade_random_motion, random_motion_volume),
    ),
    native_2d(
        code="D33",
        name="respiratory_motion",
        family="motion",
        fn=VoxelSliceKernel(degrade_respiratory_motion),
        description="Free-breathing respiratory motion",
        severity=SeveritySpec(
            (
                _P("peak_si_translation", "mm", 0.0, 20.0),
                _P("peak_ap_scale", "fraction", 0.0, 0.04),
            ),
            notes="per acquired line, an SI translation of 20 mm x theta x (u(t) - u_ref) and "
            "an AP scaling about the image centre of 1 + 0.04 x theta x (u(t) - u_ref) "
            "(chest-wall expansion), u the hysteresis breathing trace of motion_simulation "
            "(one breath per 4 s period, i.e. 15 breaths/min; TR 0.5 s; seeded phase) normalised to [0, 1], so 20 mm and 0.04 are "
            "peak-to-peak excursions; u_ref is the time-averaged state (kwarg reference='mean'; "
            "'exhale' takes u_ref = 0). Lines are timed by acquisition_timing: per ky (2-D), "
            "per (slice, ky) by slice order (multislice), per (kz, ky) by line order (3-D). SI "
            "and AP are kwargs si_axis / ap_axis in {slice, pe, ro} (defaults pe / ro in 2-D, "
            "slice / pe in volumes); mm -> voxels by voxel_mm, 1 mm isotropic without it. An "
            "encoded-axis translation is the exact per-line Fourier phase; scaling and a "
            "multislice through-plane shift are rendered on 16 quantised trace levels. A scanner "
            "profile's TR is not read.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes={
            MS: (respiratory_motion_volume, Realisation.NATIVE),
            VOL: (respiratory_motion_volume, Realisation.NATIVE),
        },
    ),
    native_2d(
        code="D34",
        name="local_intermittent_motion",
        family="motion",
        fn=VoxelSliceKernel(degrade_local_intermittent_motion),
        description="Intermittent regional (swallowing) motion",
        severity=SeveritySpec(
            (
                _P("peak_regional_displacement", "mm", 0.0, 10.0),
                _P("motion_block_fraction", "fraction", 0.0, 0.3),
            ),
            notes="a compact-support smooth warp inside one seeded ellipsoidal region (peak "
            "10 mm x theta at its centre, zero outside), present only while a seeded contiguous "
            "block of round(0.3 x theta x lines) acquired lines is acquired; the other lines see "
            "the object at rest. Lines are ordered in time by acquisition_timing as for "
            "respiratory_motion; the region is in-plane in 2-D and 3-D in volumes, mm -> voxels "
            "by voxel_mm (1 mm isotropic without it). A scanner profile's TR is not read.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.PATIENT,
        modes={
            MS: (local_intermittent_motion_volume, Realisation.NATIVE),
            VOL: (local_intermittent_motion_volume, Realisation.NATIVE),
        },
    ),
)
