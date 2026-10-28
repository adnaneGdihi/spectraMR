"""The field family, field-map entries: B0, bias, chemical shift, susceptibility, coils, T2, phase."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.conditions import (
    b0_condition,
    b0_geometric_condition,
    chemical_shift_condition,
    concomitant_condition,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG, PHASE_ONLY
from spectramr.infrastructure.physics.degradations.families import (
    MS,
    VOL,
    exact,
    native_2d,
)
from spectramr.infrastructure.physics.degradations.slice2d.fields import (
    _delegate_b0,
    _delegate_b0_geometric,
    _delegate_chemical_shift,
    _delegate_magic_angle,
    degrade_bias_field,
    degrade_coil_sensitivity_miscal,
    degrade_susceptibility,
)
from spectramr.infrastructure.physics.degradations.slice2d.kspace_phase import (
    degrade_concomitant_phase,
)
from spectramr.infrastructure.physics.degradations.spec import ActivationGate, Realisation
from spectramr.infrastructure.physics.degradations.volume.fields_b0 import (
    b0_geometric_volume,
    b0_volume,
    susceptibility_volume,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import (
    bias_volume,
    coil_miscal_volume,
    concomitant_volume,
)
from spectramr.infrastructure.physics.degradations.volume.magic_angle import magic_angle_volume
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
# The fat map is a voxel field, so it folds with the volume.
_FAT_MAP = ("fat_fraction_map",)
_NAT = Realisation.NATIVE


def _native(impl):
    """One native volume implementation for both multislice and 3-D (it branches on mode)."""
    return {MS: (impl, _NAT), VOL: (impl, _NAT)}

ENTRIES = (
    native_2d(
        code="D10",
        name="b0",
        family="field",
        fn=_delegate_b0,
        description="B0 inhomogeneity (delegated)",
        severity=SeveritySpec(
            (_P("strength", "normalised field-map scale", 0.0, 1.0),),
            notes="dimensionless scale on the polynomial field map (NOT Hz): drives "
            "both a phase ramp and intra-voxel-dephasing dropout. In volume modes the field "
            "is drawn on coordinates normalised by the grid's largest half-extent, so the "
            "in-plane phase scale depends on whether voxel_mm is set and on the slab aspect "
            "ratio.",
        ),
        affects=BOTH,  # phase ramp AND intra-voxel-dephasing magnitude dropout
        origin=ArtifactOrigin.SCANNER,
        modes=_native(b0_volume),
        condition=b0_condition,
    ),
    native_2d(
        code="D10b",
        name="b0_geometric_distortion",
        family="field",
        fn=_delegate_b0_geometric,
        description="B0 voxel-shift distortion (delegated)",
        severity=SeveritySpec(
            (_P("b0_strength", "normalised field-map scale", 0.0, 1.0),),
            notes="spin-warp voxel shift along readout only, at a 5e-5 s dwell time; a "
            "scanner profile sets the dwell to 1 / (BW_px * W).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes=_native(b0_geometric_volume),
        condition=b0_geometric_condition,
    ),
    native_2d(
        code="D11",
        name="bias",
        family="field",
        fn=degrade_bias_field,
        description="Multiplicative bias field",
        severity=SeveritySpec(
            (_P("max_log_bias", "log multiplicative", 0.0, 1.2),),
            notes="smooth multiplicative bias field.",
        ),
        affects=MAG,  # real positive multiplicative field
        origin=ArtifactOrigin.SCANNER,
        modes=_native(bias_volume),
    ),
    native_2d(
        code="D16",
        name="chemical_shift",
        family="field",
        fn=_delegate_chemical_shift,
        description="Chemical shift (delegated)",
        severity=SeveritySpec(
            (_P("shift", "px", 0.0, 14.0),),
            notes="synthetic fat displacement along readout. The physical 3.5 ppm shift is "
            "gamma/2pi * B0 * 3.5e-6 / BW_pixel: about 0.1-0.5 px at 0.3 T and 0.5-2 px at "
            "1.5 T for 100-500 Hz/px, so 14 px is not a real field strength. Passing b0_T and "
            "bw_per_pixel_hz makes theta scale the physical shift instead.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={
            MS: exact(_delegate_chemical_shift, volume_kwargs=_FAT_MAP),
            VOL: exact(_delegate_chemical_shift, volume_kwargs=_FAT_MAP),
        },
        condition=chemical_shift_condition,
        active_when=frozenset({ActivationGate.FAT_PRESENT}),
    ),
    native_2d(
        code="D17",
        name="susceptibility",
        family="field",
        fn=degrade_susceptibility,
        description="Susceptibility (dipole + dropout)",
        severity=SeveritySpec(
            (
                _P("phase_scale", "dimensionless", 0.0, 1.0),
                _P("max_dropout", "fraction of signal", 0.0, 0.9817),  # 1 - exp(-4)
            ),
            notes="dipole field of a random smooth susceptibility map (not localised to "
            "air-tissue interfaces), B0 in the image plane (coronal/sagittal geometry); "
            "phase and dropout ramp together, no geometric distortion. Dropout is a "
            "through-slice dephasing proxy. Volume modes run B0 along geometry.b0_axis "
            "(default 'slice', i.e. axial); the 2-D axis keeps its in-plane kernel. In "
            "multislice the through-slice dephasing |sinc(dphi_thk / 2 pi)| multiplies the "
            "in-plane dropout, so the total dropout there can reach 1.",
        ),
        affects=BOTH,  # dipole phase AND dropout
        origin=ArtifactOrigin.PATIENT,
        modes=_native(susceptibility_volume),
    ),
    native_2d(
        code="D18",
        name="coil_miscal",
        family="field",
        fn=degrade_coil_sensitivity_miscal,
        description="Coil-sensitivity miscalibration",
        severity=SeveritySpec(
            (_P("relative_error", "fraction", 0.0, 1.0),),
            notes="coil-sensitivity miscalibration.",
        ),
        affects=MAG,  # real non-negative ratio map
        origin=ArtifactOrigin.SCANNER,
        modes=_native(coil_miscal_volume),
    ),
    native_2d(
        code="D22",
        name="magic_angle",
        family="field",
        fn=_delegate_magic_angle,
        description="Magic-angle T2 modulation (delegated; applies MSK collagen patches to any anatomy)",
        severity=SeveritySpec(
            (_P("t2_magic", "ms", 10.0, 30.0),),
            notes="collagen-fibre orientation T2 lengthening at TE = 20 ms. Fibre "
            "patches are drawn from the SUPPORT of the anatomy and cover ~30% of it "
            "(an MSK slice through a tendon-rich region); before #223 they were drawn "
            "uniformly over the FOV and at seed 0 both landed in signal-free air, "
            "making the axis an exact no-op. Volume modes draw ellipsoidal patches from the "
            "3-D support with an orientation field smooth in 3-D.",
        ),
        affects=MAG,  # real positive T2 signal boost
        origin=ArtifactOrigin.ACQUISITION,
        modes=_native(magic_angle_volume),
    ),
    native_2d(
        code="D23",
        name="concomitant_phase",
        family="field",
        fn=degrade_concomitant_phase,
        description="Maxwell concomitant phase (low-field)",
        severity=SeveritySpec(
            (_P("gradient_amplitude", "mT/m", 0.0, 30.0),),
            notes="Maxwell/concomitant-field phase at B0 = 0.3 T, or at a scanner profile's B0.",
        ),
        affects=PHASE_ONLY,  # THE pure-phase axis. Magnitude-scoring it yields a constant.
        origin=ArtifactOrigin.SCANNER,
        modes=_native(concomitant_volume),
        condition=concomitant_condition,
    ),
)
