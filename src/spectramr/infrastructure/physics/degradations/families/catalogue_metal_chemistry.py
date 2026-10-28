"""The catalogue's metal and chemistry axes: metal_artifact and chemical_shift_phase."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.chemical_shift_phase import (
    chemical_shift_phase_condition,
    chemical_shift_phase_volume,
    degrade_chemical_shift_phase,
)
from spectramr.infrastructure.physics.degradations.catalogue.metal_chemistry import (
    degrade_metal_artifact,
    metal_artifact_volume,
    metal_condition,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL, native_2d
from spectramr.infrastructure.physics.degradations.spec import Realisation, VoxelSliceKernel
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_NAT = Realisation.NATIVE

ENTRIES = (
    native_2d(
        code="D35",
        name="metal_artifact",
        family="field",
        fn=VoxelSliceKernel(degrade_metal_artifact),
        description="Metal-implant susceptibility artefact (void, pile-up, GRE dephasing)",
        severity=SeveritySpec(
            (_P("sphere_radius", "mm", 0.0, 5.0),),
            notes="a metal sphere of radius 5 mm x theta at a seeded 3-D position (a voxel of the "
            "central half of each axis plus a through-plane offset z0 in [-a, a]) in every mode; "
            "dchi = 200 ppm (kwarg delta_chi_ppm, signed; titanium is ~180 ppm, Schenck 1996). "
            "Outside it the dipole offset dchi B0 a^3 (3 cos^2 th - 1) / (3 r^3) displaces signal "
            "along readout by -gamma dB / BW_px (the sign of b0_geometric_distortion; forward "
            "mapping: pile-up and voids); inside, no signal; for gradient-echo sequences also "
            "|sinc(gamma d(dB)/dz thk TE / 2)| from each slice's own gradient. B0 along "
            "geometry.b0_axis (through-plane in a 2-D slice). A 2-D slice is a one-slice "
            "multislice stack and, like multislice, is slice-selective: the signal of a source is "
            "assigned to the slice displaced by -thk df / BW_rf (kwarg rf_bandwidth_hz, default "
            "1000 Hz; Lu et al., MRM 2009), and what lands beyond the stack (all of it, for one "
            "slice) is lost. 3-D has no slice selection. Signal pushed out of the FOV is dropped "
            "(the receiver's anti-alias filter). A profile sets B0, BW_px, gradient_echo from its "
            "sequence and TE (defaults 3 T, 250 Hz/px, spin echo, 10 ms); EPI and non-Cartesian "
            "sequences raise. mm -> voxels by voxel_mm (1 mm without it).",
        ),
        affects=BOTH,  # complex sum of displaced sources mixes phases
        origin=ArtifactOrigin.PATIENT,
        modes={MS: (metal_artifact_volume, _NAT), VOL: (metal_artifact_volume, _NAT)},
        condition=metal_condition,
    ),
    native_2d(
        code="D36",
        name="chemical_shift_phase",
        family="field",
        fn=degrade_chemical_shift_phase,
        description="Fat-water in/opposed-phase interference (GRE type-2 chemical shift)",
        severity=SeveritySpec(
            (_P("te_sweep", "fraction of TE_in to TE_opp", 0.0, 1.0),),
            notes="|W + F exp(i 2 pi df TE)| with df = -sigma gamma B0, sigma the profile's chi_ppm "
            "(3.5 ppm: -447 Hz at 3 T; kwarg sigma_ppm, default scanner_profile.DEFAULT_CHI_PPM), "
            "TE = TE_in + theta (TE_opp - TE_in), TE_in = 1/|df|, TE_opp = 1.5/|df|, so theta=0 "
            "is in-phase (identity) and theta=1 opposed-phase. W = (1 - f) x, F = f x, input phase "
            "kept. f is the optional fat_map kwarg (B or 1, 1, D, H, W) in [0, 1]; absent, a "
            "seeded smooth map scaled to [0, fat_fraction] (profile.fat_fraction, default 0.1). "
            "A what-if gradient-echo TE sweep: it does not read the profile's sequence. A profile "
            "sets B0, fat_fraction and chi_ppm.",
        ),
        affects=MAG,
        origin=ArtifactOrigin.ACQUISITION,
        modes={
            MS: (chemical_shift_phase_volume, _NAT),
            VOL: (chemical_shift_phase_volume, _NAT),
        },
        condition=chemical_shift_phase_condition,
    ),
)
