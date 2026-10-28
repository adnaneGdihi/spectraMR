"""The named scanner presets and their lookup; a preset is a :class:`ScannerProfile` instance."""

from __future__ import annotations

from spectramr.config.schemas.physics import SCANNER_PRESET_FIELD_STRENGTH_T
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.scanner_profile import ScannerProfile, Sequence

__all__ = ["FASTMRI_3T", "M4RAW_03T", "PRESETS", "resolve_profile"]


def _slice_2d(voxel_mm: tuple[float, float, float]) -> AcquisitionGeometry:
    """A slice of a 2-D multislice acquisition; ``voxel_mm`` is (slice thickness, pe, ro)."""
    return AcquisitionGeometry(mode=AcquisitionMode.SLICE_2D, voxel_mm=voxel_mm)


#: M4Raw: 0.3 T ultra-low-field brain, 4-channel, single fully-sampled acquisition.
M4RAW_03T = ScannerProfile(
    name="m4raw_0.3T",
    b0_T=SCANNER_PRESET_FIELD_STRENGTH_T["m4raw_0.3T"],
    readout_bw_hz_per_px=200.0,
    sequence=Sequence.SPIN_ECHO,
    # 256 x 256 over 240 mm, 5 mm slices (docs/CLUSTER_DATA_LAYOUT.md:174-175).
    geometry=_slice_2d((5.0, 0.9375, 0.9375)),
    tr_ms=None,
    te_ms=None,
    ti_ms=None,
    fa_deg=None,
    n_coils=4,
    fat_fraction=0.1,
)

#: fastMRI brain: 3 T, ~15-channel, accelerated.
FASTMRI_3T = ScannerProfile(
    name="fastmri_3T",
    b0_T=SCANNER_PRESET_FIELD_STRENGTH_T["fastmri_3T"],
    readout_bw_hz_per_px=250.0,
    sequence=Sequence.SPIN_ECHO,
    # (16, 320, 320) at 5.0 x 0.6875 x 0.6875 mm (scripts/sim2rank/mosaic_synthseg.py:20).
    geometry=_slice_2d((5.0, 0.6875, 0.6875)),
    tr_ms=None,
    te_ms=None,
    ti_ms=None,
    fa_deg=None,
    acceleration=(4.0, 1.0),
    n_coils=15,
    fat_fraction=0.1,
)

PRESETS: dict[str, ScannerProfile] = {p.name: p for p in (M4RAW_03T, FASTMRI_3T)}


def resolve_profile(name: str) -> ScannerProfile:
    """Look up a preset by name; raise on an unknown one (never a silent default)."""
    if name not in PRESETS:
        raise KeyError(f"Unknown scanner profile {name!r}. Known presets: {sorted(PRESETS)}.")
    return PRESETS[name]
