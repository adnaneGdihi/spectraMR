"""Scanner conditions: the kwargs a :class:`ScannerProfile` implies for one degradation.

Each condition is ``condition(profile, theta, geometry, shape) -> kwargs``: the profile facts
the axis's physics reads, either a theta-scaled physical maximum (so the axis stays monotone
in severity and still differs between fields) or a protocol fact such as the sequence, B0 or
the echo spacing. ``chemical_shift`` gets a theta-scaled ``shift_pixels``, never the raw
``b0_T``/``bw``: its delegate would then compute a fixed shift and be theta-invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = [
    "B0_SEQUENCE_OF",
    "b0_condition",
    "b0_geometric_condition",
    "chemical_shift_condition",
    "concomitant_condition",
    "b0_shift_sequence",
]

Shape = tuple[int, ...]

#: Sequence family -> the B0 voxel-shift model of ``simulate_b0_geometric_distortion``. Keyed by
#: the ``Sequence`` values (a StrEnum member looks up as its string). Radial and spiral are
#: absent on purpose: their off-resonance is a blur, not a Cartesian voxel shift.
B0_SEQUENCE_OF: dict[str, str] = {
    "epi": "epi",
    "spin_echo": "spin_warp",
    "tse": "spin_warp",
    "gre": "spin_warp",
    "spoiled_gre": "spin_warp",
    "ir_tse": "spin_warp",
    "mprage": "spin_warp",
    "bssfp": "spin_warp",
}


def b0_shift_sequence(sequence: str) -> str:
    """The B0 voxel-shift model for ``sequence``; a non-Cartesian sequence raises."""
    try:
        return B0_SEQUENCE_OF[sequence]
    except KeyError:
        raise ValueError(
            f"no B0 voxel-shift model for sequence {str(sequence)!r}: non-Cartesian "
            "off-resonance is a blur, a future axis (offres_blur_noncartesian), not a shift. "
            f"Shift models exist for {sorted(B0_SEQUENCE_OF)}."
        ) from None


def chemical_shift_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """Fat displacement ``theta * gamma B0 chi / BW_px`` and the profile's fat fraction."""
    return {
        "shift_pixels": theta * profile.chemical_shift_max_px(),
        "fat_fraction": profile.fat_fraction,
    }


def b0_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """Off-resonance field scale ``theta * B0 / 3 T``."""
    return {"strength": theta * profile.b0_field_scale()}


def b0_geometric_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float | str]:
    """Field scale ``theta * B0 / 3 T``, the dwell ``1 / (BW_px W)`` and the shift model."""
    return {
        "b0_strength": theta * profile.b0_field_scale(),
        "dwell_time": profile.readout_dwell_s(shape[-1]),
        "sequence": b0_shift_sequence(profile.sequence),
    }


def concomitant_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """The Maxwell phase scales as ``1 / B0``, so the profile's field is passed through."""
    return {"b0_T": profile.b0_T}
