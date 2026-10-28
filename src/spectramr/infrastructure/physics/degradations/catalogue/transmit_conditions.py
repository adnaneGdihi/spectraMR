"""Scanner conditions of the transmit axes: the profile facts each one reads (Ruling 43).

``condition(profile, theta, geometry, shape) -> kwargs``. A fact the profile leaves unknown
(``None``) is forwarded as ``None`` where the kernel must refuse it, and omitted where the kernel
has a declared default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spectramr.infrastructure.physics.degradations.catalogue.transmit_signal import NEEDS_FA
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = [
    "dielectric_condition",
    "fat_sat_condition",
    "inversion_condition",
    "sar_condition",
    "transmit_condition",
]

Shape = tuple[int, ...]
#: Sequences with a 180-degree refocusing train.
REFOCUSED = frozenset({"spin_echo", "tse", "ir_tse"})


def transmit_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, Any]:
    """The sequence's transmit signal model and, for a GRE, its nominal flip angle and TR."""
    kw: dict[str, Any] = {"sequence": profile.sequence}
    if str(profile.sequence) in NEEDS_FA:
        kw |= {"fa_deg": profile.fa_deg, "tr_ms": profile.tr_ms}
    return kw


def dielectric_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, Any]:
    """The field sets the RF wavelength; the sequence sets the signal response."""
    return {"b0_T": profile.b0_T, **transmit_condition(profile, theta, geometry, shape)}


def fat_sat_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """Residual fat scales with the profile's fat fraction."""
    return {"fat_fraction": profile.fat_fraction}


def inversion_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float | None]:
    """The inversion time; ``None`` makes the kernel raise, there is no default TI."""
    return {"ti_ms": profile.ti_ms}


def sar_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float | int]:
    """The echo train length and, when known, the echo spacing and the effective TE.

    A sequence without a refocusing train has no SAR-limited refocusing angle and raises.
    """
    if str(profile.sequence) not in REFOCUSED:
        raise ValueError(
            f"sar_limited_flip needs a refocused sequence {sorted(REFOCUSED)}, "
            f"profile {profile.name!r} is {str(profile.sequence)!r}"
        )
    kw: dict[str, float | int] = {"etl": profile.etl}
    for key, value in (("esp_ms", profile.esp_ms), ("te_ms", profile.te_ms)):
        if value is not None:
            kw[key] = value
    return kw
