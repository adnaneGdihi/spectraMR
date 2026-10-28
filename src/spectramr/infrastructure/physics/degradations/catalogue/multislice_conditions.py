"""Scanner conditions of the multislice axes: ``condition(profile, theta, geometry, shape) -> kwargs``.

A profile field the preset leaves ``None`` (TR, flip angle, gradient limit) is not forwarded, so
the axis's stated default applies; a field the profile sets overrides it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = ["off_resonance_condition", "spin_history_condition", "tr_condition"]

Shape = tuple[int, ...]


def tr_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """The profile's TR (ms), when it sets one."""
    return {} if profile.tr_ms is None else {"tr_ms": profile.tr_ms}


def spin_history_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """The profile's TR (ms) and flip angle (deg), for those it sets."""
    out = tr_condition(profile, theta, geometry, shape)
    return out if profile.fa_deg is None else {**out, "fa_deg": profile.fa_deg}


def off_resonance_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, float]:
    """Field scale ``B0 / 3 T`` of the seeded field (as the b0 axis) and the gradient limit.

    It is realised as ``profile_field_scale``, not ``field_scale``: it applies to the seeded field
    only, so a measured ``b0_map_hz`` passed with the profile is used as given.
    """
    out = {"profile_field_scale": profile.b0_field_scale()}
    if profile.gradient_max_mT_per_m is not None:
        out["gradient_max_mT_per_m"] = profile.gradient_max_mT_per_m
    return out
