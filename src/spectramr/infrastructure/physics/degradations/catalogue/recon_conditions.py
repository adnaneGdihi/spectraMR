"""Scanner conditions of the reconstruction axes: the profile's acceleration, coil count and covariance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = ["gfactor_condition", "pi_residual_condition"]

Shape = tuple[int, ...]


def gfactor_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, object]:
    """The reduction factors, receive array and noise covariance the scanner reconstructs with."""
    return {
        "acceleration": profile.acceleration,
        "n_coils": profile.n_coils,
        "coil_covariance": profile.coil_covariance,
    }


def pi_residual_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: Shape
) -> dict[str, object]:
    """As :func:`gfactor_condition` without ``n_coils``: the input's channels are the coils."""
    return {"acceleration": profile.acceleration, "coil_covariance": profile.coil_covariance}
