"""The geometric family: geometric_warp, gradient_nonlinearity."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import MS, VOL, native_2d
from spectramr.infrastructure.physics.degradations.slice2d.geometric import (
    _delegate_gradient_nl,
    degrade_geometric_warp,
)
from spectramr.infrastructure.physics.degradations.spec import Realisation
from spectramr.infrastructure.physics.degradations.volume.geometric import (
    geometric_warp_volume,
    gradient_nonlinearity_volume,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam


def _native(impl):
    """One native volume implementation for multislice and 3-D: both see the same 3-D field."""
    return {MS: (impl, Realisation.NATIVE), VOL: (impl, Realisation.NATIVE)}


ENTRIES = (
    native_2d(
        code="D20",
        name="geometric_warp",
        family="geometric",
        fn=degrade_geometric_warp,
        description="Low-frequency geometric warp",
        severity=SeveritySpec(
            (_P("peak_displacement", "px", 0.0, 16.0),),
            notes="smooth 5-mode elastic deformation, normalised so the peak displacement "
            "magnitude is exactly the declared value for every seed and grid size. Volume "
            "modes draw a 3-D (dz, dy, dx) field isotropic in mm whose peak |d| is the declared "
            "px times the in-plane voxel size (voxels taken as 1 mm cubes without voxel_mm).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes=_native(geometric_warp_volume),
    ),
    native_2d(
        code="D21",
        name="gradient_nonlinearity",
        family="geometric",
        fn=_delegate_gradient_nl,
        description="Gradient non-linearity (delegated)",
        severity=SeveritySpec(
            (_P("severity", "polynomial coeff scale", 0.0, 7e-5),),
            notes="order >= 2 polynomial displacement only (no translation or gradient "
            "scaling), the same fraction of the FOV at every grid size. Peak |d| over the whole "
            "2-D FOV at theta=1: 2.3-7.0% of FOV, median 4.4% over 20 seeds (128 x 128; "
            "Halbach systems reach ~5%, Koolstra et al. 2021). Volume modes use the order 2..3 "
            "monomials of (z, y, x) position in mm at the same coefficient scale (each axis's "
            "own FOV without voxel_mm), so the z = 0 plane is the 2-D family. Over a 128 mm "
            "cube (64 x 128 x 128 at 2 mm slices, or 128^3 at 1 mm) the peak |d| including dz "
            "is 6.6-11.9% of the in-plane FOV, median 9.1%; multislice slices curve "
            "through-plane. The 2-D severity= override is rejected in volume modes.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes=_native(gradient_nonlinearity_volume),
    ),
)
