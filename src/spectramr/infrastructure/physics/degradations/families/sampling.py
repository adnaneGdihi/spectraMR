"""The sampling family: cartesian_undersamp, vd_undersamp, vd_cartesian, radial_undersamp, partial_fourier."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import (
    MS,
    VOL,
    exact,
    native_2d,
)
from spectramr.infrastructure.physics.degradations.slice2d.sampling_cartesian import (
    degrade_cartesian_undersampling,
    degrade_partial_fourier,
    degrade_variable_density_cartesian,
)
from spectramr.infrastructure.physics.degradations.slice2d.sampling_vd_radial import (
    degrade_radial_undersampling,
    degrade_variable_density_2d,
)
from spectramr.infrastructure.physics.degradations.spec import ActivationGate, Realisation
from spectramr.infrastructure.physics.degradations.volume.sampling import (
    cartesian_undersamp_volume,
    partial_fourier_volume,
    vd_cartesian_volume,
    vd_undersamp_volume,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_NAT = Realisation.NATIVE
# Every sampling axis is an acceleration choice: a fully sampled protocol has none of them.
_ACCELERATED = frozenset({ActivationGate.ACCELERATED})

ENTRIES = (
    native_2d(
        code="D3",
        name="cartesian_undersamp",
        family="sampling",
        fn=degrade_cartesian_undersampling,
        description="Cartesian PE-line undersampling",
        severity=SeveritySpec(
            (_P("acceleration", "x", 1.0, 8.0),),
            notes="PE-line undersampling with an 8% fully-sampled centre.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_cartesian_undersampling), VOL: (cartesian_undersamp_volume, _NAT)},
        active_when=_ACCELERATED,
    ),
    native_2d(
        code="D4",
        name="vd_undersamp",
        family="sampling",
        fn=degrade_variable_density_2d,
        description="Variable-density 2-D Bernoulli",
        severity=SeveritySpec(
            (_P("acceleration", "x", 1.0, 8.0),),
            notes="2-D Bernoulli variable density; the Bernoulli scale is calibrated so "
            "the REALISED rate equals this declared one (#246).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_variable_density_2d), VOL: (vd_undersamp_volume, _NAT)},
        active_when=_ACCELERATED,
    ),
    native_2d(
        code="D4b",
        name="vd_cartesian",
        family="sampling",
        fn=degrade_variable_density_cartesian,
        description="Variable-density Cartesian PE-line",
        severity=SeveritySpec((_P("acceleration", "x", 1.0, 8.0),)),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_variable_density_cartesian), VOL: (vd_cartesian_volume, _NAT)},
        active_when=_ACCELERATED,
    ),
    native_2d(
        code="D5",
        name="radial_undersamp",
        family="sampling",
        fn=degrade_radial_undersampling,
        description="Radial spoke undersampling (Cartesian-grid)",
        severity=SeveritySpec(
            (_P("acceleration", "x", 1.0, 8.0),),
            notes="golden-angle spokes on a Cartesian grid; the spoke count is chosen so "
            "the realised kept fraction is 1/R. R=1 is a fully-sampled radial "
            "acquisition, i.e. the identity (#246 — the axis previously ramped a raw "
            "spoke count 200 -> 16, and 200 spokes do not cover the grid).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_radial_undersampling), VOL: exact(degrade_radial_undersampling)},
        active_when=_ACCELERATED,
    ),
    # NOTE: retained fraction *decreases* with severity.
    native_2d(
        code="D24",
        name="partial_fourier",
        family="sampling",
        fn=degrade_partial_fourier,
        description="Partial-Fourier acquisition",
        severity=SeveritySpec(
            (_P("retained_fraction", "fraction of the truncated axis", 1.0, 0.625),),
            notes="asymmetric k-space truncation along ky (kwarg axis='kx' for kx); theta=1 "
            "keeps the 5/8 minimum.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_partial_fourier), VOL: (partial_fourier_volume, _NAT)},
        active_when=_ACCELERATED,
    ),
)
