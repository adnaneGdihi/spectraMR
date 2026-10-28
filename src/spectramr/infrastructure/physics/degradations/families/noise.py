"""The noise family: complex_gaussian, rician, emi."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL, native_2d
from spectramr.infrastructure.physics.degradations.slice2d.noise import (
    degrade_complex_gaussian_noise,
    degrade_rician_noise,
)
from spectramr.infrastructure.physics.degradations.spec import Realisation
from spectramr.infrastructure.physics.degradations.volume.noise import (
    complex_gaussian_volume,
    emi_volume,
    rician_volume,
)
from spectramr.infrastructure.physics.emi_degradation import degrade_emi
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_NAT = Realisation.NATIVE

ENTRIES = (
    # NOTE: SNR *decreases* with severity. theta=1 is the 2 dB (worst) end.
    native_2d(
        code="D1",
        name="complex_gaussian",
        family="noise",
        fn=degrade_complex_gaussian_noise,
        description="Complex Gaussian k-space noise (→Rician)",
        severity=SeveritySpec(
            (_P("snr", "dB", 40.0, 2.0),),
            notes="k-space AWGN; theta ramps SNR down, so the physical minimum is the "
            "most-degraded end.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes={MS: (complex_gaussian_volume, _NAT), VOL: (complex_gaussian_volume, _NAT)},
    ),
    native_2d(
        code="D2",
        name="rician",
        family="noise",
        fn=degrade_rician_noise,
        description="Image-space Rician magnitude noise",
        severity=SeveritySpec(
            (_P("sigma", "fraction of p99 signal", 0.0, 0.3),),
            notes="image-space Rician magnitude noise; sigma is scaled by the input's own "
            "p99 so the axis is scale-invariant (p99 not max, so one hot voxel cannot set "
            "the noise floor).",
        ),
        affects=MAG,  # returns a real magnitude image by construction
        origin=ArtifactOrigin.SCANNER,
        modes={MS: (rician_volume, _NAT), VOL: (rician_volume, _NAT)},
    ),
    native_2d(
        code="D32",
        name="emi",
        family="noise",
        fn=degrade_emi,
        description="Narrowband RF interference tones (unshielded low-field scanner)",
        severity=SeveritySpec(
            (_P("interference_rms", "x k-space RMS", 0.0, 0.3),),
            notes="4 narrowband tones at seeded off-grid readout frequencies, independent "
            "phase per PE line; the summed interference RMS is the declared multiple of the "
            "k-space RMS.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes={MS: (emi_volume, _NAT), VOL: (emi_volume, _NAT)},
    ),
)
