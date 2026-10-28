"""The field family, k-space and readout entries: ghosts, truncation, blur, eddy, spikes."""

from __future__ import annotations

import math

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.declarations import BOTH
from spectramr.infrastructure.physics.degradations.families import (
    MS,
    VOL,
    exact,
    independent,
    native_2d,
)
from spectramr.infrastructure.physics.degradations.slice2d.blur import (
    degrade_resolution_snr,
    degrade_slice_thickness_pve,
    degrade_t2star_blur,
)
from spectramr.infrastructure.physics.degradations.slice2d.kspace_phase import (
    _delegate_eddy,
    degrade_iq_ghost,
    degrade_nyquist_ghost,
    degrade_stimulated_echo,
)
from spectramr.infrastructure.physics.degradations.slice2d.noise import _delegate_spike
from spectramr.infrastructure.physics.degradations.slice2d.sampling_cartesian import _delegate_gibbs
from spectramr.infrastructure.physics.degradations.spec import ActivationGate, Realisation
from spectramr.infrastructure.physics.degradations.volume.kspace_phase import (
    eddy_volume,
    gibbs_volume,
    iq_ghost_volume,
)
from spectramr.infrastructure.physics.degradations.volume.noise import (
    resolution_snr_volume,
    spike_volume,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam

ENTRIES = (
    native_2d(
        code="D8",
        name="nyquist_ghost",
        family="field",
        fn=degrade_nyquist_ghost,
        description="EPI N/2 ghost",
        severity=SeveritySpec(
            (_P("phase_offset", "rad", 0.0, math.pi),),
            notes="EPI even/odd phase offset; theta=1 is a pi offset (max N/2 ghost).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_nyquist_ghost), VOL: exact(degrade_nyquist_ghost)},
        active_when=frozenset({ActivationGate.SEQUENCE_EPI}),
    ),
    # NOTE: retained fraction *decreases* with severity.
    native_2d(
        code="D9",
        name="gibbs",
        family="field",
        fn=_delegate_gibbs,
        description="Gibbs ringing (delegated)",
        severity=SeveritySpec(
            (_P("truncation_radius", "fraction of the max k-space radius", 1.0, 0.085),),
            notes="symmetric circular k-space truncation; less retained = stronger "
            "ringing. The radius is normalised by its own maximum (the grid CORNER), "
            "so 1.0 is a genuine identity — before #246 the window compared an "
            "un-normalised radius (max sqrt(2)) against the fraction and zeroed 22.8% "
            "of k-space at the nominal 'no truncation' setting.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(_delegate_gibbs), VOL: (gibbs_volume, Realisation.NATIVE)},
    ),
    native_2d(
        code="D12",
        name="t2star_blur",
        family="field",
        fn=degrade_t2star_blur,
        description="T2*-induced Gaussian blur",
        severity=SeveritySpec(
            (_P("readout_over_t2prime", "T_read / T2'", 0.0, 4.0),),
            notes="spin-echo relaxation during the readout: k-space weight exp(-r|f|) along "
            "readout only, a Lorentzian PSF of FWHM r/pi px (1.27 px at theta=1).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_t2star_blur), VOL: exact(degrade_t2star_blur)},
    ),
    native_2d(
        code="D13",
        name="slice_pve",
        family="field",
        fn=degrade_slice_thickness_pve,
        description="Slice-thickness PVE proxy (2-D only; volumes use through_plane_pve)",
        severity=SeveritySpec(
            (_P("sigma", "px", 0.0, 5.0),),
            notes="NOT a through-plane model (that needs the third dimension): a readout-axis "
            "Gaussian blur proxy (max_kernel_px 20 / 4), applied as a k-space window.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
    ),
    native_2d(
        code="D14",
        name="eddy",
        family="field",
        fn=_delegate_eddy,
        description="Eddy current (delegated)",
        severity=SeveritySpec(
            (
                _P("linear_phase_scale", "randn std, x pi rad", 0.0, 5.0),
                _P("quadratic_phase_scale", "randn std, x pi rad", 0.0, 2.5),
            ),
            notes="per-PE-line phase = pi*(alpha*u + beta*u^2), u in [-1, 1], "
            "alpha ~ N(0, scale), beta ~ N(0, 0.5*scale). The LINEAR term is a global PE "
            "translation (a reconstructor recovers it); the QUADRATIC (chirp) term is the "
            "genuine shear/ghost. The declared value is the randn STD of the pi-scaled "
            "coefficients, NOT a per-line rad increment (~0.12 rad/line at scale=5) -- the "
            "prior 'rad per PE line' label was ~40x off (advisor).",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes={MS: exact(_delegate_eddy), VOL: (eddy_volume, Realisation.NATIVE)},
    ),
    native_2d(
        code="D15",
        name="spike",
        family="field",
        fn=_delegate_spike,
        description="Spike noise (delegated)",
        severity=SeveritySpec(
            (_P("probability", "per k-space sample", 0.0, 0.05),),
            notes="RF spike / herringbone. Each spike is complex Gaussian with RMS 5x the "
            "sample's k-space RMS, at the same k-space position in every coil. The added "
            "energy is linear in the probability, so the residual grows as its square root.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes={MS: independent(_delegate_spike), VOL: (spike_volume, Realisation.NATIVE)},
    ),
    native_2d(
        code="D25",
        name="iq_ghost",
        family="field",
        fn=degrade_iq_ghost,
        description="Quadrature-receiver DC + I/Q ghost",
        severity=SeveritySpec(
            (
                _P("dc_offset", "fraction", 0.0, 0.20),
                _P("iq_imbalance", "fraction", 0.0, 0.35),
            ),
            notes="receiver DC offset and I/Q gain imbalance ramp together.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.SCANNER,
        modes={
            MS: independent(degrade_iq_ghost),
            VOL: (iq_ghost_volume, Realisation.NATIVE),
        },
    ),
    native_2d(
        code="D26",
        name="stimulated_echo",
        family="field",
        fn=degrade_stimulated_echo,
        description="Stimulated-echo TSE ghost",
        severity=SeveritySpec(
            (
                _P("amplitude", "fraction", 0.0, 0.40),
                _P("shift", "px", 0.0, 6.0),
            ),
            notes="spurious stimulated-echo pathway; amplitude and shift ramp together. The "
            "realised shift is a seeded 0.7-1.0 of the declared shift.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={MS: exact(degrade_stimulated_echo), VOL: exact(degrade_stimulated_echo)},
    ),
    native_2d(
        code="D28",
        name="resolution_snr",
        family="field",
        fn=degrade_resolution_snr,
        description="Coupled effective-resolution loss + noise (no resampling)",
        severity=SeveritySpec(
            (
                _P("eff_fwhm", "px", 0.0, 2.5),
                _P("noise_sigma", "fraction of clean k-space RMS", 0.0, 0.0707),
            ),
            notes="coupled effective-resolution loss (k-space Gaussian apodization, no resampling) + noise on ONE theta. noise_sigma is referenced to the CLEAN, un-apodized k-space RMS: 20 dB at theta=1 against that reference. The output SNR is lower, because the apodization removes signal and not the noise added after it, and how much lower depends on the image's spectrum.",
        ),
        affects=BOTH,
        origin=ArtifactOrigin.ACQUISITION,
        modes={
            MS: (resolution_snr_volume, Realisation.NATIVE),
            VOL: (resolution_snr_volume, Realisation.NATIVE),
        },
    ),
)
