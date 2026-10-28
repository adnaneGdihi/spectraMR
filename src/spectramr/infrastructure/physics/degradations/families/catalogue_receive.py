"""The receive-chain family: correlated coil noise, EMI bursts, gain drift, ADC, anti-alias, dropout.

All seven are native in every mode (SLICE_2D, MULTISLICE_2D, VOLUME_3D): their references (subject
RMS, ADC full scale, scan time) are per subject, so a per-slice adapter would not be exact.
"""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.receive_adc import (
    adc_clipping,
    adc_clipping_2d,
    adc_quantization,
    adc_quantization_2d,
    coil_dropout,
    coil_dropout_2d,
    receiver_gain_drift,
    receiver_gain_drift_2d,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_common import (
    coil_covariance_condition,
    n_coils_condition,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_noise import (
    antialias_rolloff,
    antialias_rolloff_2d,
    correlated_coil_noise,
    correlated_coil_noise_2d,
    emi_burst,
    emi_burst_2d,
)
from spectramr.infrastructure.physics.degradations.declarations import BOTH, MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import (
    DegradationSpec,
    Realisation,
    Requirement,
    SliceKernel,
    entry,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_P = PhysicalParam
_S = AcquisitionMode.SLICE_2D
_SCANNER = ArtifactOrigin.SCANNER


def _native(
    code: str,
    name: str,
    family: str,
    kernel_2d,
    impl,
    *,
    description: str,
    severity: SeveritySpec,
    affects: frozenset[str],
    requires: frozenset[Requirement] = frozenset(),
    condition=None,
) -> DegradationSpec:
    return entry(
        code=code,
        name=name,
        family=family,
        description=description,
        severity=severity,
        affects=affects,
        origin=_SCANNER,
        implementations={_S: SliceKernel(kernel_2d), MS: impl, VOL: impl},
        realisation=dict.fromkeys((_S, MS, VOL), Realisation.NATIVE),
        requires=requires,
        condition=condition,
        active_when=frozenset(),
    )


ENTRIES = (
    _native(
        "D44",
        "correlated_coil_noise",
        "noise",
        correlated_coil_noise_2d,
        correlated_coil_noise,
        description=(
            "Complex Gaussian k-space noise with inter-coil covariance (n = sigma L w, L L^H = "
            "Sigma). Sigma is the scanner profile's coil_covariance; when the profile or the "
            "call supplies none, the declared fallback is the synthetic Sigma_ij = 0.3^|i-j|. "
            "Needs at least two coils."
        ),
        severity=SeveritySpec(
            (_P("noise_rms", "x subject k-space RMS", 0.0, 0.3),),
            notes="noise RMS per complex sample over the whole-subject k-space RMS, as "
            "complex_gaussian; theta = 0 is noise-free.",
        ),
        affects=BOTH,
        requires=frozenset({Requirement.MULTICOIL}),
        condition=coil_covariance_condition,
    ),
    _native(
        "D45",
        "emi_burst",
        "noise",
        emi_burst_2d,
        emi_burst,
        description="Broadband interference on a contiguous run of acquired lines, white across kx",
        severity=SeveritySpec(
            (
                _P("noise_rms", "x subject k-space RMS", 0.0, 0.5),
                _P("burst_lines", "fraction of lines", 0.0, 0.2),
            ),
            notes="theta sets both: the noise RMS on the corrupted lines and the run length.",
        ),
        affects=BOTH,
    ),
    _native(
        "D46",
        "receiver_gain_drift",
        "field",
        receiver_gain_drift_2d,
        receiver_gain_drift,
        description=(
            "Uncorrected receiver gain-stage switch (up to 6 dB, seeded time and direction) plus "
            "a slow linear drift, both referenced to the k-space-centre gain"
        ),
        severity=SeveritySpec(
            (
                _P("gain_step", "dB of gain step", 0.0, 6.0),
                _P("gain_drift", "fraction of gain over the scan", 0.0, 0.05),
            ),
            notes="g = (1 + 0.05 theta t/T) 2^(+-theta [t >= t0]) divided by g(centre line): the "
            "step is 2^theta (6 dB) at theta = 1 on the lines after the seeded switch.",
        ),
        affects=BOTH,
    ),
    _native(
        "D47",
        "adc_clipping",
        "field",
        adc_clipping_2d,
        adc_clipping,
        description="ADC saturation: Re and Im of k-space clipped below the subject's peak",
        severity=SeveritySpec(
            (_P("clip_level", "fraction of peak |Re/Im|", 1.0, 0.1),),
            notes="clip level = 1 - 0.9 theta of the per-subject k-space peak; theta = 0 clips "
            "nothing.",
        ),
        affects=BOTH,
    ),
    _native(
        "D48",
        "adc_quantization",
        "noise",
        adc_quantization_2d,
        adc_quantization,
        description=(
            "Uniform ADC quantisation of Re and Im at a continuous effective number of bits "
            "(ENOB = 16 - 12 theta) over the subject's full scale"
        ),
        severity=SeveritySpec(
            (_P("enob", "effective bit", 16.0, 4.0),),
            notes="ENOB = 16 - 12 theta, real-valued; delta = P / 2^(ENOB - 1) for full scale P; "
            "theta = 0 bypasses the quantiser exactly.",
        ),
        affects=BOTH,
    ),
    _native(
        "D49",
        "antialias_rolloff",
        "field",
        antialias_rolloff_2d,
        antialias_rolloff,
        description=(
            "Receiver anti-alias Butterworth (n = 4) roll-off; the readout frequency maps to "
            "readout position, so the response weights the image along RO (Bernstein et al., "
            "Handbook of MRI Pulse Sequences, 2004, ch. 13). Cutoff x_c = 0.6 / theta in units of "
            "the band edge: infinite as theta -> 0+, so the response tends continuously to the "
            "identity, and 0.6 at theta = 1"
        ),
        severity=SeveritySpec(
            (_P("inverse_cutoff", "band edge / cutoff", 0.0, 1.0 / 0.6),),
            notes="cutoff x_c = 0.6 / theta (1 / x_c linear in theta); theta = 0 is no filter. The "
            "magnitude response weights the image along the readout axis.",
        ),
        affects=MAG,
    ),
    _native(
        "D50",
        "coil_dropout",
        "field",
        coil_dropout_2d,
        coil_dropout,
        description="A seeded subset of coils dead for the scan or from a seeded line on",
        severity=SeveritySpec(
            (_P("dropped_coils", "fraction of coils", 0.0, 0.5),),
            notes="ceil(theta/2 C) coils, at most C // 2, so the count is integer-valued and plateaus "
            "in theta (C = 4: one coil for every theta in (0, 0.5]); mode full or partial (kwarg).",
        ),
        affects=BOTH,
        requires=frozenset({Requirement.MULTICOIL}),
        condition=n_coils_condition,
    ),
)
