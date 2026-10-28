"""Witness: the ``acs`` normalization band is acquired at every rung.

``data.processing.kspace_scale_domain: acs`` measures the k-space scale over the
centred ``log_scaling_center_fraction`` band of phase-encode lines, so the scale
the dataloader stamps from the fully sampled reference equals the one inference
computes from the undersampled measurement -- but only if every line of the band
survives to the last rung. The ladder is nested, so that is one inclusion,
``band ⊆ mask(T-1)``, checked on the mask the config actually builds.
"""

from __future__ import annotations

from spectramr.data.transforms.normalization import acs_band_rows
from spectramr.infrastructure.validation.witness.checks.schedule_certification_common import (
    _CATEGORY,
    _matrix,
    _timesteps,
    build_process_from_config,
    phase_encode_axis_of,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

_NAME = "normalization.acs_scale_band_acquired"
_KEYS = ("data.processing.kspace_scale_domain", "data.processing.log_scaling_center_fraction")


def _verdict(passed: bool, message: str, severity: Severity = Severity.WARNING) -> WitnessVerdict:
    return WitnessVerdict(
        witness_name=_NAME,
        passed=passed,
        message=message,
        severity=severity,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=Tier.T1,
        yaml_keys=_KEYS,
    )


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.CONSTRUCT,
    tiers=(Tier.T1,),
    subjects=(Subject.CONFIG,),
    severity=Severity.WARNING,
    description="The acs normalization band lies inside the top rung's mask",
    fix_hint=(
        "Lower data.processing.log_scaling_center_fraction until its band of "
        "phase-encode lines fits inside undersampling.min_center_fraction's core."
    ),
)
def acs_scale_band_acquired(subject: WitnessSubject) -> WitnessVerdict:
    """An ``acs`` band line missing from ``mask(T-1)`` makes the scale rung-dependent."""
    processing = (subject.raw_config.get("data") or {}).get("processing") or {}
    if processing.get("kspace_scale_domain") != "acs":
        return _verdict(True, "kspace_scale_domain is not 'acs'; nothing to check", Severity.INFO)
    process = build_process_from_config(subject.raw_config)
    if process is None:
        return _verdict(
            True,
            "kspace_scale_domain 'acs' on an arm with no k-space ladder; the band "
            "cannot be checked against an acquisition",
            Severity.INFO,
        )
    axis = phase_encode_axis_of(subject.raw_config)
    if axis is None:
        return _verdict(
            False,
            "kspace_scale_domain 'acs' reads a band of phase-encode lines, and this arm "
            "declares no data.acquisition.phase_encode_axis to place it on.",
        )
    shape = _matrix(subject.raw_config)
    # The band's lines run along the phase-encode axis; a line is acquired when
    # the top rung keeps it whole across the other axis.
    n_lines = shape[0] if axis == -2 else shape[1]
    word = "rows" if axis == -2 else "columns"
    lines = acs_band_rows(n_lines, float(processing.get("log_scaling_center_fraction", 0.25)))
    top = _timesteps(subject.raw_config) - 1
    mask = process.mask_at(top, shape).reshape(shape).bool()
    acquired = mask.all(dim=1) if axis == -2 else mask.all(dim=0)
    missing = [r for r in range(lines.start, lines.stop) if not bool(acquired[r])]
    band = f"{word} {lines.start}..{lines.stop - 1} of {n_lines}"
    if not missing:
        return _verdict(True, f"acs band ({band}) is fully acquired at t={top}")
    return _verdict(
        False,
        f"acs band ({band}) misses {len(missing)} phase-encode lines at t={top} "
        f"(first: {word[:-1]} {missing[0]}); the scale then depends on the rung, and "
        "inference computes a different one than training stamped.",
    )


__all__ = ["acs_scale_band_acquired"]
