"""Scanner profiles: the field- and sequence-dependent facts that condition the degradations.

A :class:`ScannerProfile` bundles the acquisition parameters that decide WHICH artefacts a
scanner produces and HOW STRONG they are. ``apply_degradation(..., profile=p)`` hands it to
each spec's ``condition`` (``degradations/conditions.py``), and :func:`active_axes` reads
each spec's ``active_when`` gates. It lives in the physics layer, next to the formulas it
drives; ``scripts/sim2rank`` passes it in, so it never reaches ``core/``.

Why conditioning is **theta-aware**, and the trap it avoids
-----------------------------------------------------------

Each axis's worst-case physical amount is computed from the profile, then **scaled by
theta**, so the axis stays monotone in severity *and* differs between scanners (0.3 T
fat-water shift is ~10x smaller than 3 T). Passing the raw scanner parameters straight to
a degrader is the trap that made ``chemical_shift`` theta-INVARIANT:
``_delegate_chemical_shift`` ignores ``shift_pixels`` and computes a fixed physical shift
the moment ``b0_T``/``bw`` are supplied. The conditions therefore pass a theta-scaled
``shift_pixels`` and never the raw ``b0_T``/``bw``.

Scope
-----

An axis is conditioned when its spec declares a ``condition`` (``degradations/conditions.py``
and the catalogue ``*_conditions`` modules); an axis without one keeps its default theta ramp.
The presets live in ``scanner_presets``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from enum import Enum, StrEnum
from typing import Any

from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    SliceOrder,
)
from spectramr.infrastructure.physics.degradations.registry import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.degradations.spec import ActivationGate
from spectramr.infrastructure.physics.digital_twin_simulator import (
    DEFAULT_CHI_PPM,
    chemical_shift_pixels,
)

__all__ = [
    "DEFAULT_CHI_PPM",
    "ScannerProfile",
    "Sequence",
    "active_axes",
]

#: Field strength treated as the un-scaled reference for the B0 field-map scale.
_REFERENCE_B0_T = 3.0


class Sequence(StrEnum):
    """The pulse-sequence family a profile acquires with."""

    SPIN_ECHO = "spin_echo"
    TSE = "tse"
    GRE = "gre"
    SPOILED_GRE = "spoiled_gre"
    IR_TSE = "ir_tse"
    MPRAGE = "mprage"
    BSSFP = "bssfp"
    EPI = "epi"
    RADIAL = "radial"
    SPIRAL = "spiral"


def _positive(value: float | None) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True, slots=True)
class ScannerProfile:
    """Acquisition parameters of one scanner; ``None`` on an optional fact means unknown."""

    name: str
    b0_T: float  # noqa: N815 -- B0 in tesla
    readout_bw_hz_per_px: float
    sequence: Sequence
    geometry: AcquisitionGeometry
    tr_ms: float | None
    te_ms: float | None
    ti_ms: float | None
    fa_deg: float | None
    etl: int = 1
    esp_ms: float | None = None
    #: Reduction factors (phase-encode, slice/kz).
    acceleration: tuple[float, float] = (1.0, 1.0)
    n_coils: int = 1
    coil_covariance: tuple[tuple[float, ...], ...] | None = None
    chi_ppm: float = DEFAULT_CHI_PPM  # fat-water chemical-shift difference
    fat_fraction: float = 0.1
    gradient_max_mT_per_m: float | None = None  # noqa: N815
    slew: float | None = None  # T/m/s
    #: Explicit artefact allow-list; ``None`` derives it from the protocol.
    active_axes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        bad = [
            f"{k}={getattr(self, k)!r}"
            for k in ("b0_T", "readout_bw_hz_per_px", "chi_ppm")
            if not _positive(getattr(self, k))
        ]
        for key in ("tr_ms", "te_ms", "ti_ms", "esp_ms", "gradient_max_mT_per_m", "slew"):
            value = getattr(self, key)
            if value is not None and not _positive(value):
                bad.append(f"{key}={value!r}")
        if self.fa_deg is not None and not (_positive(self.fa_deg) and self.fa_deg <= 180):
            bad.append(f"fa_deg={self.fa_deg!r}")
        if self.tr_ms is not None and self.te_ms is not None and self.te_ms >= self.tr_ms:
            bad.append(f"te_ms={self.te_ms!r} >= tr_ms={self.tr_ms!r}")
        if not (_count(self.etl) and _count(self.n_coils)):
            bad.append(f"etl={self.etl!r}, n_coils={self.n_coils!r}")
        acc = self.acceleration
        if not (
            isinstance(acc, tuple) and len(acc) == 2 and all(_positive(r) and r >= 1 for r in acc)
        ):
            bad.append(f"acceleration={acc!r}")
        if not 0.0 <= self.fat_fraction <= 1.0:
            bad.append(f"fat_fraction={self.fat_fraction!r}")
        if not isinstance(self.sequence, Sequence):
            bad.append(f"sequence={self.sequence!r} (must be a Sequence)")
        if not isinstance(self.geometry, AcquisitionGeometry):
            bad.append(f"geometry={self.geometry!r} (must be an AcquisitionGeometry)")
        cov = self.coil_covariance
        if cov is not None and not (
            len(cov) == self.n_coils
            and all(len(row) == self.n_coils for row in cov)
            and all(cov[i][j] == cov[j][i] for i in range(self.n_coils) for j in range(i))
            and all(cov[i][i] > 0 for i in range(self.n_coils))
        ):
            bad.append(
                "coil_covariance must be a symmetric n_coils x n_coils positive-diagonal matrix"
            )
        if self.active_axes is not None:
            unknown = sorted(set(self.active_axes) - set(DEGRADATION_REGISTRY))
            if unknown:
                bad.append(f"active_axes has unknown degradations {unknown}")
        if bad:
            raise ValueError(f"ScannerProfile {self.name!r}: invalid {'; '.join(bad)}")

    def chemical_shift_max_px(self) -> float:
        """Worst-case fat-water displacement ``gamma B0 chi / BW_px`` in pixels (linear in B0)."""
        return chemical_shift_pixels(self.b0_T, self.readout_bw_hz_per_px, self.chi_ppm)

    def b0_field_scale(self) -> float:
        """Off-resonance scales ~linearly with B0; 3.0 T is the reference (== 1.0)."""
        return self.b0_T / _REFERENCE_B0_T

    def readout_dwell_s(self, n_readout: int) -> float:
        """ADC dwell per readout sample, ``1 / (BW_px * n_readout)``."""
        return 1.0 / (self.readout_bw_hz_per_px * n_readout)

    def to_meta(self) -> dict[str, object]:
        """Every field as a plain value (enums as strings, geometry via its own ``to_meta``)."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    def meta_sha256(self) -> str:
        """SHA-256 of :meth:`to_meta` as sorted JSON: names the whole profile in a capped record."""
        return hashlib.sha256(json.dumps(self.to_meta(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_acquisition_config(cls, acq: Any, *, field_strength_t: float) -> ScannerProfile:
        """The profile a ``physics.digital_twin.acquisition`` block declares, at ``field_strength_t``.

        The ``scanner_profile`` preset supplies the defaults and every non-null field overrides
        it; without a preset the geometry starts from a 2-D slice of unknown voxel size.
        """
        geometry = {
            k: cast(getattr(acq, k))
            for k, cast in _GEOMETRY_FIELDS.items()
            if getattr(acq, k) is not None
        }
        facts: dict[str, Any] = {"b0_T": float(field_strength_t)}
        if acq.readout_bw_hz_per_px is not None:
            facts["readout_bw_hz_per_px"] = float(acq.readout_bw_hz_per_px)
        if acq.sequence is not None:
            facts["sequence"] = Sequence(acq.sequence)
        if acq.scanner_profile is not None:
            # Call-time import: the presets module builds ScannerProfile instances.
            from spectramr.infrastructure.physics.scanner_presets import resolve_profile

            base = resolve_profile(acq.scanner_profile)
            return replace(base, geometry=replace(base.geometry, **geometry), **facts)
        geom = AcquisitionGeometry(
            **{"mode": AcquisitionMode.SLICE_2D, "voxel_mm": None, **geometry}
        )
        return cls(
            name="custom", geometry=geom, tr_ms=None, te_ms=None, ti_ms=None, fa_deg=None, **facts
        )


#: Acquisition-block field -> the geometry value it becomes.
_GEOMETRY_FIELDS: dict[str, Callable[[Any], Any]] = {
    "mode": AcquisitionMode,
    "voxel_mm": tuple,
    "slice_thickness_mm": float,
    "slice_gap_mm": float,
    "slice_order": SliceOrder,
    "line_order": LineOrder,
}


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, AcquisitionGeometry):
        return value.to_meta()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


_GATE_MET: dict[ActivationGate, Callable[[ScannerProfile], bool]] = {
    ActivationGate.SEQUENCE_EPI: lambda p: p.sequence is Sequence.EPI,
    ActivationGate.ACCELERATED: lambda p: max(p.acceleration) > 1.0,
    ActivationGate.FAT_PRESENT: lambda p: p.fat_fraction > 0.0,
}


def active_axes(profile: ScannerProfile, candidates: Iterable[str]) -> list[str]:
    """The subset of ``candidates`` a scanner with this profile actually produces.

    Each degradation's ``active_when`` gates must all hold for the profile; an explicit
    ``profile.active_axes`` overrides the derivation. An unknown candidate raises.
    """
    cand = list(candidates)
    unknown = sorted(set(cand) - set(DEGRADATION_REGISTRY))
    if unknown:
        raise KeyError(f"Unknown degradations {unknown}. Valid: {sorted(DEGRADATION_REGISTRY)}")
    if profile.active_axes is not None:
        allowed = set(profile.active_axes)
        return [a for a in cand if a in allowed]
    return [
        a for a in cand if all(_GATE_MET[g](profile) for g in DEGRADATION_REGISTRY[a].active_when)
    ]
