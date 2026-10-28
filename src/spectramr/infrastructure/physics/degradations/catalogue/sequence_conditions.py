"""Scanner conditions of the sequence-timing axes: the protocol facts a profile supplies as kwargs.

A fact the profile does not carry (``None``, or a sequence it does not describe) is left out, so
the axis's declared default applies; the severity ramp is never rescaled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = ["t1_shot_condition", "tse_condition"]

#: Sequences whose ``etl`` / ``esp_ms`` describe a turbo-spin-echo train.
_TSE_SEQUENCES = frozenset({"tse", "ir_tse"})
#: IR-prepared / gradient-echo sequences -> the profile field holding their readout-pulse repetition
#: (a GRE's TR; an MPRAGE's echo spacing).
_READOUT_TR_FIELD = {"gre": "tr_ms", "spoiled_gre": "tr_ms", "mprage": "esp_ms"}


def _present(profile: ScannerProfile, mapping: dict[str, str]) -> dict[str, object]:
    return {
        kwarg: getattr(profile, field)
        for kwarg, field in mapping.items()
        if getattr(profile, field) is not None
    }


def tse_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: tuple[int, ...]
) -> dict[str, object]:
    """The echo-train length and spacing of a TSE profile."""
    if str(profile.sequence) not in _TSE_SEQUENCES:
        return {}
    return _present(profile, {"etl": "etl", "esp_ms": "esp_ms"})


def t1_shot_condition(
    profile: ScannerProfile, theta: float, geometry: AcquisitionGeometry, shape: tuple[int, ...]
) -> dict[str, object]:
    """TI, flip angle and readout repetition of an inversion / gradient-echo profile.

    Any other sequence has no such readout, so its ``fa_deg`` (an excitation or refocusing angle) and
    ``ti_ms`` are not used: the declared defaults apply and ``profile_ti_fa_ignored`` records it.
    """
    sequence = str(profile.sequence)
    if sequence not in _READOUT_TR_FIELD:
        return {"profile_ti_fa_ignored": sequence}
    return _present(
        profile, {"ti_ms": "ti_ms", "fa_deg": "fa_deg", "tr_ro_ms": _READOUT_TR_FIELD[sequence]}
    )
