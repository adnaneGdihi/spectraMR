r"""Bottomley T1(B0) relaxation table (XField-FM / idea 2 §2.5).

Implements the closed-form approximation

.. math::

    T_1(B_0) = a \cdot B_0^{\alpha} + b

with tissue-class-specific coefficients per
Bottomley et al. *Med. Phys.* 1984 [1] and the Marques 2019 update
[17] for low field. Coefficients are stored as a frozen dataclass —
not user-editable — to preserve cross-paper reproducibility.

The Bottomley table feeds:

- The contrast-consistency loss in idea 2.
- The Bloch-constrained ULF→HF translator in idea 3.
- The Tier-2 ``_probe_field_conditioning_diverges`` audit, which
  cross-checks the network's predicted contrast change against the
  Bloch-simulated change.

Coefficients are quoted *for white matter, gray matter, and CSF in
brain*; extending to other tissue classes is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class TissueClass(str, Enum):
    """Brain tissue classes covered by the Bottomley table."""

    WHITE_MATTER = "white_matter"
    GRAY_MATTER = "gray_matter"
    CSF = "csf"


@dataclass(frozen=True)
class BottomleyCoefficients:
    """T1 = ``a · B0^α + b`` parameters for a single tissue class."""

    a: float
    alpha: float
    b: float


# Coefficients in (a, α, b) with B0 in T and T1 in milliseconds.
# Values are the standard Bottomley-1984 + Marques-2019 reconciled set.
# These reproduce: WM ≈ 712 ms at 1.5 T, ≈ 900 ms at 3 T (canonical).
# CSF α is intentionally small so T1(CSF) is near-field-independent
# (water-like behaviour): change from 0.064 T to 3 T is < 18 %.
_BOTTOMLEY_TABLE: dict[TissueClass, BottomleyCoefficients] = {
    TissueClass.WHITE_MATTER: BottomleyCoefficients(a=620.0, alpha=0.34, b=0.0),
    TissueClass.GRAY_MATTER: BottomleyCoefficients(a=820.0, alpha=0.32, b=0.0),
    TissueClass.CSF: BottomleyCoefficients(a=4400.0, alpha=0.04, b=0.0),
}


def bottomley_t1(b0_T: float, tissue: TissueClass | str) -> float:
    """Return the Bottomley-1984 / Marques-2019 ``T1(B0)`` in milliseconds.

    Args:
        b0_T: Static-field strength in Tesla (must be > 0).
        tissue: Tissue class — either a :class:`TissueClass` enum
            value or its string spelling (``"white_matter"``,
            ``"gray_matter"``, ``"csf"``).

    Returns:
        Predicted ``T1`` in milliseconds.

    Raises:
        ValueError: invalid arguments (negative B0, unknown tissue).
    """
    if b0_T <= 0.0:
        raise ValueError(f"b0_T must be > 0; got {b0_T}")
    try:
        cls = TissueClass(tissue) if isinstance(tissue, str) else tissue
    except ValueError as exc:
        valid = sorted(t.value for t in TissueClass)
        raise ValueError(f"Unknown tissue class {tissue!r}. Valid options: {valid}") from exc
    coeffs = _BOTTOMLEY_TABLE[cls]
    return coeffs.a * (b0_T**coeffs.alpha) + coeffs.b


def bottomley_t1_tensor(b0_T: torch.Tensor, tissue: TissueClass | str) -> torch.Tensor:
    """Batched / differentiable Bottomley ``T1(B0) = a B0^alpha + b`` (ms) — SSOT for the closed
    form used by the A-5.2 physics-field embedding. Same coefficients as :func:`bottomley_t1`.

    Args:
        b0_T: field strength tensor (Tesla), any shape; values clamped > 0.
        tissue: tissue class (enum or string).
    """
    cls = TissueClass(tissue) if isinstance(tissue, str) else tissue
    if cls not in _BOTTOMLEY_TABLE:
        valid = sorted(t.value for t in TissueClass)
        raise ValueError(f"Unknown tissue class {tissue!r}. Valid options: {valid}")
    # Match the scalar bottomley_t1: raise, don't clamp (#9). Host-checked on CPU only: on
    # CUDA the read would sync every training step, and b0 <= 0 gives a NaN/inf the
    # step's non-finite loss guard refuses.
    if not b0_T.is_cuda and bool((b0_T <= 0.0).any()):
        raise ValueError(f"b0_T must be > 0 T; got min={float(b0_T.min()):.4g}.")
    c = _BOTTOMLEY_TABLE[cls]
    return c.a * b0_T**c.alpha + c.b


def bottomley_table_snapshot() -> dict[str, BottomleyCoefficients]:
    """Read-only copy of the underlying table — for logging / reproducibility.

    Returned dict is keyed by the enum value strings so it is JSON-friendly
    and can be embedded in run-summary manifests.
    """
    return {k.value: v for k, v in _BOTTOMLEY_TABLE.items()}


__all__ = [
    "BottomleyCoefficients",
    "TissueClass",
    "bottomley_t1",
    "bottomley_t1_tensor",
    "bottomley_table_snapshot",
]
