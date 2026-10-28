"""The flip-angle response of the image signal to a B1+ scale map: one owner of the equation.

A transmit scale ``kappa(r)`` makes every nominal flip angle ``kappa * alpha``. The signal ratio
(inhomogeneous / nominal) is looked up by sequence, so ``b1_transmit`` and
``dielectric_b1`` share it:

* GRE / spoiled GRE: with TR and T1 the Ernst steady state ``S(a) = sin(a)(1 - E1) / (1 - E1
  cos a)``, ``E1 = exp(-TR / T1)``, so the ratio is ``S(kappa a) / S(a)`` (Ernst & Anderson 1966;
  Bernstein, Handbook of MRI Pulse Sequences, 2004). Without a TR the long-TR limit (TR >> T1,
  full relaxation, E1 -> 0) ``sin(kappa a) / sin(a)`` applies: the excitation-only ratio.
* SE / TSE: excitation x refocusing^2, ``sin(kappa 90) sin^2(kappa 180 / 2)`` = ``sin^3(kappa 90)``
  relative to 1 at ``kappa = 1`` (Hennig 1991 for the refocusing factor ``sin^2(a / 2)``). The
  flip angle and TR are fixed by the sequence, so giving either raises.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from spectramr.infrastructure.physics.digital_twin_simulator import gre_flip_modulation

__all__ = [
    "DEFAULT_SEQUENCE",
    "DEFAULT_T1_MS",
    "SIGNAL_MODELS",
    "SUPPORT_FRACTION",
    "apply_transmit",
    "calibrated_kappa",
    "check_signal_args",
    "lift_maps",
    "object_support",
    "signal_ratio",
]

#: Declared defaults of a call without a scanner profile: a spin echo (the sequence of both scanner
#: presets; a GRE must name its flip angle) and a 1 s T1 used only when a TR gives the Ernst form.
DEFAULT_SEQUENCE = "spin_echo"
DEFAULT_T1_MS = 1000.0
_SE_EXCITATION_DEG = 90.0
_SE_REFOCUSING_DEG = 180.0
#: A voxel belongs to the object support above this fraction of the sample's peak magnitude.
SUPPORT_FRACTION = 0.05


def _gre(kappa: Tensor, fa_deg: float | None, tr_ms: float | None, t1_ms: Any) -> Tensor:
    alpha = math.radians(fa_deg)
    if tr_ms is None:
        return gre_flip_modulation(kappa, alpha)
    e1 = torch.exp(-tr_ms / torch.as_tensor(t1_ms, dtype=kappa.dtype, device=kappa.device))
    ernst = torch.sin(alpha * kappa) / (1.0 - e1 * torch.cos(alpha * kappa))
    return ernst * (1.0 - e1 * math.cos(alpha)) / math.sin(alpha)


def _spin_echo(kappa: Tensor, fa_deg: float | None, tr_ms: float | None, t1_ms: Any) -> Tensor:
    exc = torch.sin(kappa * math.radians(_SE_EXCITATION_DEG))
    return exc * torch.sin(kappa * math.radians(_SE_REFOCUSING_DEG) / 2.0) ** 2


#: Models whose response depends on the nominal flip angle (a spin echo's is fixed).
NEEDS_FA = frozenset({"gre", "spoiled_gre"})
_MAX_GRE_FA_DEG = 120.0
SIGNAL_MODELS: Mapping[str, Callable[..., Tensor]] = {
    "gre": _gre,
    "spoiled_gre": _gre,
    "spin_echo": _spin_echo,
    "tse": _spin_echo,
}


def check_signal_args(sequence: str, fa_deg: float | None, tr_ms: float | None, t1_ms: Any) -> None:
    """Raise unless ``sequence`` has a transmit model and the numbers are physical."""
    if str(sequence) not in SIGNAL_MODELS:
        raise ValueError(
            f"no transmit signal model for sequence {str(sequence)!r}; "
            f"modelled: {sorted(SIGNAL_MODELS)}"
        )
    if str(sequence) in NEEDS_FA:
        if fa_deg is None:
            raise ValueError(f"the {str(sequence)!r} transmit model needs fa_deg (no default)")
        if not 0.0 < fa_deg < _MAX_GRE_FA_DEG:
            raise ValueError(
                f"fa_deg must be in (0, {_MAX_GRE_FA_DEG}) so that kappa * fa stays below 180 "
                f"degrees for kappa <= 1.5, got {fa_deg!r}"
            )
    elif fa_deg is not None or tr_ms is not None:
        raise ValueError(
            f"fa_deg and tr_ms are not modelled by this axis for sequence {str(sequence)!r}; "
            f"fa_deg={fa_deg!r}, tr_ms={tr_ms!r} would be ignored"
        )
    if tr_ms is not None and not tr_ms > 0.0:
        raise ValueError(f"tr_ms must be positive, got {tr_ms!r}")
    # A T1 map is not inspected: reading its minimum would be a host sync.
    if isinstance(t1_ms, int | float) and not t1_ms > 0.0:
        raise ValueError(f"t1_ms must be positive, got {t1_ms!r}")


def signal_ratio(
    kappa: Tensor,
    *,
    sequence: str = DEFAULT_SEQUENCE,
    fa_deg: float | None = None,
    tr_ms: float | None = None,
    t1_ms: Any = DEFAULT_T1_MS,
) -> Tensor:
    """Signal(kappa) / signal(1) for the sequence's transmit model."""
    check_signal_args(sequence, fa_deg, tr_ms, t1_ms)
    return SIGNAL_MODELS[str(sequence)](kappa, fa_deg, tr_ms, t1_ms)


def apply_transmit(x: Tensor, kappa: Tensor, **signal: Any) -> Tensor:
    """``x`` times the signal ratio of the B1+ scale map ``kappa`` (broadcast over channels)."""
    return x * signal_ratio(kappa, **signal)


def lift_maps(kwargs: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """A 2-D call's rank-4 ``(B, 1, H, W)`` maps among ``names`` gain a slice axis (5-D path)."""
    out = dict(kwargs)
    for name in names:
        m = out.get(name)
        if m is not None and torch.as_tensor(m).dim() == 4:
            out[name] = torch.as_tensor(m).unsqueeze(2)
    return out


def object_support(x: Tensor) -> Tensor:
    """``[B, 1, D, H, W]`` mask of voxels above ``SUPPORT_FRACTION`` of the sample's peak."""
    mag = x.abs().amax(dim=1, keepdim=True)
    return mag > SUPPORT_FRACTION * mag.amax(dim=(-3, -2, -1), keepdim=True)


def calibrated_kappa(full: Tensor, support: Tensor, theta: float) -> Tensor:
    """``1 + theta (full / <full>_support - 1)`` in the support, 1 outside it.

    The prescan transmit calibration sets the reference voltage so that the mean flip angle over
    the excited object is nominal, so only the spatial pattern of a B1+ map remains.
    """
    w = support.to(full.dtype)
    dims = (-3, -2, -1)
    mean = (full * w).sum(dim=dims, keepdim=True) / w.sum(dim=dims, keepdim=True).clamp(min=1.0)
    return 1.0 + theta * torch.where(support, full / mean - 1.0, torch.zeros_like(mean))
