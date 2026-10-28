"""t1_shot_blur: longitudinal recovery across the readout train after an inversion (IR / MPRAGE).

After a perfect inversion and a delay TI, Mz is ``M_0 = 1 - 2 exp(-TI / T1)``; each readout pulse of
flip angle ``alpha`` and repetition ``TR_ro`` then drives it towards ``M_ss = (1 - E1) / (1 - E1 cos alpha)``,
``E1 = exp(-TR_ro / T1)``, so the n-th line sees ``M_n = M_ss + (M_0 - M_ss) (E1 cos alpha)^n`` and carries
``M_n sin alpha`` (Mugler & Brookeman, MRM 15:152, 1990, MPRAGE; Deichmann et al., NeuroImage 12:112, 2000 for the
steady-state recursion). A train of ``N_train = theta T1 / TR_ro`` pulses spans the inner loop, so line ``j``
of ``L`` inner-loop lines sees pulse ``p = j N_train / L`` (at ``N_train = L`` that is the recursion itself).
The weights ``w = 1 + (M_p - M_centre) / M_ss`` (the deviation from constant weighting, scaled by the
positive readout steady state, so never singular; the common ``sin alpha`` cancels) filter k-space: blur or
ringing along the inner loop. TI is the delay from the inversion to the first readout pulse.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import shot_index
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_common import (
    apply_line_weights,
    check_positive,
    ordered_line_times,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

__all__ = [
    "DEFAULT_FA_DEG",
    "DEFAULT_T1_MS",
    "DEFAULT_TI_MS",
    "DEFAULT_TR_RO_MS",
    "degrade_t1_shot_blur",
    "mz_recursion",
    "t1_shot_blur_volume",
    "t1_weights",
]

#: Declared defaults when neither a kwarg nor a profile gives them.
DEFAULT_T1_MS = 1000.0
DEFAULT_TI_MS = 900.0
DEFAULT_FA_DEG = 5.0
DEFAULT_TR_RO_MS = 7.0


def _steady_state(t1_ms: float, fa_deg: float, tr_ro_ms: float) -> float:
    e1 = math.exp(-tr_ro_ms / t1_ms)
    return (1.0 - e1) / (1.0 - e1 * math.cos(math.radians(fa_deg)))


def mz_recursion(p: Tensor, t1_ms: float, ti_ms: float, fa_deg: float, tr_ro_ms: float) -> Tensor:
    """``M_p = M_ss + (M_0 - M_ss) (E1 cos alpha)^p`` before pulse ``p`` (float64, ``p`` may be fractional)."""
    ratio = math.exp(-tr_ro_ms / t1_ms) * math.cos(math.radians(fa_deg))
    m_ss = _steady_state(t1_ms, fa_deg, tr_ro_ms)
    m_0 = 1.0 - 2.0 * math.exp(-ti_ms / t1_ms)
    return m_ss + (m_0 - m_ss) * torch.exp(p.to(torch.float64) * math.log(ratio))


def t1_weights(
    j: Tensor, inner: int, theta: float, t1_ms: float, ti_ms: float, fa_deg: float, tr_ro_ms: float
) -> Tensor:
    """``1 + (M_p - M_centre) / M_ss`` for inner-loop positions ``j`` ``[D, H]``, ``p = j N_train / inner``."""
    n_train = theta * t1_ms / tr_ro_ms
    m = mz_recursion(j * (n_train / inner), t1_ms, ti_ms, fa_deg, tr_ro_ms)
    centre = m[j.shape[0] // 2, j.shape[1] // 2]
    return (1.0 + (m - centre) / _steady_state(t1_ms, fa_deg, tr_ro_ms)).to(torch.float32)


def t1_shot_blur_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    t1_ms: float = DEFAULT_T1_MS,
    ti_ms: float = DEFAULT_TI_MS,
    fa_deg: float = DEFAULT_FA_DEG,
    tr_ro_ms: float = DEFAULT_TR_RO_MS,
    order: str | None = None,
    profile_ti_fa_ignored: str | None = None,
) -> Tensor:
    # ``profile_ti_fa_ignored`` only records, in the realised kwargs, that the profile's sequence has no
    # inversion / gradient-echo readout, so its TI and flip angle were not used (the defaults were).
    if profile_ti_fa_ignored is not None and not isinstance(profile_ti_fa_ignored, str):
        raise ValueError(
            f"profile_ti_fa_ignored must be a sequence name, got {profile_ti_fa_ignored!r}"
        )
    check_input(x, geometry, "t1_shot_blur")
    t1_ms, ti_ms, tr_ro_ms = (
        check_positive(k, v)
        for k, v in (("t1_ms", t1_ms), ("ti_ms", ti_ms), ("tr_ro_ms", tr_ro_ms))
    )
    if not 0 < check_positive("fa_deg", fa_deg) < 90:
        raise ValueError(f"fa_deg must be in (0, 90) degrees, got {fa_deg}")
    d, h = x.shape[-3:-1]
    times, inner = ordered_line_times(geometry, d, h, order)
    if theta == 0:
        return x
    # Weights depend on geometry only: built on the CPU, moved to the device once by apply_line_weights.
    j = shot_index(times, times.numel()) % inner
    w = t1_weights(j, inner, theta, t1_ms, ti_ms, fa_deg, tr_ro_ms)
    return apply_line_weights(x, w, geometry.mode)


def degrade_t1_shot_blur(x: Tensor, theta: float, seed: int = 0, **kwargs) -> Tensor:
    """2-D slice ``(B, C, H, W)``: one inversion, one readout train over the phase-encode lines."""
    return run_as_volume(t1_shot_blur_volume, x, theta, seed, **kwargs)
