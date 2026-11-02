r"""Expected Cramér-Rao bound of an MRF pulse train over a T1/T2 prior (MRF §4).

The signal model is inversion-recovery FISP simulated with the extended phase graph (EPG),
the sequence of Jiang et al. [2] in the EPG formalism of Weigel [3]. Each repetition applies
the RF mixing about :math:`y` (which keeps every configuration state real), relaxes over
:math:`T_E`, reads :math:`F_0^+`, relaxes over :math:`T_{R,t}-T_E` and dephases by one
configuration order (the unbalanced FISP gradient). T2 therefore shapes the fingerprint
through the stimulated-echo pathways, not only as a constant :math:`e^{-T_E/T_2}` factor.

Tangents with respect to :math:`(\log T_1, \log T_2)` are propagated alongside the states;
the :math:`\log M_0` column is the signal itself (unit :math:`M_0`). With
:math:`F = J^\top J` (unit noise) over :math:`(\log T_1, \log T_2, \log M_0)`, the objective
is the trace of the T1/T2 block of :math:`F^{-1}`, i.e. the inverse Schur complement
:math:`(C - b b^\top / a)^{-1}` with :math:`M_0` a nuisance, as in Zhao et al. [18]: the
normalised bound :math:`\sum_{i\in\{T_1,T_2\}}\operatorname{Var}(\hat\theta_i)/\theta_i^2`.
:func:`~spectramr.infrastructure.physics.fisher_efficient_fingerprinting.cramer_rao_lower_bound`
is the general certificate this specialises (a parity test pins the two).

References:
    [2] Y. Jiang, D. Ma, N. Seiberlich, V. Gulani, M. A. Griswold, "MR fingerprinting using
        fast imaging with steady state precession (FISP) with spiral readout", *MRM*, 74(6),
        2015.
    [3] M. Weigel, "Extended phase graphs: dephasing, RF pulses, and echoes - pure and
        simple", *JMRI*, 41(2), 2015.
    [18] B. Zhao et al., "Optimal experiment design for MRF: Cramér-Rao bound meets spin
         dynamics", *IEEE TMI*, 38(3), 2019.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: Numerical floor on Fisher pivots; a singular design maps to a huge, finite bound.
_DET_FLOOR = 1e-12
#: Highest dephasing order kept. Orders above it are dropped; the truncation error on the
#: objective is pinned by a test against the untruncated graph.
DEFAULT_EPG_STATES = 64


def ir_fisp_signal_and_jacobian(
    pulse: Tensor,
    t1_ms: Tensor,
    t2_ms: Tensor,
    *,
    te_ms: float,
    n_states: int = DEFAULT_EPG_STATES,
) -> tuple[Tensor, Tensor]:
    """Simulate the IR-FISP EPG fingerprint and its log-parameter Jacobian.

    Args:
        pulse: ``[T, 2]`` design, flip angle (rad) and TR (ms) per excitation; every TR must
            exceed ``te_ms`` (callers check this statically).
        t1_ms: ``[P]`` T1 prior samples (ms).
        t2_ms: ``[P]`` T2 prior samples (ms).
        te_ms: Echo time (ms), fixed across the train.
        n_states: Highest dephasing order ``K`` tracked; ``0`` is the fully spoiled limit.

    Returns:
        ``(signal [P, T], jacobian [P, T, 3])``, the Jacobian taken with respect to
        ``(log T1, log T2, log M0)``.
    """
    if pulse.dim() != 2 or pulse.shape[-1] != 2:
        raise ValueError(f"pulse must be [T, 2] (alpha, TR); got {tuple(pulse.shape)}")
    if t1_ms.shape != t2_ms.shape or t1_ms.dim() != 1:
        raise ValueError("t1_ms and t2_ms must be matching 1-D prior samples")
    if te_ms <= 0:
        raise ValueError(f"te_ms must be > 0; got {te_ms}")
    if n_states < 0:
        raise ValueError(f"n_states must be >= 0; got {n_states}")
    k = min(int(n_states), pulse.shape[0])
    alpha = pulse[:, 0]
    sin_a, cos_a = alpha.sin(), alpha.cos()
    c2, s2 = (alpha / 2).cos().pow(2), (alpha / 2).sin().pow(2)
    inv_t1, inv_t2 = 1.0 / t1_ms, 1.0 / t2_ms
    # Relaxation over TE (fixed) and over the rest of each TR, for every prior sample at once.
    te_relax = _relaxation(torch.full_like(alpha[:1], te_ms), inv_t1, inv_t2)  # 4 x [P, 1]
    rest_relax = _relaxation(pulse[:, 1] - te_ms, inv_t1, inv_t2)  # 4 x [P, T]

    # state[d, p, c, j]: d = (value, d/dlog T1, d/dlog T2), c = (F+, F-, Z), j = order.
    state = pulse.new_zeros(3, t1_ms.shape[0], 3, k + 1)
    state[0, :, 2, 0] = -1.0  # ideal inversion
    onehot = state.new_zeros(k + 1)
    onehot[0] = 1.0
    reads = []
    # Sequential by physics; each step is vectorised over tangents, prior samples and orders.
    for t in range(pulse.shape[0]):
        fp, fm, z = state.unbind(dim=2)
        state = torch.stack(
            [
                c2[t] * fp - s2[t] * fm + sin_a[t] * z,
                -s2[t] * fp + c2[t] * fm + sin_a[t] * z,
                -0.5 * sin_a[t] * (fp + fm) + cos_a[t] * z,
            ],
            dim=2,
        )
        state = _relax(state, [r[:, 0] for r in te_relax], onehot)
        reads.append(state[:, :, 0, 0])  # [3, P]
        state = _relax(state, [r[:, t] for r in rest_relax], onehot)
        state = _dephase(state)
    value, d_t1, d_t2 = torch.stack(reads, dim=-1).unbind(0)  # each [P, T]
    return value, torch.stack([d_t1, d_t2, value], dim=-1)


def _relaxation(tau_ms: Tensor, inv_t1: Tensor, inv_t2: Tensor) -> list[Tensor]:
    """``[e1, de1/dlog T1, e2, de2/dlog T2]`` for durations ``tau_ms`` ``[T]`` -> each ``[P, T]``."""
    x1 = tau_ms.unsqueeze(0) * inv_t1.unsqueeze(1)
    x2 = tau_ms.unsqueeze(0) * inv_t2.unsqueeze(1)
    e1, e2 = torch.exp(-x1), torch.exp(-x2)
    return [e1, e1 * x1, e2, e2 * x2]


def _relax(state: Tensor, relax: list[Tensor], onehot: Tensor) -> Tensor:
    """Apply T1/T2 relaxation and Z0 recovery, propagating the two log-parameter tangents."""
    e1, de1, e2, de2 = (r[:, None] for r in relax)  # [P, 1]
    (fp, fm, z), (afp, afm, az), (bfp, bfm, bz) = (s.unbind(1) for s in state.unbind(0))
    value = torch.stack([fp * e2, fm * e2, z * e1 + (1.0 - e1) * onehot], dim=1)
    tan_t1 = torch.stack([afp * e2, afm * e2, az * e1 + (z - onehot) * de1], dim=1)
    tan_t2 = torch.stack([bfp * e2 + fp * de2, bfm * e2 + fm * de2, bz * e1], dim=1)
    return torch.stack([value, tan_t1, tan_t2], dim=0)


def _dephase(state: Tensor) -> Tensor:
    """One unbalanced-gradient period: F+ moves up one order, F- down, F+_0 = conj F-_1."""
    fp, fm, z = state.unbind(dim=2)
    zero = fm.new_zeros((*fm.shape[:-1], 1))
    fm_new = torch.cat([fm[..., 1:], zero], dim=-1)
    fp_new = torch.cat([fm_new[..., :1], fp[..., :-1]], dim=-1)
    return torch.stack([fp_new, fm_new, z], dim=2)


def relative_crlb(jacobian: Tensor) -> Tensor:
    """T1/T2 trace of ``F^{-1}`` per prior sample for a ``[P, T, 3]`` Jacobian (M0 nuisance)."""
    if jacobian.dim() != 3 or jacobian.shape[-1] != 3:
        raise ValueError(f"jacobian must be [P, T, 3]; got {tuple(jacobian.shape)}")
    fisher = jacobian.transpose(-1, -2) @ jacobian  # [P, 3, 3]
    a = fisher[:, 2, 2].clamp_min(_DET_FLOOR)
    b = fisher[:, :2, 2]
    schur = fisher[:, :2, :2] - b.unsqueeze(-1) * b.unsqueeze(-2) / a[:, None, None]
    s11, s22, s12 = schur[:, 0, 0], schur[:, 1, 1], schur[:, 0, 1]
    det = (s11 * s22 - s12 * s12).clamp_min(_DET_FLOOR)
    return (s11 + s22) / det


def expected_relative_crlb(pulse: Tensor, t1_ms: Tensor, t2_ms: Tensor, *, te_ms: float) -> Tensor:
    """Mean relative CRLB of ``pulse`` over the prior samples (the design objective)."""
    _, jac = ir_fisp_signal_and_jacobian(pulse, t1_ms, t2_ms, te_ms=te_ms)
    return relative_crlb(jac).mean()


def sample_log_uniform_prior(
    n: int,
    t1_range_ms: tuple[float, float],
    t2_range_ms: tuple[float, float],
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw ``n`` independent (T1, T2) samples, log-uniform within each range."""
    u = torch.rand(2, n, device=device, generator=generator)
    return _log_interp(u[0], t1_range_ms), _log_interp(u[1], t2_range_ms)


def log_grid_prior(
    n: int,
    t1_range_ms: tuple[float, float],
    t2_range_ms: tuple[float, float],
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """A fixed ``ceil(sqrt(n))``-per-axis log-spaced (T1, T2) grid, for comparable validation."""
    side = max(2, math.ceil(math.sqrt(n)))
    u = torch.linspace(0.0, 1.0, side, device=device)
    t1, t2 = torch.meshgrid(_log_interp(u, t1_range_ms), _log_interp(u, t2_range_ms), indexing="ij")
    return t1.reshape(-1), t2.reshape(-1)


def _log_interp(u: Tensor, bounds: tuple[float, float]) -> Tensor:
    lo, hi = float(bounds[0]), float(bounds[1])
    if not 0.0 < lo < hi:
        raise ValueError(f"prior range must satisfy 0 < lo < hi; got {bounds}")
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)


__all__ = [
    "DEFAULT_EPG_STATES",
    "expected_relative_crlb",
    "ir_fisp_signal_and_jacobian",
    "log_grid_prior",
    "relative_crlb",
    "sample_log_uniform_prior",
]
