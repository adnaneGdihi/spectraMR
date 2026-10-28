"""1D Teichmüller / square-root-velocity time reparameterisation.

Shared infrastructure for:

* fMRI §5 — HRF manifold diffusion with conformal time reparameterisation
* MRF §5 — Cross-scanner MRF harmonisation as a per-scanner τ_s

The space of orientation-preserving diffeomorphisms of ``[0, T]``
with the Fisher-Rao metric is a homogeneous space whose geodesics
are the square-root-velocity reparameterisations of Srivastava
*et al.* [27]. The SRV transform
:math:`q(t) = \\mathrm{sgn}(\\tau'(t)) \\sqrt{|\\tau'(t)|}`
is an isometry from this space to the unit hemisphere of
:math:`L^2([0, T])`, so geodesics on the original space are
straight lines in SRV coordinates.

References:
    [27] A. Srivastava et al., "Shape analysis of elastic curves in
         Euclidean spaces", *IEEE TPAMI*, 33(7), 2011, 1415-1428.
"""

from __future__ import annotations

import torch


def srv_transform(tau: torch.Tensor) -> torch.Tensor:
    """Square-root-velocity transform.

    Given ``τ[B, T]`` evaluated on a uniform time grid, returns
    ``q[B, T-1] = sgn(τ') · √|τ'|`` via finite differences.
    """
    if tau.dim() < 1:
        raise ValueError("expected at least 1-D tau")
    dt = 1.0 / max(tau.shape[-1] - 1, 1)
    diffs = (tau[..., 1:] - tau[..., :-1]) / dt
    return diffs.sign() * diffs.abs().clamp_min(1e-8).sqrt()


def inverse_srv_transform(q: torch.Tensor, *, T: int) -> torch.Tensor:
    """Recover ``τ`` from SRV ``q``.

    Computes ``τ'(t) = sgn(q) · q²`` then integrates with τ(0) = 0
    and rescales so τ(T-1) = T-1 (unit endpoint normalisation).
    """
    rates = q.sign() * q.pow(2)
    cumulative = torch.cumsum(rates, dim=-1)
    z = torch.zeros_like(cumulative[..., :1])
    full = torch.cat([z, cumulative], dim=-1)
    norm = full[..., -1:].clamp_min(1e-6)
    return full / norm * float(T - 1)


def teichmuller_1d_step(tau0: torch.Tensor, tau1: torch.Tensor, r: float) -> torch.Tensor:
    """Geodesic step in the 1D Teichmüller space, parameter ``r ∈ [0, 1]``.

    Implementation: SRV-coordinatise both endpoints, linearly
    interpolate (straight lines are geodesics under SRV), and
    invert the SRV transform.
    """
    T = tau0.shape[-1]
    q0, q1 = srv_transform(tau0), srv_transform(tau1)
    qr = (1.0 - r) * q0 + r * q1
    return inverse_srv_transform(qr, T=T)


def enforce_1d_beltrami_constraint(raw: torch.Tensor, *, eps: float = 0.5) -> torch.Tensor:
    """Squash raw scores to the 1D Beltrami ball ``|τ' - 1| < ε``.

    Used by MRF §5 to anchor per-scanner reparameterisations near
    the identity. ``τ'`` is parameterised as ``1 + ε · tanh(raw)``;
    the cumulative integral gives a strictly monotone τ.
    """
    derivative = 1.0 + eps * torch.tanh(raw)
    cumulative = torch.cumsum(derivative, dim=-1)
    z = torch.zeros_like(cumulative[..., :1])
    full = torch.cat([z, cumulative], dim=-1)
    T = full.shape[-1]
    norm = full[..., -1:].clamp_min(1e-6)
    return full / norm * float(T - 1)


def apply_time_reparam(signal: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Resample ``signal[..., T]`` along the reparameterised axis ``τ[..., T]``.

    Uses linear interpolation. ``τ`` is expected to map ``[0, T-1]``
    onto itself.
    """
    if signal.shape[-1] != tau.shape[-1]:
        raise ValueError(f"signal length {signal.shape[-1]} ≠ tau length {tau.shape[-1]}")
    T = signal.shape[-1]
    idx_floor = tau.floor().clamp(0, T - 2).long()
    frac = (tau - idx_floor.float()).unsqueeze(-1)
    # Gather along last dim of signal.
    expand = idx_floor.unsqueeze(-1).expand(*idx_floor.shape, 1)
    while expand.dim() < signal.dim():
        expand = expand.unsqueeze(0)
    left = signal.gather(-1, idx_floor)
    right = signal.gather(-1, (idx_floor + 1).clamp(0, T - 1))
    return (1.0 - frac.squeeze(-1)) * left + frac.squeeze(-1) * right


__all__ = [
    "apply_time_reparam",
    "enforce_1d_beltrami_constraint",
    "inverse_srv_transform",
    "srv_transform",
    "teichmuller_1d_step",
]
