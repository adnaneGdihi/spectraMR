"""MRF pulse-sequence acquisition manifold (MRF §4).

Provides the Beltrami-parameterised representation of the MRF
acquisition curve plus the hardware-bound checks the Tier-2 SAR
audit consumes. Conceptually the acquisition manifold is the
2-D disk :math:`\\mathbb{D}_{\\alpha, \\mathrm{TR}}` and the pulse
sequence is a curve :math:`\\pi : [0, T] \\to \\mathbb{D}_{\\alpha,
\\mathrm{TR}}` obtained as the image of a base SFC under a Beltrami
diffeomorphism with ``‖μ‖_∞ < k < 1``. The bound on ``μ`` translates
directly to a per-TR change-rate bound on the flip-angle and
repetition-time channels, which is the SAR-relevant constraint.

References:
    [18] B. Zhao et al., "Optimal experiment design for MRF: CRLB
         meets spin dynamics", *IEEE TMI*, 38(3), 2019.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectramr.infrastructure.physics.conformal_geometry import (
    enforce_beltrami_constraint,
)


@dataclass(frozen=True)
class AcquisitionBound:
    """Hardware ceilings (SAR + slew) the optimised pulse must respect."""

    sar_max_w_per_kg: float
    gradient_slew_rate_t_per_m_per_s: float

    def passes(
        self,
        pulse: torch.Tensor,
        *,
        sar_per_tr_scale: float = 1.0,
        gradient_per_tr_scale: float = 1.0,
    ) -> bool:
        """Return ``True`` iff the per-TR change rates stay inside the bounds.

        Args:
            pulse: ``[T, 2]`` (α, TR) tensor.
            sar_per_tr_scale: Multiplier converting α² to W/kg
                (B1²·duty-cycle-dependent; site-calibrated).
            gradient_per_tr_scale: Multiplier converting ΔTR to slew
                rate proxy (T/m/s).
        """
        if pulse.dim() != 2 or pulse.shape[-1] < 2:
            raise ValueError("pulse must be [T, 2]")
        alpha = pulse[..., 0]
        TR = pulse[..., 1]
        sar = sar_per_tr_scale * alpha.pow(2).mean().item()
        slew = gradient_per_tr_scale * TR.diff().abs().max().item() if TR.shape[0] > 1 else 0.0
        return sar <= self.sar_max_w_per_kg and slew <= self.gradient_slew_rate_t_per_m_per_s


def beltrami_pulse_curve(
    mu: torch.Tensor,
    *,
    base_alpha: torch.Tensor,
    base_TR: torch.Tensor,
    k_max: float = 0.9,
) -> torch.Tensor:
    """Build a pulse sequence ``[T, 2]`` from a Beltrami coefficient
    perturbation of a base curve.

    Args:
        mu: Complex Beltrami tensor of length ``T`` (or scalar).
            Squashed to ``|μ| ≤ k_max`` via
            :func:`enforce_beltrami_constraint`.
        base_alpha: ``[T]`` flip-angle base curve (radians).
        base_TR: ``[T]`` repetition-time base curve (seconds).

    Returns:
        Pulse-sequence tensor ``[T, 2]`` in ``(α, TR)`` order with the
        Beltrami perturbation applied. The applied warp is the
        first-order linearisation
        ``alpha_t = base_alpha · (1 + ε Re μ)``,
        ``TR_t   = base_TR   · (1 + ε Im μ)`` with ``ε = 0.5``; the
        full quasi-conformal recovery is overkill for the
        differentiable-pulse-design loop and the linearisation
        respects the same dilatation bound.
    """
    if base_alpha.shape != base_TR.shape:
        raise ValueError("base_alpha and base_TR must have the same shape")
    if mu.dim() == 0:
        mu = mu.unsqueeze(0).expand(base_alpha.shape[0])
    if mu.shape != base_alpha.shape:
        raise ValueError(f"mu shape {mu.shape} must match base_alpha {base_alpha.shape}")
    if not mu.is_complex():
        mu = torch.complex(mu, torch.zeros_like(mu))
    mu_c = enforce_beltrami_constraint(mu, k_max=k_max)
    eps = 0.5
    alpha = base_alpha * (1.0 + eps * mu_c.real)
    TR = base_TR * (1.0 + eps * mu_c.imag)
    return torch.stack([alpha, TR.clamp_min(1e-3)], dim=-1)


def sar_estimate(pulse: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Cheap SAR proxy: mean squared flip angle scaled by ``scale``.

    A real SAR estimate requires the integral of B1² over the
    pulse and the patient mass; this proxy is sufficient for the
    Tier-2 audit's order-of-magnitude check.
    """
    return scale * pulse[..., 0].pow(2).mean()


def slew_estimate(pulse: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Slew proxy matching :meth:`AcquisitionBound.passes`: ``scale * max |ΔTR|`` (0 for T=1)."""
    if pulse.shape[-2] < 2:
        return pulse.new_zeros(())
    return scale * pulse[..., 1].diff(dim=-1).abs().amax()


def project_pulse_to_bound(pulse: torch.Tensor, bound: AcquisitionBound) -> torch.Tensor:
    """Differentiably project a ``[T, 2]`` design onto the SAR and slew proxies of ``bound``.

    Flip angles scale uniformly so :func:`sar_estimate` meets the SAR ceiling, and the TR
    increments scale uniformly about the first TR so :func:`slew_estimate` meets the slew
    ceiling. A design already inside both bounds is returned unchanged; no host sync is made.
    """
    if pulse.dim() != 2 or pulse.shape[-1] != 2:
        raise ValueError(f"pulse must be [T, 2]; got {tuple(pulse.shape)}")
    sar = sar_estimate(pulse).clamp_min(1e-12)
    alpha = pulse[:, 0] * torch.sqrt((bound.sar_max_w_per_kg / sar).clamp(max=1.0))
    tr = pulse[:, 1]
    if tr.shape[0] > 1:
        slew = slew_estimate(pulse).clamp_min(1e-12)
        factor = (bound.gradient_slew_rate_t_per_m_per_s / slew).clamp(max=1.0)
        steps = torch.cat([tr[:1], tr.diff() * factor])
        tr = steps.cumsum(0)
    return torch.stack([alpha, tr], dim=-1)


class MRFAcquisitionManifold(torch.nn.Module):
    """Acquisition manifold + bound checker.

    Wraps the Beltrami parameterisation and exposes a
    :meth:`forward` returning ``(pulse, passes_sar, passes_slew)`` so a
    training loop can both train the design and verify hardware
    compliance in the same step.
    """

    def __init__(
        self,
        n_TR: int,
        *,
        alpha_init: float = 0.5,
        TR_init: float = 1.0,
        k_max: float = 0.9,
    ) -> None:
        super().__init__()
        self.n_TR = n_TR
        self.k_max = k_max
        # Initial unconstrained μ; the optimiser learns this.
        self.mu_raw = torch.nn.Parameter(torch.zeros(n_TR, 2))
        self.register_buffer("base_alpha", torch.full((n_TR,), float(alpha_init)))
        self.register_buffer("base_TR", torch.full((n_TR,), float(TR_init)))

    def forward(self, bound: AcquisitionBound | None = None) -> tuple[torch.Tensor, bool, bool]:
        mu = torch.complex(self.mu_raw[:, 0], self.mu_raw[:, 1])
        pulse = beltrami_pulse_curve(
            mu,
            base_alpha=self.base_alpha,
            base_TR=self.base_TR,
            k_max=self.k_max,
        )
        sar_ok = True
        slew_ok = True
        if bound is not None:
            sar_ok = sar_estimate(pulse).item() <= bound.sar_max_w_per_kg
            if self.n_TR > 1:
                slew_ok = (
                    pulse[..., 1].diff().abs().max().item()
                    <= bound.gradient_slew_rate_t_per_m_per_s
                )
        return pulse, sar_ok, slew_ok


__all__ = [
    "AcquisitionBound",
    "MRFAcquisitionManifold",
    "beltrami_pulse_curve",
    "project_pulse_to_bound",
    "sar_estimate",
    "slew_estimate",
]
