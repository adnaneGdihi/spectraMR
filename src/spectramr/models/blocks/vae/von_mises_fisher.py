r"""von Mises-Fisher distribution: reparameterised sampler + closed-form KL.

Implements the vMF latent of Davidson *et al.* [1]. The vMF density on
the unit sphere :math:`S^{d-1}` is

.. math::

    p(z; \mu, \kappa) = C_d(\kappa)\,\exp(\kappa\,\mu^\top z),
    \qquad
    C_d(\kappa) = \frac{\kappa^{d/2 - 1}}{(2\pi)^{d/2} I_{d/2 - 1}(\kappa)}.

The KL with the uniform prior on the sphere (vMF with :math:`\kappa = 0`)
has the closed form

.. math::

    \mathrm{KL}(q \,\|\, U)
        = \kappa\,\frac{I_{d/2}(\kappa)}{I_{d/2 - 1}(\kappa)}
          + \log C_d(\kappa) - \log C_d(0).

Sampling uses the rejection sampler of Ulrich [2]: draw the tangential
direction uniformly on :math:`S^{d-2}` and the polar coordinate
:math:`w` from the marginal via the Wood acceptance-rejection scheme,
then rotate the canonical mean :math:`e_1` onto :math:`\mu` with a
Householder reflection. Gradients flow through :math:`\kappa` via the
sampled :math:`w` and through :math:`\mu` via the Householder rotation.

References:
    [1] T. R. Davidson *et al.*, "Hyperspherical variational
        auto-encoders," UAI 2018.
    [2] G. Ulrich, "Computer generation of distributions on the
        m-sphere," J. R. Stat. Soc. C, 1984.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _log_iv(nu: float, kappa: Tensor) -> Tensor:
    r"""Stable log of the modified Bessel function :math:`\log I_\nu(\kappa)`.

    Uses the uniform asymptotic expansion (DLMF 10.41) blended with the
    small-argument series, evaluated entirely in log-space so it stays
    finite across the full positive ``kappa`` range used in training.
    """
    kappa = kappa.clamp_min(1e-8)
    # Uniform large-argument expansion:
    #   I_nu(k) ~ exp(sqrt(k^2 + nu^2)) / sqrt(2*pi) * (k^2+nu^2)^(-1/4) * series
    # We keep the leading two terms which are accurate for the kappa>=1
    # regime that softplus(.) + 1 guarantees in the model.
    t = torch.sqrt(kappa * kappa + nu * nu)
    eta = t + nu * torch.log(kappa / (nu + t))
    log_pref = -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(t)
    return eta + log_pref


def _bessel_ratio(d: int, kappa: Tensor) -> Tensor:
    r"""Ratio :math:`I_{d/2}(\kappa) / I_{d/2 - 1}(\kappa)`.

    Computed in log-space from :func:`_log_iv` and exponentiated; the
    ratio lies in ``(0, 1)`` for all positive ``kappa`` so the
    exponentiation is numerically safe.
    """
    log_num = _log_iv(d / 2.0, kappa)
    log_den = _log_iv(d / 2.0 - 1.0, kappa)
    return torch.exp(log_num - log_den)


def _log_c_d(d: int, kappa: Tensor) -> Tensor:
    r"""Log normaliser :math:`\log C_d(\kappa)`."""
    kappa = kappa.clamp_min(1e-8)
    return (
        (d / 2.0 - 1.0) * torch.log(kappa)
        - (d / 2.0) * math.log(2 * math.pi)
        - _log_iv(d / 2.0 - 1.0, kappa)
    )


class VonMisesFisher:
    r"""A batched von Mises-Fisher distribution on :math:`S^{d-1}`.

    Args:
        mu: Mean directions ``(B, d)``; must be unit norm.
        kappa: Concentrations ``(B, 1)`` or ``(B,)``; must be positive.
        validate: When True, assert ``mu`` is unit-norm at construction.
    """

    def __init__(self, mu: Tensor, kappa: Tensor, validate: bool = True) -> None:
        if kappa.dim() == mu.dim():
            # already (B, 1)
            kappa = kappa.squeeze(-1)
        self.mu = mu
        self.kappa = kappa
        self.d = mu.shape[-1]
        if validate:
            norms = mu.norm(dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
                raise ValueError(
                    "VonMisesFisher.mu must be unit-norm; got norms in "
                    f"[{float(norms.min()):.4f}, {float(norms.max()):.4f}]"
                )

    # ------------------------------------------------------------------
    # Sampling (Ulrich / Wood rejection sampler)
    # ------------------------------------------------------------------
    def rsample(self) -> Tensor:
        """Reparameterised sample, shape ``(B, d)`` on the unit sphere."""
        kappa = self.kappa.unsqueeze(-1)  # (B, 1)
        w = self._sample_w(self.kappa)  # (B,)
        w = w.unsqueeze(-1)  # (B, 1)

        # Tangential direction v ~ Uniform(S^{d-2}), orthogonal to e_1.
        v = torch.randn(self.mu.shape[0], self.d - 1, device=self.mu.device, dtype=self.mu.dtype)
        v = F.normalize(v, dim=-1)

        # Sample on the canonical sphere about e_1 = (1, 0, ..., 0).
        scale = torch.sqrt((1.0 - w * w).clamp_min(0.0))
        z_canonical = torch.cat([w, scale * v], dim=-1)  # (B, d)

        # Householder reflection mapping e_1 -> mu.
        return self._householder_rotate(z_canonical, self.mu)

    def _sample_w(self, kappa: Tensor, rounds: int = 50) -> Tensor:
        r"""Sample the polar coordinate ``w = z·e_1`` via Wood's scheme.

        Wood (1994): with :math:`x_0 = (1-b)/(1+b)` and
        :math:`c = \kappa x_0 + (d-1)\log(1-x_0^2)`, a Beta proposal ``W`` is
        accepted when :math:`\kappa W + (d-1)\log(1 - x_0 W) - c \ge \log U`.
        ``b``, :math:`1-x_0^2` and :math:`1-x_0W` are evaluated in rationalised
        form so float32 neither cancels ``b`` to 0 at large ``kappa`` nor hits
        ``log 0`` as ``kappa -> 0``. All ``rounds`` proposals are drawn in one
        batch and the first accepted one is kept, which is exact rejection
        sampling without a per-round host sync; an entry with no acceptance
        keeps the last proposal (probability < 1e-9 for 50 rounds).
        """
        d = self.d
        b = (d - 1) / (2.0 * kappa + torch.sqrt(4.0 * kappa * kappa + (d - 1) ** 2))
        log_one_minus_x0_sq = torch.log(4.0 * b) - 2.0 * torch.log1p(b)
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (d - 1) * log_one_minus_x0_sq

        conc = torch.full_like(kappa, (d - 1) / 2.0)
        eps = torch.distributions.Beta(conc, conc).sample((rounds,))  # (R, *kappa.shape)
        denom = 1.0 - (1.0 - b) * eps
        w_prop = (1.0 - (1.0 + b) * eps) / denom
        log_one_minus_x0w = torch.log(2.0 * b) - torch.log1p(b) - torch.log(denom)
        t = kappa * w_prop + (d - 1) * log_one_minus_x0w - c
        accept = t >= torch.log(torch.rand_like(t))
        accept[-1] = True  # fallback: the last proposal when nothing was accepted
        first = accept.to(torch.uint8).argmax(dim=0, keepdim=True)
        return w_prop.gather(0, first).squeeze(0)

    @staticmethod
    def _householder_rotate(x: Tensor, mu: Tensor) -> Tensor:
        """Reflect ``x`` so the canonical pole ``e_1`` maps onto ``mu``."""
        e1 = torch.zeros_like(mu)
        e1[..., 0] = 1.0
        u = F.normalize(e1 - mu, dim=-1)
        # Reflection H = I - 2 u u^T ; H e_1 = mu (up to sign convention).
        return x - 2.0 * (x * u).sum(dim=-1, keepdim=True) * u

    # ------------------------------------------------------------------
    # KL divergence with the uniform prior
    # ------------------------------------------------------------------
    def kl_uniform(self) -> Tensor:
        r"""Closed-form KL with the uniform prior on :math:`S^{d-1}`.

        Returns the per-sample KL ``(B,)``. Non-negative for all
        positive ``kappa`` (the uniform measure maximises differential
        entropy on the sphere).
        """
        d = self.d
        ratio = _bessel_ratio(d, self.kappa)
        log_c_kappa = _log_c_d(d, self.kappa)
        # log C_d(0) = -log surface area of S^{d-1}
        #            = log Gamma(d/2) - log 2 - (d/2) log pi
        log_c_zero = math.lgamma(d / 2.0) - math.log(2.0) - (d / 2.0) * math.log(math.pi)
        return self.kappa * ratio + log_c_kappa - log_c_zero


__all__ = ["VonMisesFisher"]
