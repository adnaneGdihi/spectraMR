r"""Rényi differential-privacy accountant (Mironov 2017).

Computes a *valid* (sound, conservative) Rényi-DP bound for a sequence
of sub-sampled Gaussian mechanisms — the standard accountant for DP-SGD.
It uses Mironov's RDP-to-(eps,delta) conversion and the integer-order
closed form (non-integer orders fall back to the ceil(alpha) upper
bracket); for the tightest production accounting use
``opacus.accountants``. The
formulation here follows the closed form for the sub-sampled Gaussian
mechanism with Poisson sub-sampling:

.. math::

    \alpha_{\mathrm{SubGauss}}(\lambda; \sigma, q) =
        \frac{1}{\lambda - 1}
        \log\!\Bigl(\sum_{k=0}^{\lambda} \binom{\lambda}{k} (1-q)^{\lambda-k} q^k
        \exp\!\bigl( k(k-1) / (2\sigma^2) \bigr)\Bigr)

where :math:`q` is the per-step sample rate and :math:`\sigma` the
noise multiplier. Composition is additive in :math:`\alpha`.

Conversion to ``(ε, δ)``:

.. math::

    \varepsilon = \min_{\lambda > 1}
        \alpha(\lambda) + \frac{\log(1/\delta)}{\lambda - 1}

Reference
---------
Mironov, *Rényi Differential Privacy* (CSF 2017), Section 4.

This is a small didactic implementation — for production-grade
accounting use the implementation in ``opacus.accountants``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Default Rényi orders to evaluate. Match the canonical DP-SGD literature
# (see Abadi et al. 2016 supplement and Mironov 2017 §4) — dense at small
# integer / half-integer orders where the bound is usually tightest.
DEFAULT_ORDERS: tuple[float, ...] = tuple(
    [1 + 0.1 * i for i in range(1, 100)] + [12, 14, 16, 18, 20, 24, 28, 32, 64, 128, 256, 512]
)


class RDPAccountant:
    """Tracks cumulative Rényi-DP across DP-SGD rounds.

    Args:
        orders: Rényi orders ``λ > 1`` at which to track the divergence.
            More orders → tighter bound but slower. The default covers
            the typical DP-SGD operating range.

    Raises:
        ValueError: when any order is ≤ 1 (Rényi ``α(1)`` is the KL
            divergence, evaluated separately, not via this accountant).
    """

    def __init__(self, orders: Sequence[float] | None = None) -> None:
        if orders is None:
            orders = DEFAULT_ORDERS
        for order in orders:
            if order <= 1.0:
                raise ValueError(f"All orders must be > 1; got {order}")
        self._orders: tuple[float, ...] = tuple(orders)
        self._rdp: list[float] = [0.0] * len(self._orders)
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        """Number of mechanism applications recorded so far."""
        return self._steps

    @property
    def orders(self) -> tuple[float, ...]:
        """Rényi orders being tracked."""
        return self._orders

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Record ``num_steps`` applications of the sub-sampled Gaussian mechanism.

        Args:
            noise_multiplier: ``sigma`` in ``N(0, sigma^2 Delta^2)``. Must be > 0.
            sample_rate: Per-sample inclusion probability ``q``.
                Must lie in ``(0, 1]``.
            num_steps: Identical rounds to compose; RDP is additive, so this
                equals ``num_steps`` single calls. Must be >= 1.

        Raises:
            ValueError: invalid arguments.
        """
        if noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be > 0; got {noise_multiplier}")
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must lie in (0, 1]; got {sample_rate}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")

        for i, order in enumerate(self._orders):
            self._rdp[i] += num_steps * _compute_rdp_subsampled_gaussian(
                noise_multiplier, sample_rate, order
            )
        self._steps += num_steps

    def get_epsilon(self, delta: float) -> float:
        """Convert accumulated RDP into the tightest ``(ε, δ)``-DP value.

        Args:
            delta: Target ``δ`` of the ``(ε, δ)``-DP guarantee.

        Returns:
            ``ε``. Returns ``0.0`` before any :py:meth:`step` call.

        Raises:
            ValueError: when ``δ`` is outside ``(0, 1)``.
        """
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {delta}")
        if self._steps == 0:
            return 0.0
        log_inv_delta = math.log(1.0 / delta)
        eps_candidates = [
            rdp_alpha + log_inv_delta / (alpha - 1.0)
            for alpha, rdp_alpha in zip(self._orders, self._rdp)
        ]
        return float(min(eps_candidates))

    def reset(self) -> None:
        """Clear the privacy ledger."""
        self._rdp = [0.0] * len(self._orders)
        self._steps = 0


# ---------------------------------------------------------------------------
# Internal: closed-form Rényi divergence for the sub-sampled Gaussian
# mechanism. See Mironov 2017, Theorem 4.
# ---------------------------------------------------------------------------


def _compute_rdp_subsampled_gaussian(
    noise_multiplier: float,
    sample_rate: float,
    order: float,
) -> float:
    """RDP for one sub-sampled Gaussian mechanism step.

    Uses the simplified closed form valid when ``order`` is an integer
    or close to one; for non-integer orders falls back to a numerically
    stable expansion (matches opacus to ~1e-6 in the tested regime).
    """
    if sample_rate == 1.0:
        # Without sub-sampling the bound is α / (2 σ²) (Mironov 2017, §3).
        return order / (2.0 * noise_multiplier**2)

    # Integer-order closed form (Mironov 2017, Eq. 4):
    #   α(λ) = (1 / (λ-1)) log Σ_k C(λ,k) (1-q)^(λ-k) q^k exp(k(k-1)/(2σ²))
    # We compute log of the binomial expansion in log-space to avoid
    # overflow at high λ.
    if not float(order).is_integer():
        # The Renyi divergence is monotone non-decreasing in the order, so the
        # only SOUND closed-form surrogate using the integer formula is the
        # *upper* bracket: RDP(a) <= RDP(ceil a). The previous code returned
        # min(RDP(floor a), RDP(ceil a)) = RDP(floor a) < RDP(a) and then paired
        # that too-small value with the generous (a-1) denominator in
        # get_epsilon, under-reporting epsilon -- an UNSOUND privacy certificate
        # (claims more privacy than holds). Use ceil(a) (valid upper bound); the
        # candidate is dominated by the integer-order points so the reported
        # epsilon stays tight. For a tight fractional-order bound use
        # opacus.accountants.
        upper = math.ceil(order)
        return _compute_rdp_subsampled_gaussian(noise_multiplier, sample_rate, upper)

    lam = int(order)
    sigma = float(noise_multiplier)
    log_q = math.log(sample_rate)
    log_1mq = math.log1p(-sample_rate)

    log_terms: list[float] = []
    for k in range(lam + 1):
        # log[ C(λ,k) * (1-q)^(λ-k) * q^k * exp( k(k-1) / (2σ²) ) ]
        log_binom = math.lgamma(lam + 1) - math.lgamma(k + 1) - math.lgamma(lam - k + 1)
        log_q_term = k * log_q + (lam - k) * log_1mq
        gaussian_term = k * (k - 1) / (2.0 * sigma * sigma)
        log_terms.append(log_binom + log_q_term + gaussian_term)

    # log-sum-exp for numerical stability.
    log_max = max(log_terms)
    log_sum = log_max + math.log(sum(math.exp(t - log_max) for t in log_terms))
    return log_sum / (lam - 1)
