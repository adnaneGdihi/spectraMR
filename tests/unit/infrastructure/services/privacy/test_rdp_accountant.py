"""Tests for ``RDPAccountant``.

Targets ``spectramr.infrastructure.services.privacy.rdp_accountant`` —
PR-7 (H5) of ``TODO/backlog_paradigm_expansion_roadmap.md``.

Plan acceptance criterion covered:

- *"(ε, δ) accountant matches known closed-form value for a single round"*
  — for the un-sub-sampled Gaussian, ε at order λ equals λ/(2σ²) +
  log(1/δ)/(λ-1). The accountant's minimum over orders should
  approach the closed-form optimum.

Standard contract tests:

- ``step`` and ``get_epsilon`` argument validation
- Epsilon is monotone non-decreasing in steps
- Epsilon is monotone increasing in noise_multiplier⁻¹ (more noise → smaller ε)
- Empty-budget accountant returns ε = 0
- Reset clears history
- Protocol satisfaction
"""

from __future__ import annotations

import math

import pytest

from spectramr.domain.interfaces import IPrivacyAccountant
from spectramr.infrastructure.services.privacy.rdp_accountant import (
    RDPAccountant,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_orders_all_above_one() -> None:
    """All Rényi orders are strictly > 1."""
    acc = RDPAccountant()
    assert all(o > 1.0 for o in acc.orders)


def test_order_at_one_rejected() -> None:
    """``α = 1`` is the KL divergence limit — outside this accountant's domain."""
    with pytest.raises(ValueError, match="orders must be > 1"):
        RDPAccountant(orders=[1.0, 2.0])


def test_fresh_accountant_zero_steps() -> None:
    """No mechanism applications recorded yet."""
    acc = RDPAccountant()
    assert acc.steps == 0


# ---------------------------------------------------------------------------
# step() validation
# ---------------------------------------------------------------------------


def test_zero_noise_multiplier_rejected() -> None:
    """``σ = 0`` would be the no-noise mechanism — undefined here."""
    acc = RDPAccountant()
    with pytest.raises(ValueError, match="noise_multiplier"):
        acc.step(noise_multiplier=0.0, sample_rate=0.5)


def test_negative_noise_multiplier_rejected() -> None:
    """Negative σ is meaningless."""
    acc = RDPAccountant()
    with pytest.raises(ValueError, match="noise_multiplier"):
        acc.step(noise_multiplier=-1.0, sample_rate=0.5)


def test_zero_sample_rate_rejected() -> None:
    """``q = 0`` would be a no-op step."""
    acc = RDPAccountant()
    with pytest.raises(ValueError, match="sample_rate"):
        acc.step(noise_multiplier=1.0, sample_rate=0.0)


def test_sample_rate_above_one_rejected() -> None:
    """``q > 1`` is invalid (probability)."""
    acc = RDPAccountant()
    with pytest.raises(ValueError, match="sample_rate"):
        acc.step(noise_multiplier=1.0, sample_rate=1.5)


# ---------------------------------------------------------------------------
# Epsilon contract
# ---------------------------------------------------------------------------


def test_get_epsilon_zero_before_step() -> None:
    """No mechanism applications → ε = 0."""
    acc = RDPAccountant()
    assert acc.get_epsilon(delta=1e-5) == 0.0


def test_get_epsilon_invalid_delta_rejected() -> None:
    """``δ`` must lie in ``(0, 1)``."""
    acc = RDPAccountant()
    acc.step(noise_multiplier=1.0, sample_rate=0.1)
    with pytest.raises(ValueError, match="delta"):
        acc.get_epsilon(delta=0.0)
    with pytest.raises(ValueError, match="delta"):
        acc.get_epsilon(delta=1.5)


def test_epsilon_monotone_in_steps() -> None:
    """Each additional step weakens the privacy guarantee."""
    acc = RDPAccountant()
    eps_history = []
    for _ in range(5):
        acc.step(noise_multiplier=1.0, sample_rate=0.1)
        eps_history.append(acc.get_epsilon(delta=1e-5))
    # Epsilon never decreases as we accumulate.
    for prev, nxt in zip(eps_history, eps_history[1:]):
        assert nxt >= prev


def test_epsilon_decreases_with_more_noise() -> None:
    """Higher noise multiplier → smaller ε for the same step count."""
    acc_low = RDPAccountant()
    acc_high = RDPAccountant()
    for _ in range(10):
        acc_low.step(noise_multiplier=1.0, sample_rate=0.1)
        acc_high.step(noise_multiplier=4.0, sample_rate=0.1)
    eps_low = acc_low.get_epsilon(delta=1e-5)
    eps_high = acc_high.get_epsilon(delta=1e-5)
    assert eps_high < eps_low


# ---------------------------------------------------------------------------
# Closed-form sanity: full-sampling Gaussian mechanism
# ---------------------------------------------------------------------------


def test_full_sampling_matches_closed_form() -> None:
    r"""For ``q = 1``, RDP at order λ is λ / (2σ²); ε ≤ that + log(1/δ)/(λ-1).

    Mironov 2017, §3 (un-sub-sampled Gaussian mechanism).
    """
    sigma = 4.0
    delta = 1e-5
    n_steps = 10

    acc = RDPAccountant(orders=[2.0, 4.0, 8.0, 16.0, 32.0])
    for _ in range(n_steps):
        acc.step(noise_multiplier=sigma, sample_rate=1.0)

    eps = acc.get_epsilon(delta=delta)
    # Closed form: at each order λ, total RDP across n steps is
    # n · λ / (2σ²); ε ≤ n·λ/(2σ²) + log(1/δ)/(λ-1).
    log_inv_delta = math.log(1.0 / delta)
    closed_form_min = min(
        n_steps * order / (2.0 * sigma ** 2) + log_inv_delta / (order - 1.0)
        for order in [2.0, 4.0, 8.0, 16.0, 32.0]
    )
    # The accountant returns the minimum over its own orders. With the
    # same orders, the values must agree.
    assert math.isclose(eps, closed_form_min, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_history() -> None:
    """``reset`` zeroes the ledger and step counter."""
    acc = RDPAccountant()
    for _ in range(3):
        acc.step(noise_multiplier=1.0, sample_rate=0.1)
    assert acc.steps == 3
    acc.reset()
    assert acc.steps == 0
    assert acc.get_epsilon(delta=1e-5) == 0.0


def test_num_steps_composes_like_repeated_calls() -> None:
    """``step(num_steps=n)`` equals ``n`` single calls; ``num_steps < 1`` raises."""
    one, many = RDPAccountant(), RDPAccountant()
    for _ in range(7):
        one.step(noise_multiplier=1.1, sample_rate=0.02)
    many.step(noise_multiplier=1.1, sample_rate=0.02, num_steps=7)
    assert many.steps == 7
    assert math.isclose(many.get_epsilon(1e-5), one.get_epsilon(1e-5), rel_tol=1e-12)
    with pytest.raises(ValueError, match="num_steps"):
        many.step(noise_multiplier=1.1, sample_rate=0.02, num_steps=0)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_satisfies_protocol() -> None:
    """``RDPAccountant`` is a runtime instance of ``IPrivacyAccountant``."""
    acc = RDPAccountant()
    assert isinstance(acc, IPrivacyAccountant)


# ---------------------------------------------------------------------------
# Soundness of the non-integer-order bound (regression — 2026-06-12 audit)
# ---------------------------------------------------------------------------


def test_noninteger_order_does_not_undercut_integer_bound() -> None:
    """The dense non-integer order grid must not report an epsilon BELOW the
    sound integer-order bound.

    Regression for the unsound non-integer fallback: it returned
    ``min(RDP(floor a), RDP(ceil a)) = RDP(floor a) < RDP(a)`` and paired that
    too-small RDP with the generous ``(a-1)`` denominator, producing an
    epsilon candidate below both valid bracketing bounds (an under-reported,
    unsound privacy certificate).
    """
    from spectramr.infrastructure.services.privacy.rdp_accountant import (
        DEFAULT_ORDERS,
    )

    integer_orders = [o for o in DEFAULT_ORDERS if float(o).is_integer()]
    acc_full = RDPAccountant()
    acc_int = RDPAccountant(orders=integer_orders)
    for _ in range(1000):
        acc_full.step(noise_multiplier=0.8, sample_rate=0.02)
        acc_int.step(noise_multiplier=0.8, sample_rate=0.02)
    eps_full = acc_full.get_epsilon(delta=1e-5)
    eps_int = acc_int.get_epsilon(delta=1e-5)
    assert eps_full >= eps_int - 1e-9, (
        f"dense-grid eps {eps_full} undercut the sound integer bound {eps_int}"
    )
