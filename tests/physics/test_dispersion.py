r"""Physics tests for the BPP relaxation-dispersion module (DL-BAE core).

Targets ``spectramr.infrastructure.physics.dispersion``.

The Bloembergen-Purcell-Pound spectral density :math:`J(\omega;\tau_c)=\tau_c/(1+
\omega^2\tau_c^2)` generates the field dependence of relaxation:

.. math::

   \frac1{T_1(B_0)} = a_0 + \sum_k b_k\bigl[J(\gamma B_0;\tau_k)
       + 4 J(2\gamma B_0;\tau_k)\bigr].

The latent :math:`(a_0,\{b_k,\tau_k\})` parametrises the dispersion *law*, not its
value at one field, so it is field-invariant by construction and -- for a single
pool -- identifiable from measurements at :math:`M \ge 2P+1 = 3` distinct fields.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.physics


def test_bpp_spectral_density_zero_frequency_equals_tau() -> None:
    from spectramr.infrastructure.physics.dispersion import bpp_spectral_density

    tau = torch.tensor([1e-9, 5e-9, 1e-8])
    j0 = bpp_spectral_density(torch.zeros_like(tau), tau)
    assert torch.allclose(j0, tau)


def test_bpp_spectral_density_decreases_with_frequency() -> None:
    from spectramr.infrastructure.physics.dispersion import bpp_spectral_density

    tau = torch.tensor(5e-9)
    omegas = torch.tensor([1e7, 1e8, 1e9, 1e10])
    j = bpp_spectral_density(omegas, tau)
    assert torch.all(j[1:] < j[:-1]), "J(omega) must be monotone decreasing"


def test_t1_increases_with_field_for_positive_pool() -> None:
    """The dispersion law reproduces the measured trend: T1 rises with B0."""
    from spectramr.infrastructure.physics.dispersion import dispersion_t1

    b0 = torch.tensor([0.05, 0.5, 1.5, 3.0, 7.0])
    a0 = 0.3  # s^-1 baseline
    b = torch.tensor([3.0e8])  # single pool, s^-2 scaling
    tau = torch.tensor([4e-9])
    t1 = dispersion_t1(b0, a0=a0, b=b, tau_c=tau)
    assert torch.all(t1[1:] >= t1[:-1] - 1e-6), "T1 must be non-decreasing in B0"


def test_extreme_narrowing_gives_field_independent_plateau() -> None:
    """tau_c -> 0 makes J -> tau_c (constant), so 1/T1 -> a0 + const: flat in B0."""
    from spectramr.infrastructure.physics.dispersion import dispersion_r1

    b0 = torch.tensor([0.05, 3.0, 7.0])
    r1 = dispersion_r1(b0, a0=0.3, b=torch.tensor([1.0e8]), tau_c=torch.tensor([1e-13]))
    assert torch.allclose(r1, r1[0].expand_as(r1), atol=1e-4)


def test_single_pool_identifiable_from_three_fields() -> None:
    """M >= 2P+1 = 3 fields recover a single-pool (a0, b, tau_c)."""
    from spectramr.infrastructure.physics.dispersion import (
        dispersion_r1,
        fit_single_pool_dispersion,
    )

    b0 = torch.tensor([0.1, 0.5, 1.5, 3.0, 7.0])
    a0_true, b_true, tau_true = 0.35, 2.5e8, 5e-9
    r1 = dispersion_r1(
        b0, a0=a0_true, b=torch.tensor([b_true]), tau_c=torch.tensor([tau_true])
    )
    a0_hat, b_hat, tau_hat = fit_single_pool_dispersion(b0, r1)
    # tau recovered within a relative tolerance; a0/b within absolute/relative.
    assert math.isclose(tau_hat, tau_true, rel_tol=0.15)
    assert math.isclose(a0_hat, a0_true, abs_tol=0.05)
    assert math.isclose(b_hat, b_true, rel_tol=0.15)


def test_fit_rejects_too_few_fields() -> None:
    from spectramr.infrastructure.physics.dispersion import fit_single_pool_dispersion

    b0 = torch.tensor([1.5, 3.0])  # only 2 fields < 2P+1 = 3
    r1 = torch.tensor([0.8, 0.7])
    with pytest.raises(ValueError, match="3"):
        fit_single_pool_dispersion(b0, r1)


# --- power-law T1 transport (idea 2.1 bloch_synth) ---------------------------


def test_power_law_transport_identity_when_same_field() -> None:
    """b0_tgt == b0_src (ratio 1) reproduces T1_ref exactly (Prop 3 consistency)."""
    from spectramr.infrastructure.physics.dispersion import power_law_t1_transport

    t1 = torch.tensor([1000.0, 800.0, 1400.0])
    out = power_law_t1_transport(t1, 3.0, 3.0, 0.35)
    assert torch.allclose(out, t1)


def test_power_law_transport_field_invariant_when_beta_zero() -> None:
    """beta == 0 -> field-invariant T1 (the beta=0 baseline sibling)."""
    from spectramr.infrastructure.physics.dispersion import power_law_t1_transport

    t1 = torch.tensor([1000.0, 800.0])
    out = power_law_t1_transport(t1, 0.1, 7.0, 0.0)
    assert torch.allclose(out, t1)


def test_power_law_transport_increases_with_field_for_positive_beta() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t1_transport

    t1 = torch.tensor([1000.0])
    out = power_law_t1_transport(t1, 3.0, 7.0, 0.34)
    assert float(out) > float(t1)
    # T1(7T) = T1(3T) * (7/3)^0.34
    assert math.isclose(float(out), 1000.0 * (7.0 / 3.0) ** 0.34, rel_tol=1e-5)


def test_power_law_transport_log_field_ratio_sensitivity() -> None:
    """Corollary: ULF->UHF is ~5x (not 30x) more beta-sensitive than 3T->7T,
    because sensitivity is linear in ln(field ratio)."""
    ratio_ulf = math.log(7.0 / 0.1)
    ratio_hf = math.log(7.0 / 3.0)
    assert 4.5 < ratio_ulf / ratio_hf < 5.5


def test_power_law_transport_rejects_nonpositive_field() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t1_transport

    with pytest.raises(ValueError):
        power_law_t1_transport(torch.tensor([1000.0]), 0.0, 7.0, 0.34)


def test_power_law_transport_is_differentiable_in_beta() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t1_transport

    beta = torch.tensor(0.34, requires_grad=True)
    t1 = torch.tensor([1000.0])
    power_law_t1_transport(t1, 3.0, 7.0, beta).sum().backward()
    assert beta.grad is not None and torch.isfinite(beta.grad).all()


# --------------------------------------------------------------------------
# power_law_t2_transport -- the transverse companion, and why it is not
# dispersion_t2 (non-negotiable 17: a copy kept for a real difference is named
# for that difference, and the difference is executed, not asserted in prose).
# --------------------------------------------------------------------------


def test_power_law_t2_transport_identity_when_same_field() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t2_transport

    t2 = torch.tensor([80.0, 110.0])
    out = power_law_t2_transport(t2, 3.0, 3.0, 0.05)
    assert torch.allclose(out, t2)


def test_power_law_t2_transport_field_invariant_when_gamma_zero() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t2_transport

    t2 = torch.tensor([80.0])
    out = power_law_t2_transport(t2, 0.1, 7.0, 0.0)
    assert torch.allclose(out, t2)


def test_power_law_t2_transport_shortens_toward_low_field() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t2_transport

    t2 = torch.tensor([80.0])
    out = power_law_t2_transport(t2, 7.0, 0.1, 0.05)
    assert (out < t2).all()


def test_power_law_t2_transport_rejects_nonpositive_field() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t2_transport

    with pytest.raises(ValueError, match="positive field strengths"):
        power_law_t2_transport(torch.tensor([80.0]), 0.0, 7.0, 0.05)


def test_power_law_t2_transport_is_differentiable_in_gamma() -> None:
    from spectramr.infrastructure.physics.dispersion import power_law_t2_transport

    t2 = torch.tensor([80.0])
    gamma = torch.tensor(0.05, requires_grad=True)
    power_law_t2_transport(t2, 3.0, 7.0, gamma).sum().backward()
    assert gamma.grad is not None and torch.isfinite(gamma.grad)


def test_the_two_t2_laws_agree_in_sign_and_disagree_in_magnitude() -> None:
    """The power law and the BPP form are different models, not two copies.

    This is the assertion that keeps both functions in the module. If a future
    change makes them agree numerically, one of them is redundant and should be
    deleted -- so this test failing is informative in both directions.

    Compared as *ratios*, which are unit-free: ``dispersion_t2`` returns seconds
    while ``power_law_t2_transport`` preserves its input units.
    """
    from spectramr.infrastructure.physics.dispersion import (
        dispersion_t2,
        power_law_t2_transport,
    )

    b0 = torch.tensor([7.0, 0.1])
    pool = {"c0": 0.0, "b": torch.tensor([1.0e9]), "tau_c": torch.tensor([1.0e-8])}
    bpp = dispersion_t2(b0, **pool)
    bpp_factor = (bpp[0] / bpp[1]).item()

    t2 = torch.tensor([80.0])
    pow_factor = (t2 / power_law_t2_transport(t2, 7.0, 0.1, 0.05)).item()

    # Same sign: J(w;tau) is maximal at w = 0, so both shorten T2 toward low field.
    assert bpp_factor > 1.0 and pow_factor > 1.0
    # Different magnitude, by a wide margin at these pool parameters.
    assert bpp_factor > 2.0 * pow_factor, (
        f"BPP shortens by {bpp_factor:.3f}x, power law by {pow_factor:.3f}x -- "
        "if these have converged, one of the two laws is now redundant"
    )


@pytest.mark.parametrize(("n_pools", "expected"), [(1, 5), (2, 7), (3, 9)])
def test_magnitude_stack_counts_rho_a0_c0_and_two_per_pool(n_pools: int, expected: int) -> None:
    """One magnitude per field must determine 2P+3 constants, not the 2P+1 of R1 (#2417)."""
    from spectramr.infrastructure.physics.dispersion import magnitude_stack_unknowns

    assert magnitude_stack_unknowns(n_pools) == expected


def test_magnitude_stack_unknowns_rejects_zero_pools() -> None:
    from spectramr.infrastructure.physics.dispersion import magnitude_stack_unknowns

    with pytest.raises(ValueError, match="n_pools"):
        magnitude_stack_unknowns(0)
