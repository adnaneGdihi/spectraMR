"""Tests for the derivative-free degradation-chain fitter."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.chain_fitter import (
    DegenerateFitError,
    fit_chain,
    warm_start_theta,
)
from spectramr.infrastructure.physics.degradation_chain import ChainLink, DegradationChain
from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.quality_descriptors import measure_attributes

ATTRS = ["tenengrad_variance"]


def _volume() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(2, 32, 32)


def _degrade(chain: DegradationChain, x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply a chain to an [S, H, W] stack and return the [S, H, W] magnitude."""
    return chain.apply(x.unsqueeze(1), seed=seed).squeeze(1).abs()


# ── warm start: the closed-form affine inverse ────────────────────────


def test_warm_start_inverts_the_declared_affine_severity():
    # PhysicalParam.value_at is affine, so the inverse is exact.
    spec = DEGRADATION_REGISTRY["complex_gaussian"].severity
    midpoint = spec.primary.value_at(0.5)
    assert warm_start_theta("complex_gaussian", midpoint) == pytest.approx(0.5, abs=1e-6)


def test_warm_start_handles_a_decreasing_axis():
    # complex_gaussian's primary is SNR: 40 dB at theta=0 down to 2 dB at theta=1.
    # A naive inverse that assumed an increasing parameter would return a negative.
    assert warm_start_theta("complex_gaussian", 40.0) == pytest.approx(0.0, abs=1e-6)
    assert warm_start_theta("complex_gaussian", 2.0) == pytest.approx(1.0, abs=1e-6)


def test_warm_start_clamps_a_target_outside_the_declared_range():
    # An SNR far better than the clean endpoint cannot be reached by degrading.
    assert warm_start_theta("complex_gaussian", 200.0) == 0.0
    assert warm_start_theta("complex_gaussian", -100.0) == 1.0


def test_warm_start_can_target_a_co_varying_parameter():
    spec = DEGRADATION_REGISTRY["resolution_snr"].severity
    co = spec.co_varying[0]
    theta = warm_start_theta("resolution_snr", co.value_at(0.5), param=co.name)
    assert theta == pytest.approx(0.5, abs=1e-6)


def test_warm_start_rejects_an_undeclared_parameter():
    with pytest.raises(KeyError):
        warm_start_theta("complex_gaussian", 1.0, param="not_a_param")


# ── fitting ───────────────────────────────────────────────────────────


def test_fit_recovers_theta_for_an_identifiable_single_axis_problem():
    """One axis, one monotone attribute: theta IS identifiable, so assert on theta.

    This is the test that separates a fitter that works from one that merely runs.
    With more axes than attributes the problem is underdetermined and only the
    achieved attribute can be asserted (see the two-axis test below).
    """
    x = _volume()
    truth = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.6),))
    target = measure_attributes(_degrade(truth, x, seed=5), attributes=ATTRS)

    result = fit_chain(
        x,
        axes=["t2star_blur"],
        target=target,
        attributes=ATTRS,
        seed=5,
        max_evals=120,
    )
    assert result.chain.thetas[0] == pytest.approx(0.6, abs=0.08)


def test_fit_matches_the_target_attribute_for_an_underdetermined_chain():
    """Two axes, one attribute: assert the ACHIEVED value, never theta recovery.

    Tolerance is derived from the objective's own seed-to-seed variation rather than
    guessed, so the assertion tracks the real noise floor.
    """
    x = _volume()
    truth = DegradationChain(
        links=(
            ChainLink(axis="complex_gaussian", theta=0.55),
            ChainLink(axis="t2star_blur", theta=0.30),
        )
    )
    target = measure_attributes(_degrade(truth, x, seed=5), attributes=ATTRS)

    spread = [
        measure_attributes(_degrade(truth, x, seed=s), attributes=ATTRS)[ATTRS[0]]
        for s in (5, 6, 7, 8)
    ]
    floor = max(spread) - min(spread)

    result = fit_chain(
        x,
        axes=["complex_gaussian", "t2star_blur"],
        target=target,
        attributes=ATTRS,
        seed=5,
        max_evals=200,
    )
    achieved = result.achieved[ATTRS[0]]
    assert abs(achieved - target[ATTRS[0]]) <= max(floor, 1e-9) * 3.0


def test_fit_raises_when_the_gap_cannot_be_closed():
    # Sharpness far ABOVE the clean image: no degradation can increase sharpness, so
    # the fit must say so rather than emit an uncalibrated chain.
    x = _volume()
    clean = measure_attributes(x, attributes=ATTRS)[ATTRS[0]]
    impossible = {ATTRS[0]: clean * 100.0}

    with pytest.raises(DegenerateFitError, match="gap"):
        fit_chain(
            x,
            axes=["complex_gaussian", "t2star_blur"],
            target=impossible,
            attributes=ATTRS,
            seed=0,
            max_evals=60,
            min_gap_closed=0.5,
        )


def test_fit_result_records_full_provenance():
    x = _volume()
    truth = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.5),))
    target = measure_attributes(_degrade(truth, x, seed=5), attributes=ATTRS)

    result = fit_chain(
        x,
        axes=["t2star_blur"],
        target=target,
        attributes=ATTRS,
        seed=3,
        max_evals=80,
    )
    assert result.seed == 3
    assert result.method == "differential_evolution"
    assert result.n_evals > 0
    assert 0.0 <= result.gap_closed <= 1.0
    assert result.chain.axes == ("t2star_blur",)
    assert set(result.weights) == set(ATTRS)
    assert result.residual <= result.initial_residual


def test_fit_is_reproducible_for_a_fixed_seed():
    x = _volume()
    truth = DegradationChain(links=(ChainLink(axis="t2star_blur", theta=0.5),))
    target = measure_attributes(_degrade(truth, x, seed=5), attributes=ATTRS)

    kw = {
        "axes": ["t2star_blur"],
        "target": target,
        "attributes": ATTRS,
        "seed": 1,
        "max_evals": 80,
    }
    a = fit_chain(x, **kw)
    b = fit_chain(x, **kw)
    assert a.chain.thetas == pytest.approx(b.chain.thetas)


def test_fit_rejects_an_unknown_method():
    x = _volume()
    with pytest.raises(ValueError, match="Unknown fit method"):
        fit_chain(
            x,
            axes=["t2star_blur"],
            target={ATTRS[0]: 0.01},
            attributes=ATTRS,
            seed=0,
            max_evals=20,
            method="gradient_descent",
        )


def test_fit_honours_an_explicit_theta0():
    # theta0 is the warm start. Supplying one out of range must raise rather than
    # being silently clipped into something the caller did not ask for.
    x = _volume()
    with pytest.raises(ValueError, match="theta0"):
        fit_chain(
            x,
            axes=["t2star_blur"],
            target={ATTRS[0]: 0.01},
            attributes=ATTRS,
            seed=0,
            max_evals=20,
            theta0=[5.0],
        )


def test_fit_rejects_a_target_missing_an_attribute():
    x = _volume()
    with pytest.raises(ValueError, match="target"):
        fit_chain(
            x,
            axes=["t2star_blur"],
            target={},
            attributes=ATTRS,
            seed=0,
            max_evals=20,
        )


# ── the acquisition-derived warm start ────────────────────────────────


def test_acquisition_warm_start_sets_only_the_noise_axis():
    from spectramr.infrastructure.physics.chain_fitter import acquisition_warm_start

    # complex_gaussian declares snr [dB] 40 -> 2. A -19 dB prediction targets 21 dB,
    # which is exactly the midpoint => theta 0.5. t2star_blur is not a noise axis, so
    # it keeps the default.
    got = acquisition_warm_start(["complex_gaussian", "t2star_blur"], -19.0, default=0.3)
    assert got[0] == pytest.approx(0.5, abs=1e-6)
    assert got[1] == pytest.approx(0.3)


def test_acquisition_warm_start_discovers_the_noise_axis_from_the_registry():
    """No hardcoded axis name: the prior must follow the DECLARED severity.

    A hardcoded 'complex_gaussian' would silently stop applying the moment a chain
    used a different noise operator, with nothing reporting it.
    """
    from spectramr.infrastructure.physics.chain_fitter import acquisition_warm_start

    noise_axes = [
        n
        for n, spec in DEGRADATION_REGISTRY.items()
        if spec.severity.primary.name == "snr" and spec.severity.primary.units == "dB"
    ]
    assert noise_axes, "no axis declares an SNR-in-dB severity; the prior has no target"
    for axis in noise_axes:
        # A large negative prediction must push the axis toward its noisy end.
        assert acquisition_warm_start([axis], -100.0)[0] == pytest.approx(1.0)
        # A positive prediction cannot be realised by degrading; clamp at clean.
        assert acquisition_warm_start([axis], +100.0)[0] == pytest.approx(0.0)


def test_acquisition_warm_start_matches_the_declared_affine_inverse():
    from spectramr.infrastructure.physics.chain_fitter import acquisition_warm_start

    spec = DEGRADATION_REGISTRY["complex_gaussian"].severity.primary
    delta = spec.value_at(0.25) - spec.at_theta_min
    assert acquisition_warm_start(["complex_gaussian"], delta)[0] == pytest.approx(0.25, abs=1e-6)


def test_fit_raises_when_the_residual_exceeds_the_absolute_ceiling():
    # Planted: a fit that closes its gap relatively but misses the target absolutely.
    # min_gap_closed=0 lets only the ceiling decide.
    from spectramr.infrastructure.physics.chain_fitter import DegenerateFitError

    x = _volume()
    truth = DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.5),))
    target = measure_attributes(_degrade(truth, x, seed=1), attributes=["tenengrad_variance"])
    with pytest.raises(DegenerateFitError, match="absolute ceiling"):
        fit_chain(
            x,
            axes=["complex_gaussian"],
            target={k: 10.0 * v for k, v in target.items()},
            attributes=["tenengrad_variance"],
            seed=1,
            max_evals=40,
            min_gap_closed=0.0,
            max_residual=1e-6,
        )


def test_fit_without_a_ceiling_is_unchanged_by_the_new_knob():
    x = _volume()
    truth = DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.5),))
    target = measure_attributes(_degrade(truth, x, seed=1), attributes=["tenengrad_variance"])
    result = fit_chain(
        x,
        axes=["complex_gaussian"],
        target=target,
        attributes=["tenengrad_variance"],
        seed=1,
        max_evals=60,
    )
    assert result.residual >= 0.0
