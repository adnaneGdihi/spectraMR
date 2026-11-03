"""Powered ranking certification — finite-sample gate (critique §3 repair).

Verifies the executable counterpart of the Lean theorems
``Sim2Rank.powered_sample_size`` / ``sample_size_for_recovery`` against the
closed-form bound, including the exact numeric the critique cites (K=20,
delta=0.1, 99% confidence -> ~1.3e4 samples).
"""

from __future__ import annotations

import math

import pytest

from spectramr.core.metrics.meta_evaluation.powered import (
    certify_partial_order,
    min_samples_for_gap,
    min_samples_for_pool,
    powered_gap_threshold,
)


def test_min_samples_matches_closed_form() -> None:
    # n_min = 2 * ceil(8 ln(2/alpha) / delta^2)  (critique eq. 2, per-pair).
    delta, alpha = 0.1, 0.01
    expected = 2 * math.ceil(8.0 * math.log(2.0 / alpha) / delta**2)
    assert min_samples_for_gap(delta, alpha) == expected


def test_pool_is_pair_with_unit_k() -> None:
    # The pool formula at K=1 reduces to the per-pair formula (ln(2*1/a)=ln(2/a)).
    assert min_samples_for_pool(0.1, 1, 0.01) == min_samples_for_gap(0.1, 0.01)


def test_min_samples_critique_numeric() -> None:
    # K=20 pool, min gap 0.1, 99% confidence. The whole-sort guarantee union-bounds
    # over the C(20,2)=190 PAIRS that must all be correct (critique L6), so the log
    # term is ln(2*190/0.01)=ln(38000), giving ~1.69e4 samples per axis — more than
    # the earlier (under-counted) ln(2*20/0.01) ~1.33e4.
    n = min_samples_for_pool(delta=0.1, n_metrics=20, alpha=0.01)
    assert 16_800 <= n <= 16_950


def test_bounded_scores_rescales_unit_range_preserving_gaps() -> None:
    # A3: powered certification must run on the [-1, 1] scale the Hoeffding gate
    # assumes. The rescale preserves ordering AND relative gaps (it is linear).
    import math

    from spectramr.core.metrics.meta_evaluation.pipeline import _bounded_scores

    b = _bounded_scores({"a": 10.0, "b": 30.0, "c": 50.0})
    assert min(b.values()) == pytest.approx(-1.0)
    assert max(b.values()) == pytest.approx(1.0)
    assert b["a"] < b["b"] < b["c"]
    # equal raw gaps (20, 20) stay equal after the linear rescale.
    assert (b["c"] - b["b"]) == pytest.approx(b["b"] - b["a"])
    # non-finite scores (crashed metrics) pass through for the cert's own guard.
    passthrough = _bounded_scores({"x": 1.0, "y": float("inf"), "z": float("nan")})
    assert passthrough["y"] == float("inf")
    assert math.isnan(passthrough["z"])


def test_gap_threshold_inverts_min_samples() -> None:
    # A sample size at exactly n_min must make delta certifiable (gap >= threshold).
    delta, alpha = 0.3, 0.05
    n = min_samples_for_gap(delta, alpha)
    thr = powered_gap_threshold(n, alpha)
    assert thr <= delta + 1e-9


def test_gap_threshold_monotone_in_n() -> None:
    # More samples -> finer (smaller) certifiable gap.
    a = powered_gap_threshold(100, 0.05)
    b = powered_gap_threshold(10_000, 0.05)
    assert b < a


def test_gap_threshold_infinite_below_one_block() -> None:
    assert powered_gap_threshold(1, 0.05) == math.inf
    assert powered_gap_threshold(0, 0.05) == math.inf


def test_certify_orders_only_powered_pairs() -> None:
    # Large N: even small gaps clear the threshold -> all pairs certified.
    scores = {"a": 0.9, "b": 0.6, "c": 0.3}
    po = certify_partial_order(scores, n_samples=1_000_000, alpha=0.05)
    assert po.is_certified("a", "b")
    assert po.is_certified("a", "c")
    assert po.is_certified("b", "c")
    assert po.n_ambiguous == 0


def test_certify_withholds_underpowered_pairs() -> None:
    # Tiny N: nothing is powered -> all pairs ambiguous, none certified.
    scores = {"a": 0.51, "b": 0.50, "c": 0.49}
    po = certify_partial_order(scores, n_samples=4, alpha=0.05)
    assert po.n_certified == 0
    assert po.n_ambiguous == 3
    assert not po.is_certified("a", "c")


def test_certify_partitions_pairs() -> None:
    # Mixed: a wide gap certified, an adjacent narrow gap withheld.
    scores = {"top": 1.0, "mid": 0.05, "low": 0.0}
    po = certify_partial_order(scores, n_samples=200, alpha=0.05)
    total_pairs = 3  # C(3, 2)
    assert po.n_certified + po.n_ambiguous == total_pairs
    # top dominates both others by a wide margin.
    assert po.is_certified("top", "mid")
    assert po.is_certified("top", "low")
    # mid vs low gap (0.05) is below threshold at N=200.
    assert ("mid", "low") in po.ambiguous


def test_certify_never_certifies_nonfinite_scores() -> None:
    # crash→NaN contract: a metric with a non-finite score carries no defensible
    # gap. An ``inf`` score is the dangerous case — ``gap = inf - x = inf`` and
    # ``abs(inf) >= threshold`` is True, so the naive rule would SPURIOUSLY certify
    # the crashed metric as the better one. It must land in ambiguous instead.
    po = certify_partial_order(
        {"crash": float("inf"), "nan": float("nan"), "real": 1.0},
        n_samples=200,
        alpha=0.05,
    )
    flat = [m for pair in po.certified for m in pair]
    assert "crash" not in flat and "nan" not in flat
    assert ("crash", "real") in po.ambiguous or ("real", "crash") in po.ambiguous
    assert po.n_certified + po.n_ambiguous == 3  # C(3, 2), nothing lost


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_alpha_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        min_samples_for_gap(0.1, alpha)
    with pytest.raises(ValueError):
        powered_gap_threshold(100, alpha)


def test_nonpositive_delta_rejected() -> None:
    with pytest.raises(ValueError):
        min_samples_for_gap(0.0, 0.05)
    with pytest.raises(ValueError):
        min_samples_for_gap(-0.2, 0.05)


class TestHolmAcrossPairs:
    """The per-pair alpha holds per pair only; Holm controls the family of pairs."""

    def test_one_pair_under_holm_matches_the_uncorrected_gate(self) -> None:
        n, alpha = 200, 0.05
        t = powered_gap_threshold(n, alpha)
        for gap in (0.999 * t, 1.001 * t):
            plain = certify_partial_order({"a": gap, "b": 0.0}, n, alpha)
            holm = certify_partial_order({"a": gap, "b": 0.0}, n, alpha, correction="holm")
            assert plain.certified == holm.certified

    def test_holm_withholds_a_pair_the_per_pair_gate_certifies(self) -> None:
        n, alpha = 200, 0.05
        t = powered_gap_threshold(n, alpha)
        scores = {f"m{i}": i * 1.05 * t for i in range(6)}
        plain = certify_partial_order(scores, n, alpha)
        holm = certify_partial_order(scores, n, alpha, correction="holm")
        assert ("m1", "m0") in plain.certified
        assert ("m1", "m0") not in holm.certified
        assert set(holm.certified) < set(plain.certified)
        assert holm.correction == "holm"

    def test_an_unknown_correction_raises(self) -> None:
        with pytest.raises(ValueError, match="correction"):
            certify_partial_order({"a": 1.0, "b": 0.0}, 100, 0.05, correction="bonferonni")
