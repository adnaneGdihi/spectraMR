"""Aggregator + permutation calibration of the defensibility flag (critique §2)."""

from __future__ import annotations

from spectramr.core.metrics.meta_evaluation.aggregation import (
    AggregatorConfig,
    aggregate,
    calibrate_defensibility,
)
from spectramr.core.metrics.meta_evaluation.types import RankingResult


def _six_axis_result(method: str, scores: dict[str, float]) -> RankingResult:
    return RankingResult(
        method=method, scores=scores, ranks=sorted(scores, key=lambda k: -scores[k])
    )


def test_calibration_off_by_default() -> None:
    r = _six_axis_result("CDSCR", {"a": 1.0, "b": 0.0})
    out = aggregate([r], AggregatorConfig(weights={"CDSCR": 1.0}, min_positive_axes=1))
    assert out.calibration is None


def test_strong_unanimous_metric_is_calibrated_significant() -> None:
    # 'winner' is top on all 6 axes; 'a'..'d' are a noisy cloud. With 6 axes the
    # null prob of being non-negative on all 6 by chance is small, so the winner's
    # breadth p-value should clear alpha while the flag stays True.
    methods = ["CDSCR", "LGDR", "SFA", "ESD", "FSPD", "Sim2Rank"]
    pool = ["winner", "a", "b", "c", "d"]
    results = []
    for i, meth in enumerate(methods):
        scores = {"winner": 5.0}
        # deterministic but varied middling scores for the others
        for j, m in enumerate(pool[1:]):
            scores[m] = float(((i + 1) * (j + 1)) % 3) - 1.0
        results.append(_six_axis_result(meth, scores))
    cfg = AggregatorConfig(
        weights=dict.fromkeys(methods, 1.0),
        min_positive_axes=5,
        floor_z=-1.5,
        calibrate=True,
        n_perm=400,
        calibration_seed=0,
    )
    out = aggregate(results, cfg)
    cal = out.calibration
    assert cal is not None
    # The flag fired for the winner...
    assert out.defensibility_flags["winner"] is True
    # ...and it survives calibration (beyond-chance breadth of agreement).
    assert cal.calibrated_flags["winner"] is True
    assert cal.pvalues["winner"] <= 0.05
    # null_fpr is a probability.
    assert 0.0 <= cal.null_fpr <= 1.0


def test_calibration_is_deterministic_under_seed() -> None:
    methods = ["CDSCR", "LGDR", "SFA", "ESD", "FSPD", "Sim2Rank"]
    results = [
        _six_axis_result(m, {"x": float(i) - 2.5, "y": 2.5 - float(i), "z": 0.0})
        for i, m in enumerate(methods)
    ]
    z = {r.method: r.standardize() for r in results}
    flags = dict.fromkeys(["x", "y", "z"], True)
    a = calibrate_defensibility(
        z, flags, min_positive_axes=5, floor_z=-1.0, n_perm=200, seed=7
    )
    b = calibrate_defensibility(
        z, flags, min_positive_axes=5, floor_z=-1.0, n_perm=200, seed=7
    )
    assert a.pvalues == b.pvalues
    assert a.null_fpr == b.null_fpr


def test_empty_pool_calibration_safe() -> None:
    cal = calibrate_defensibility(
        {}, {}, min_positive_axes=5, floor_z=-1.0, n_perm=10, seed=0
    )
    assert cal.null_fpr == 0.0
    assert cal.pvalues == {}


def test_unweighted_present_method_gets_uniform_fallback_not_zero() -> None:
    """A present method with no declared weight must still affect the aggregate.

    Pre-fix ``weights.get(method, 0.0)`` zeroed any ranker (e.g. resolved via
    ``from_registry``) whose name was not in the curated 6-key weight dict — it ran
    at full cost but contributed nothing (a dead axis; critique B1). The uniform
    fallback gives it the mean of the declared weights so it genuinely counts.
    """
    # Two axes: one weighted ("CDSCR"), one UNweighted ("ADR"). They disagree on
    # which metric is best. If ADR were weight-0, "a" would win outright; with the
    # uniform fallback ADR pulls "b" up so the gap closes / flips.
    weighted = _six_axis_result("CDSCR", {"a": 2.0, "b": -2.0})
    unweighted = _six_axis_result("ADR", {"a": -2.0, "b": 2.0})
    cfg = AggregatorConfig(weights={"CDSCR": 0.5})  # ADR absent from the dict
    out = aggregate([weighted, unweighted], cfg)
    # ADR contributed: the two opposite axes cancel, so the scores are ~equal,
    # NOT dominated by CDSCR alone (which would give final["a"] > 0 >> final["b"]).
    assert abs(out.final_scores["a"] - out.final_scores["b"]) < 1e-9


def test_aggregate_stamps_breadth_semantics() -> None:
    """The z-aggregator labels its defensibility flag as 'breadth' (critique A2)."""
    out = aggregate(
        [_six_axis_result("CDSCR", {"a": 1.0, "b": 0.0})],
        AggregatorConfig(weights={"CDSCR": 1.0}, min_positive_axes=1),
    )
    assert out.defensibility_semantics == "breadth"


def _pool_results(n_methods: int, positive_methods: int) -> list[RankingResult]:
    """``n_methods`` rankers; ``target`` is above-mean (z>=0) on the first
    ``positive_methods`` of them and below-mean on the rest."""
    results = []
    for i in range(n_methods):
        hi = i < positive_methods
        scores = {"target": 5.0 if hi else -5.0, "f1": 0.0, "f2": 0.0}
        results.append(
            RankingResult(
                method=f"M{i}",
                scores=scores,
                ranks=sorted(scores, key=lambda k: -scores[k]),
            )
        )
    return results


def test_min_positive_scales_with_ranker_pool_size() -> None:
    # #296: "5 of 6 axes" was an ABSOLUTE 5. from_registry wires ~12 rankers, so
    # absolute-5 silently became "5 of 12" (~42%), not the intended ~83%. The
    # DEFAULT (no min_positive_axes) now scales as ceil(5/6 * n) = 10 at n=12.
    n = 12
    weights = {f"M{i}": 1.0 for i in range(n)}
    cfg = AggregatorConfig(weights=weights, floor_z=-100.0)  # default fraction path

    out5 = aggregate(_pool_results(n, positive_methods=5), cfg)
    assert (
        out5.defensibility_flags["target"] is False
    ), "5 of 12 (~42%) must NOT be defensible under the fraction rule"
    out10 = aggregate(_pool_results(n, positive_methods=10), cfg)
    assert out10.defensibility_flags["target"] is True, "10 of 12 clears ceil(5/6*12)"


def test_explicit_min_positive_axes_stays_absolute() -> None:
    # No silent override (#15): an explicit min_positive_axes is honored as an
    # absolute count regardless of pool size, so pre-existing callers are unchanged.
    n = 12
    weights = {f"M{i}": 1.0 for i in range(n)}
    cfg = AggregatorConfig(weights=weights, min_positive_axes=5, floor_z=-100.0)
    out5 = aggregate(_pool_results(n, positive_methods=5), cfg)
    assert out5.defensibility_flags["target"] is True, "absolute 5 met at 5 of 12"


class TestMissingScoresAndTies:
    """A crashed metric is worst on that axis; +inf is best; ties are deterministic."""

    @staticmethod
    def _crash_pool() -> list[RankingResult]:
        # "crashy" is middling on the one axis where it scored and crashed on the other two.
        nan = float("nan")
        rows = {
            "A": {"crashy": 0.9, "good": 2.0, "mid": 1.0, "weak": 0.0},
            "B": {"crashy": nan, "good": 2.0, "mid": 1.0, "weak": 0.0},
            "C": {"crashy": nan, "good": 2.0, "mid": 1.0, "weak": 0.0},
        }
        from spectramr.core.metrics.meta_evaluation.ranking_guards import order_from_scores

        # Real rankers order through order_from_scores, which puts NaN last.
        return [RankingResult(m, s, order_from_scores(s)) for m, s in rows.items()]

    def test_a_crashed_metric_is_scored_as_worst_on_that_axis(self) -> None:
        out = aggregate(self._crash_pool(), AggregatorConfig(weights=dict.fromkeys("ABC", 1.0)))
        z_c = out.per_method_z["C"]
        assert z_c["crashy"] < min(v for k, v in z_c.items() if k != "crashy")

    def test_z_and_kemeny_agree_on_placing_a_crashed_metric(self) -> None:
        from spectramr.core.metrics.meta_evaluation.aggregation import kemeny_consensus

        pool = self._crash_pool()
        z = aggregate(pool, AggregatorConfig(weights=dict.fromkeys("ABC", 1.0))).final_ranks
        k = kemeny_consensus(pool).final_ranks
        assert z.index("crashy") == k.index("crashy") == 3

    def test_an_absent_metric_counts_as_worst(self) -> None:
        pool = [
            _six_axis_result("A", {"a": 1.0, "b": 0.0}),
            _six_axis_result("B", {"b": 0.0}),
        ]
        z_b = aggregate(pool, AggregatorConfig(weights=dict.fromkeys("AB", 1.0))).per_method_z["B"]
        assert z_b["a"] < z_b["b"]

    def test_plus_inf_is_kept_as_best(self) -> None:
        pool = [_six_axis_result("A", {"inf": float("inf"), "x": 1.0, "y": 0.0})]
        out = aggregate(pool, AggregatorConfig(weights={"A": 1.0}, min_positive_axes=1))
        assert out.final_ranks[0] == "inf"

    def test_ties_sort_by_name_and_share_a_rank(self) -> None:
        pool = [_six_axis_result("A", {"zeta": 1.0, "alpha": 1.0, "mid": 0.5, "low": 0.0})]
        out = aggregate(pool, AggregatorConfig(weights={"A": 1.0}, min_positive_axes=1))
        assert out.final_ranks == ["alpha", "zeta", "mid", "low"]
        assert out.final_mid_ranks == {"alpha": 1.5, "zeta": 1.5, "mid": 3.0, "low": 4.0}

    def test_kemeny_gives_indistinguishable_metrics_one_rank(self) -> None:
        from spectramr.core.metrics.meta_evaluation.aggregation import kemeny_consensus

        nan = float("nan")
        pool = [
            RankingResult("A", {"a": 2.0, "b": 1.0, "x": nan, "y": nan}, ["a", "b", "x", "y"]),
            RankingResult("B", {"a": 2.0, "b": 1.0, "x": nan, "y": nan}, ["a", "b", "x", "y"]),
        ]
        out = kemeny_consensus(pool)
        assert out.final_ranks == ["a", "b", "x", "y"]
        assert out.final_mid_ranks == {"a": 1.0, "b": 2.0, "x": 3.5, "y": 3.5}

    def test_order_is_stable_across_hash_seeds(self) -> None:
        import os
        import subprocess
        import sys

        script = (
            "from spectramr.core.metrics.meta_evaluation.aggregation import "
            "AggregatorConfig, aggregate\n"
            "from spectramr.core.metrics.meta_evaluation.types import RankingResult\n"
            "names = [f'm{i}' for i in range(40)]\n"
            "r = RankingResult('A', dict.fromkeys(names, 1.0), names)\n"
            "print(','.join(aggregate([r], AggregatorConfig(weights={'A': 1.0}, "
            "min_positive_axes=1)).final_ranks))\n"
        )
        outs = set()
        for seed in ("1", "2", "3"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            res = subprocess.run(
                [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
            )
            outs.add(res.stdout.strip())
        assert len(outs) == 1


class TestImputedScoresAndTheFloor:
    """An imputed (crashed) score trips the floor and stays out of the null."""

    def test_a_crash_on_a_constant_axis_trips_the_floor(self) -> None:
        nan = float("nan")
        # Axis B is constant over its finite scores, so worst - 1 == -1.0 == floor_z.
        pool = [
            _six_axis_result("A", {"a": 2.0, "b": 1.0, "c": 0.0}),
            _six_axis_result("B", {"a": nan, "b": 1.0, "c": 1.0}),
        ]
        cfg = AggregatorConfig(weights=dict.fromkeys("AB", 1.0), min_positive_axes=1, floor_z=-1.0)
        out = aggregate(pool, cfg)
        assert out.per_method_z["B"]["a"] == -1.0
        assert out.defensibility_flags["a"] is False

    def test_imputed_entries_are_masked_from_the_permutation_null(self) -> None:
        nan = float("nan")
        methods = ["A", "B", "C"]
        rows = {
            "A": {"a": 1.0, "b": 0.5, "c": 0.0, "d": 0.7},
            "B": {"a": 0.2, "b": 1.0, "c": 0.0, "d": 0.4},
            "C": {"a": 1.0, "b": 0.0, "c": 0.5, "d": nan},
        }
        pool = [_six_axis_result(m, s) for m, s in rows.items()]
        cfg = AggregatorConfig(
            weights=dict.fromkeys(methods, 1.0), calibrate=True, n_perm=200, min_positive_axes=2
        )
        out = aggregate(pool, cfg)
        # The old null: the crashed cell is simply absent from axis C.
        old_z = {m: dict(out.per_method_z[m]) for m in methods}
        del old_z["C"]["d"]
        ref = calibrate_defensibility(
            old_z,
            out.defensibility_flags,
            min_positive_axes=2,
            floor_z=cfg.floor_z,
            n_perm=200,
            seed=cfg.calibration_seed,
        )
        assert out.calibration.pvalues == ref.pvalues
        assert out.calibration.null_fpr == ref.null_fpr
