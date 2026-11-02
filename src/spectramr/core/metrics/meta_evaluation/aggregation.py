"""Six-method aggregator with the asymmetric defensibility rule.

Each method's per-metric scores are first standardized to z-scores across
metrics, then combined as a Dirichlet-weighted mean. The defensibility
flag is set when:

    * z >= 0 on at least a supermajority of the ranker pool, AND
    * z >= ``floor_z`` on every ranker (no catastrophic failure).

The default weights ``{CDSCR: 0.17, LGDR: 0.17, SFA: 0.21, ESD: 0.13,
FSPD: 0.17, Sim2Rank: 0.15}`` keep SFA highest (most theoretically
grounded), demote ESD slightly (Monte-Carlo noisy on small datasets),
and treat the new Sim2Rank composite at parity with the original
spectral / FPCA rankers.

By default the breadth threshold is a FRACTION of the ranker pool
(``min_positive_frac`` = 5/6, i.e. the original "5 of 6" rule), so it
generalizes as the pool grows: ``MetaEvaluationPipeline.from_registry`` wires
~12 rankers, and a hard-coded absolute 5 silently degraded that rule to "5 of
12" (~42%, not the intended ~83%) — #296. An explicit ``min_positive_axes``
overrides it with an absolute count (backward compatible). ``floor_z`` (-1.0)
keeps the asymmetric "breadth of agreement, no catastrophic failure" rule.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import numpy as np

from .ranking_guards import order_from_scores, ranks_from_scores
from .types import AggregateRankingResult, DefensibilityCalibration, RankingResult

logger = logging.getLogger(__name__)


@dataclass
class AggregatorConfig:
    weights: dict[str, float] | None = None
    # Absolute breadth threshold (number of non-negative rankers). ``None`` (the
    # default) means "scale with the pool" via ``min_positive_frac`` — see #296.
    # An explicit int overrides the fraction with a fixed count (backward compat).
    min_positive_axes: int | None = None
    # Supermajority fraction of the ranker pool used when ``min_positive_axes`` is
    # None. 5/6 reproduces the original "5 of 6" rule at 6 rankers and generalizes
    # to ceil(5/6 * n) as the pool grows.
    min_positive_frac: float = 5.0 / 6.0
    floor_z: float = -1.0
    # Permutation calibration of the defensibility flag (critique §2). Off by
    # default — the B·R permutation cost is opt-in.
    calibrate: bool = False
    n_perm: int = 1000
    calibration_alpha: float = 0.05
    calibration_seed: int = 0

    def resolved_weights(self) -> dict[str, float]:
        return self.weights or {
            "CDSCR": 0.17,
            "LGDR": 0.17,
            "SFA": 0.21,
            "ESD": 0.13,
            "FSPD": 0.17,
            "Sim2Rank": 0.15,
        }


def _required_positive_methods(
    n_methods: int,
    min_positive_axes: int | None,
    min_positive_frac: float,
) -> int:
    """Non-negative-ranker count a metric needs to clear the breadth rule.

    When ``min_positive_axes`` is set it is an absolute count (backward compat).
    Otherwise the count scales as ``ceil(min_positive_frac * n_methods)`` so the
    original "5 of 6 rankers" rule generalizes to whatever pool ``from_registry``
    wired — an absolute 5 silently became "5 of 12" (~42%, not ~83%) once the pool
    grew (#296).
    """
    if min_positive_axes is not None:
        return int(min_positive_axes)
    return max(1, math.ceil(min_positive_frac * max(n_methods, 1)))


def aggregate(
    results: list[RankingResult],
    config: AggregatorConfig | None = None,
) -> AggregateRankingResult:
    cfg = config or AggregatorConfig()
    weights = cfg.resolved_weights()

    # Uniform fallback for any present-but-unweighted method. Without this, a
    # ranker resolved via ``MetaEvaluationPipeline.from_registry`` whose method
    # name is not in the (curated, six-key) default weight dict would silently get
    # ``weights.get(method, 0.0) == 0`` — computed at full cost but contributing
    # nothing to the aggregate (a dead axis; the from_registry "wires every ranker"
    # promise broken). Give unweighted methods the mean of the declared weights so
    # they count comparably, and warn (named) so the omission is auditable, never
    # silent (CLAUDE.md pitfall #16).
    declared = list(weights.values())
    fallback_w = (sum(declared) / len(declared)) if declared else 1.0
    present_methods = {r.method for r in results}
    unweighted = sorted(present_methods - set(weights))
    if unweighted:
        logger.warning(
            "aggregate(): methods %s have no declared weight in AggregatorConfig."
            "weights; using the uniform fallback weight %.4f for each. Add an "
            "explicit weight to override.",
            unweighted,
            fallback_w,
        )

    # Collect z-scored per-method scores; a metric any axis scored is placed on all.
    metrics = sorted({m for r in results for m in r.scores})
    per_method_z: dict[str, dict[str, float]] = {r.method: r.standardize(metrics) for r in results}
    # Cells whose z was imputed (no finite score); a crash among them trips the floor.
    imputed = {
        r.method: {m for m in metrics if not math.isfinite(r.scores.get(m, math.nan))}
        for r in results
    }
    crashed = {
        r.method: {m for m in imputed[r.method] if r.scores.get(m, math.nan) != math.inf}
        for r in results
    }

    # Breadth threshold: absolute when min_positive_axes is set, else a fraction
    # of the ranker pool so it tracks the pool size (#296).
    required_positive = _required_positive_methods(
        len(per_method_z), cfg.min_positive_axes, cfg.min_positive_frac
    )

    # Final weighted z mean.
    final: dict[str, float] = {}
    flags: dict[str, bool] = {}
    for m in metrics:
        weighted_sum = 0.0
        weight_total = 0.0
        positive_count = 0
        any_below_floor = False
        for method, z_dict in per_method_z.items():
            w = weights.get(method, fallback_w)
            if m not in z_dict:
                continue
            z = z_dict[m]
            weighted_sum += w * z
            weight_total += w
            if z >= 0:
                positive_count += 1
            if z < cfg.floor_z or m in crashed[method]:
                any_below_floor = True
        final[m] = weighted_sum / max(weight_total, 1e-12)
        flags[m] = positive_count >= required_positive and not any_below_floor

    final_ranks = order_from_scores(final)
    calibration = None
    if cfg.calibrate:
        calibration = calibrate_defensibility(
            per_method_z,
            flags,
            min_positive_axes=cfg.min_positive_axes,
            min_positive_frac=cfg.min_positive_frac,
            floor_z=cfg.floor_z,
            alpha=cfg.calibration_alpha,
            n_perm=cfg.n_perm,
            seed=cfg.calibration_seed,
            imputed=imputed,
        )
    return AggregateRankingResult(
        final_scores=final,
        final_ranks=final_ranks,
        per_method_z=per_method_z,
        defensibility_flags=flags,
        weights=weights,
        calibration=calibration,
        defensibility_semantics="breadth",
        final_mid_ranks=_mid_ranks(final_ranks, final),
    )


def _mid_ranks(order: list[str], scores: dict[str, float]) -> dict[str, float]:
    """1-indexed mid-ranks of ``scores`` keyed by name; equal scores share a rank."""
    mid = ranks_from_scores([scores[m] for m in order], names=order)
    return {m: float(r) for m, r in zip(order, mid, strict=True)}


def calibrate_defensibility(
    per_method_z: dict[str, dict[str, float]],
    flags: dict[str, bool],
    *,
    min_positive_axes: int | None,
    floor_z: float,
    min_positive_frac: float = 5.0 / 6.0,
    alpha: float = 0.05,
    n_perm: int = 1000,
    seed: int = 0,
    imputed: dict[str, set[str]] | None = None,
) -> DefensibilityCalibration:
    """Permutation-calibrate the false-positive rate of the defensibility flag.

    Under the exchangeability null (a metric carries no real cross-axis signal),
    its per-axis z-scores are interchangeable with the rest of the pool's on that
    axis. Permuting the metric labels *independently within each axis* destroys any
    genuine cross-axis consistency while preserving each axis's marginal z
    distribution — the correct null for the joint Type-I error the critique flags
    (the rankers share data, so the unanimity event is not a product of marginals).

    Returns the flag's empirical null false-positive rate, plus a per-metric
    permutation p-value for the breadth-of-agreement statistic (number of
    non-negative axes) and the calibrated flags (raw flag AND ``p <= alpha``).
    ``imputed`` names, per axis, the cells whose z was placed rather than measured
    (see :meth:`RankingResult.standardize`); they are masked out of the null.
    """
    methods = list(per_method_z.keys())
    metrics = sorted({m for z in per_method_z.values() for m in z})
    n_axes, n_metrics = len(methods), len(metrics)
    if n_axes == 0 or n_metrics == 0:
        return DefensibilityCalibration(0.0, {}, {}, n_perm, alpha)

    # Same breadth threshold as aggregate() so the null matches the observed rule
    # (#296): absolute when set, else a fraction of the ranker pool.
    required_positive = _required_positive_methods(n_axes, min_positive_axes, min_positive_frac)

    # z-matrix [axes, metrics]; NaN masks a cell with no measured score on that axis.
    z_mat = np.full((n_axes, n_metrics), np.nan)
    skip = imputed or {}
    for ai, method in enumerate(methods):
        zd = per_method_z[method]
        for ki, m in enumerate(metrics):
            if m in zd and m not in skip.get(method, ()):
                z_mat[ai, ki] = zd[m]

    present = ~np.isnan(z_mat)
    pos_mat = (z_mat >= 0.0) & present  # non-negative axes
    below_mat = (z_mat < floor_z) & present  # catastrophic axes
    obs_count = pos_mat.sum(axis=0)  # breadth-of-agreement per metric

    rng = np.random.default_rng(seed)
    null_counts = np.empty(n_perm * n_metrics)
    fire = 0
    idx = 0
    for _ in range(n_perm):
        perm = np.stack([rng.permutation(n_metrics) for _ in range(n_axes)])
        rows = np.arange(n_axes)[:, None]
        c = pos_mat[rows, perm].sum(axis=0)
        b = below_mat[rows, perm].any(axis=0)
        fire += int(((c >= required_positive) & ~b).sum())
        null_counts[idx : idx + n_metrics] = c
        idx += n_metrics

    null_fpr = fire / (n_perm * n_metrics)
    denom = n_perm * n_metrics + 1
    pvalues: dict[str, float] = {}
    calibrated: dict[str, bool] = {}
    for ki, m in enumerate(metrics):
        # +1 correction: a valid Monte-Carlo permutation p-value.
        p = (1 + int((null_counts >= obs_count[ki]).sum())) / denom
        pvalues[m] = float(p)
        calibrated[m] = bool(flags.get(m, False) and p <= alpha)
    return DefensibilityCalibration(
        null_fpr=float(null_fpr),
        pvalues=pvalues,
        calibrated_flags=calibrated,
        n_perm=n_perm,
        alpha=alpha,
    )


# --------------------------------------------------------------------------- #
# Layer L3⁺ — Kemeny-median (Condorcet-consistent) cross-axis aggregation
# --------------------------------------------------------------------------- #
# The default ``aggregate`` above is z-weighted (Pareto-consistent only). The
# Kemeny consensus below is the Condorcet-consistent alternative the self-critique
# (TODO/sim2rank_critique.md §1) calls for, and is the executable shadow of the
# Lean module ``Sim2Rank.Kemeny`` (docs/lean/Sim2Rank/Sim2Rank/Kemeny.lean):
# ``kemeny_consensus`` realises ``kemenyScore`` + ``kemeny_optimal_ranks_*``, and
# ``kemeny_optimal_ranks_pareto_winner_first`` guarantees a unanimous (Pareto)
# winner is ranked first — strictly stronger than the z-aggregator's ``borda_pareto``.


@dataclass
class KemenyConfig:
    """Configuration for the Kemeny-median cross-axis aggregator."""

    weights: dict[str, float] | None = None  # per-axis (per-ranker) voting weights
    max_exact: int = 8  # exact permutation search up to this pool size; else local search

    def resolved_weights(self) -> dict[str, float]:
        return self.weights or {}  # empty ⇒ uniform (weight 1 per axis)


def _axis_positions(result: RankingResult, metrics: list[str]) -> dict[str, int]:
    """Position of each metric in one axis's ranking (lower position = better).

    A metric this axis did not score (absent, NaN or ``-inf``) shares the bottom
    position, the placement :meth:`RankingResult.standardize` gives the z path.
    """

    def scored(m: str) -> bool:
        v = result.scores.get(m, math.nan)
        return math.isfinite(v) or v == math.inf

    order = [m for m in result.ranks if m in metrics and scored(m)]
    seen = set(order)
    # metrics this axis scored but did not rank, by descending score then name
    extra = order_from_scores({m: result.scores[m] for m in metrics if scored(m) and m not in seen})
    full = order + extra
    pos = {m: i for i, m in enumerate(full)}
    pos.update(dict.fromkeys((m for m in metrics if not scored(m)), len(full)))
    return pos


def _majority_margin(
    results: list[RankingResult], metrics: list[str], weights: dict[str, float]
) -> dict[tuple[str, str], float]:
    """Antisymmetric majority margin ``w[a, b]`` (Lean ``margin`` / ``margin_antisymm``).

    ``w[a, b] = ∑_axes weight·(+1 if a above b, −1 if b above a)``. A unanimous
    preference yields a strictly positive margin (Lean ``margin_pos_of_unanimous``),
    making a Pareto winner a Condorcet winner.
    """
    w: dict[tuple[str, str], float] = {(a, b): 0.0 for a in metrics for b in metrics if a != b}
    for r in results:
        weight = weights.get(r.method, 1.0)
        pos = _axis_positions(r, metrics)
        for a in metrics:
            for b in metrics:
                if a == b:
                    continue
                if pos[a] < pos[b]:
                    w[(a, b)] += weight
                elif pos[b] < pos[a]:
                    w[(a, b)] -= weight
    return w


def _kemeny_score(order: list[str], w: dict[tuple[str, str], float]) -> float:
    """Total agreement of a full order with the margin (Lean ``kemenyScore``)."""
    total = 0.0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            total += w[(order[i], order[j])]
    return total


def _local_search_order(metrics: list[str], w: dict[tuple[str, str], float]) -> list[str]:
    """Copeland-seeded adjacent-swap local search (for pools beyond ``max_exact``)."""
    copeland = {m: sum(w[(m, b)] for b in metrics if b != m) for m in metrics}
    order = sorted(metrics, key=lambda m: -copeland[m])
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            # swapping adjacent (a, b) changes the score by w[b,a] - w[a,b] = -2 w[a,b]
            if w[(a, b)] < 0:  # b should be above a
                order[i], order[i + 1] = b, a
                improved = True
    return order


def kemeny_consensus(
    results: list[RankingResult], config: KemenyConfig | None = None
) -> AggregateRankingResult:
    """Condorcet-consistent Kemeny-median consensus over per-axis rankings.

    Maximises the total majority-margin agreement (Lean ``kemenyScore``); for pools
    up to ``config.max_exact`` the optimum is found by exhaustive permutation search
    (exact Kemeny), otherwise by Copeland-seeded local search. The returned
    ``AggregateRankingResult`` carries the consensus order in ``final_ranks`` and a
    position-descending ``final_scores`` (so ``sorted(final_scores)`` reproduces the
    order); ``defensibility_flags`` mark Condorcet winners (a metric whose margin
    beats every opponent — Lean ``kemeny_optimal_ranks_condorcet_first``).
    """
    cfg = config or KemenyConfig()
    weights = cfg.resolved_weights()
    metrics = sorted({m for r in results for m in r.scores})
    per_method_z = {r.method: r.standardize(metrics) for r in results}
    if not metrics:
        return AggregateRankingResult(
            final_scores={},
            final_ranks=[],
            per_method_z=per_method_z,
            defensibility_flags={},
            weights=weights,
            calibration=None,
            defensibility_semantics="condorcet",
        )

    w = _majority_margin(results, metrics, weights)

    if len(metrics) <= cfg.max_exact:
        best_order = max(itertools.permutations(metrics), key=lambda o: _kemeny_score(list(o), w))
        order = list(best_order)
    else:
        order = _local_search_order(metrics, w)

    n = len(order)
    final_scores = {m: float(n - 1 - i) for i, m in enumerate(order)}
    # Metrics placed identically on every axis are indistinguishable and share a rank.
    axis_pos = [_axis_positions(r, metrics) for r in results]
    profile = {m: tuple(p[m] for p in axis_pos) for m in order}
    tied_scores = dict(final_scores)
    for i in range(1, n):
        if profile[order[i]] == profile[order[i - 1]]:
            tied_scores[order[i]] = tied_scores[order[i - 1]]
    # Condorcet winner: strictly positive margin against every opponent. Requires
    # at least two metrics — for a singleton pool ``all(<empty>)`` is vacuously
    # True, which would spuriously certify the lone metric a "Condorcet winner".
    flags = {
        m: (len(metrics) >= 2 and all(w[(m, b)] > 0.0 for b in metrics if b != m)) for m in metrics
    }
    return AggregateRankingResult(
        final_scores=final_scores,
        final_ranks=order,
        per_method_z=per_method_z,
        defensibility_flags=flags,
        weights=weights,
        calibration=None,
        defensibility_semantics="condorcet",
        final_mid_ranks=_mid_ranks(order, tied_scores),
    )
