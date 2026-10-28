"""End-to-end orchestrator.

The pipeline:

1. Runs the simulator to build a ``MetricEvaluationDataset``.
2. Computes per-metric scalar values (cached on the dataset).
3. Runs each ranker.
4. Aggregates with the asymmetric-defensibility rule.

Usage
-----

>>> pipeline = MetaEvaluationPipeline.from_defaults()
>>> result = pipeline.run(metric_set, clean_volumes)
>>> result.final_ranks   # ['psnr', 'ssim', 'rmse', ...]
>>> result.defensibility_flags
{'psnr': False, 'ssim': True, ...}
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from spectramr.core.metrics.context import MetricContext

from .aggregation import AggregatorConfig, KemenyConfig, aggregate, kemeny_consensus
from .betting import (
    BettingPartialOrder,
    certify_betting_order,
    kendall_concordance_increments,
)
from .pool_hygiene import PoolScreening, screen_pool
from .powered import PoweredPartialOrder, certify_partial_order
from .rankers import (
    BaseRanker,
    CDSCRRanker,
    ESDRanker,
    FSPDRanker,
    LGDRRanker,
    SFARanker,
    Sim2RankRanker,
)
from .simulator import SimulatorConfig, precompute_metric_values, run_simulator
from .transfer import (
    TransferCertificate,
    tv_distance_joint_samples,
    tv_distance_samples,
    tv_transfer_certificate,
)
from .types import (
    AggregateRankingResult,
    MetricEvaluationDataset,
    MetricSet,
    RankingResult,
)

#: Evaluation dimensionalities the meta-evaluation pipeline understands. ``2d``
#: is the per-slice path that ships today; ``3d`` is the volumetric path tracked
#: in ``TODO/backlog_sim2rank_3d_eval.md`` and is intentionally *not* built — it
#: fails loud rather than silently degrading to 2-D (CLAUDE.md pitfall #15, the
#: ``SIM2RANK_MODE=3d`` silent-no-op incident that produced two byte-identical
#: "2-D" bundles).
EVAL_MODES = ("2d", "3d")

logger = logging.getLogger(__name__)

#: |Spearman rho| between two rankers' score vectors above which they are flagged as
#: redundant voters. The aggregate is an equal-weight vote, so two near-clone rankers
#: give their shared axis ~2x the weight of an independent one and the "N independent
#: rankers concur" claim quietly becomes "N-1 rankers and an echo" (issue #256).
RANKER_REDUNDANCY_RHO = 0.95


def _inter_ranker_spearman(
    results: list[RankingResult],
) -> dict[str, object]:
    """Inter-ranker Spearman matrix over the shared metric pool.

    The ranker POOL was never screened the way the metric pool is: ``screen_pool`` /
    ``clone_threshold`` de-clone the *metrics*, but nothing ever looked at whether two
    *rankers* were the same statistic twice. They were — ``Sim2Rank`` is exactly
    ``0.5·minmax(ADR) + 0.3·minmax(SCVR) + 0.2·minmax(CDRS)`` and sat in the default
    pool alongside all three (reproduced to max abs error 0.00e+00). The registry now
    drops that fusion, and this matrix is the standing guard that keeps any *future*
    near-clone visible instead of silently doubling an axis's vote.
    """
    from scipy.stats import spearmanr

    methods = [r.method for r in results]
    shared = set.intersection(*(set(r.scores) for r in results)) if results else set()
    shared_names = sorted(shared)
    matrix: dict[str, dict[str, float]] = {m: {} for m in methods}
    flagged: list[dict[str, object]] = []

    if len(shared_names) < 3 or len(results) < 2:
        return {
            "methods": methods,
            "n_shared_metrics": len(shared_names),
            "spearman": matrix,
            "flagged_pairs": flagged,
            "max_abs_offdiagonal": float("nan"),
            "threshold": RANKER_REDUNDANCY_RHO,
        }

    vectors = {
        r.method: [r.scores[n] if math.isfinite(r.scores[n]) else math.nan for n in shared_names]
        for r in results
    }
    max_off = 0.0
    for a in methods:
        for b in methods:
            if a == b:
                matrix[a][b] = 1.0
                continue
            rho = spearmanr(vectors[a], vectors[b], nan_policy="omit").statistic
            rho = float(rho) if rho == rho else float("nan")  # NaN-safe
            matrix[a][b] = rho
            if math.isfinite(rho):
                max_off = max(max_off, abs(rho))
                if abs(rho) >= RANKER_REDUNDANCY_RHO and a < b:
                    flagged.append({"a": a, "b": b, "rho": rho})

    if flagged:
        logger.warning(
            "Ranker-pool redundancy: %d pair(s) at |rho| >= %.2f — %s. These are "
            "not independent voters; the aggregate is over-weighting their shared "
            "axis (issue #256).",
            len(flagged),
            RANKER_REDUNDANCY_RHO,
            ", ".join(f"{p['a']}~{p['b']} ({p['rho']:+.3f})" for p in flagged),
        )

    return {
        "methods": methods,
        "n_shared_metrics": len(shared_names),
        "spearman": matrix,
        "flagged_pairs": flagged,
        "max_abs_offdiagonal": max_off,
        "threshold": RANKER_REDUNDANCY_RHO,
    }


def resolve_eval_mode(mode: str) -> str:
    """Validate the meta-evaluation dimensionality, failing loud on ``3d``.

    The one owner of this contract for ``spectramr meta-evaluate`` and the sim2rank
    meta-evaluation paths. ``2d`` is returned unchanged; ``3d`` raises
    :class:`NotImplementedError` (the volumetric engine is deferred); any other
    value raises :class:`ValueError`.
    """
    if mode == "2d":
        return mode
    if mode == "3d":
        raise NotImplementedError(
            "eval_mode='3d' (volumetric meta-evaluation) is not implemented. "
            "The 3-D engine path is not yet implemented. "
            "Set eval_mode='2d' (or omit it) to use the per-slice path."
        )
    raise ValueError(
        f"eval_mode={mode!r} is not a recognised evaluation mode. "
        f"Allowed values: {', '.join(EVAL_MODES)}."
    )


def _bounded_scores(scores: dict[str, float]) -> dict[str, float]:
    """Linearly rescale finite scores into ``[-1, 1]`` (non-finite passed through).

    The powered Hoeffding gate (:func:`powered.certify_partial_order`) is calibrated
    for a statistic bounded in ``[-1, 1]`` (the rank-correlation scale), but rankers
    emit raw statistics of arbitrary, method-dependent scale. Applying the gate to,
    say, scores in ``[10, 50]`` makes every gap dwarf the threshold and certifies
    everything (out of the bound's regime — critique A3). A linear min-max rescale to
    ``[-1, 1]`` preserves the ordering AND the relative gap structure while making the
    range assumption hold, so the powered certificate is valid. Non-finite scores
    (crashed metrics) pass through unchanged for ``certify_partial_order``'s own guard.
    """
    finite = [v for v in scores.values() if math.isfinite(v)]
    if not finite:
        return dict(scores)
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-12:
        return {k: (0.0 if math.isfinite(v) else v) for k, v in scores.items()}
    return {
        k: (2.0 * (v - lo) / (hi - lo) - 1.0) if math.isfinite(v) else v for k, v in scores.items()
    }


@dataclass
class MetaEvaluationPipeline:
    rankers: list[BaseRanker]
    simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    aggregator_config: AggregatorConfig = field(default_factory=AggregatorConfig)
    # Per-pair misordering budget for the powered certification gate (§3 repair).
    powered_alpha: float = 0.05
    # Pool hygiene (§8 repair): when True, clones/broken metrics are screened out
    # of the voting pool before ranking. Off by default (changes the voting set).
    screen_pool_enabled: bool = False
    clone_threshold: float = 0.95
    signal_threshold: float = 0.1
    # L3⁺ — cross-axis aggregator: "z" (default Pareto-only) or "kemeny"
    # (Condorcet-consistent, Lean ``kemeny_optimal_ranks_pareto_winner_first``).
    aggregator: str = "z"
    kemeny_config: KemenyConfig = field(default_factory=KemenyConfig)
    # L2⁺ — also emit a variance-adaptive, anytime-valid betting certification
    # alongside the (Hoeffding) powered gate. Off by default; the default run is
    # the "real" powered/Hoeffding implementation.
    use_betting: bool = False
    betting_alpha: float = 0.05
    # Evaluation dimensionality (pitfall #15: every exposed knob must be wired,
    # validated, and stamped). "2d" is the per-slice path that ships; "3d" is
    # validated then *raises* (see :func:`resolve_eval_mode`). Stamped into
    # :meth:`MetaEvaluationOutput.to_summary` so every bundle self-describes its
    # dimensionality — the gap the SIM2RANK_MODE=3d incident exposed.
    eval_mode: str = "2d"

    @classmethod
    def from_defaults(cls) -> MetaEvaluationPipeline:
        return cls(
            rankers=[
                CDSCRRanker(),
                LGDRRanker(),
                SFARanker(),
                ESDRanker(),
                FSPDRanker(),
                Sim2RankRanker(),
            ],
        )

    @classmethod
    def from_registry(
        cls,
        *,
        have_likert: bool = False,
        have_task_net: bool = False,
        ranker_kwargs: Mapping[str, Mapping[str, object]] | None = None,
        **kwargs: Any,
    ) -> MetaEvaluationPipeline:
        """Build a pipeline from every registered ranker whose deps are satisfied.

        The wired entry point of the unified ranking module: it resolves
        :func:`available_rankers` (all generations) and instantiates each. Rankers
        needing absent external data (Likert / task-net) are omitted explicitly,
        never silently degraded.

        ``ranker_kwargs`` is how the external ground truth actually gets in
        (issue #258). Before it existed, ``**kwargs`` went to the *pipeline*
        dataclass rather than to ``spec.instantiate()``, so
        ``from_registry(have_task_net=True)`` cheerfully built
        ``DCMSRanker(task_performance=None)`` and ``run()`` then died with
        "inject via from_registry override" — an override that did not exist. The
        capability gate could express **omit** or **crash**, never **supply**, which
        made the whole Gen-4 core-pipeline path dead code (and is precisely why the
        fabricated-ground-truth bug #253 was never caught by the gate that exists to
        catch it). The gate is now live: a ranker resolved as available whose
        declared ``data_kwargs`` were not injected raises **here**, at construction.

        Args:
            have_likert: Radiologist Likert observations are available.
            have_task_net: A frozen downstream task network's measured performance
                is available.
            ranker_kwargs: ``{ranker_name: {kwarg: value}}`` passed through to each
                ranker's ``__init__`` — e.g.
                ``{"dcms": {"task_performance": p}, "hibsl": {"observations": obs}}``.
            **kwargs: Pipeline-level fields (``simulator_config``, ``aggregator``, …).

        Raises:
            ValueError: A capability flag is set but the corresponding data was not
                injected via ``ranker_kwargs``.
        """
        from .rankers import available_rankers, check_injected_data

        rk: dict[str, Mapping[str, object]] = dict(ranker_kwargs or {})
        specs = available_rankers(have_likert=have_likert, have_task_net=have_task_net)

        unknown = set(rk) - {s.name for s in specs}
        if unknown:
            raise ValueError(
                f"ranker_kwargs names {sorted(unknown)} that are not in the resolved "
                f"pool {sorted(s.name for s in specs)}. A typo here silently does "
                "nothing — the data would never reach a ranker."
            )

        errors = check_injected_data(specs, rk)
        if errors:
            raise ValueError(
                "Capability-gated ranker(s) resolved as available but their external "
                "ground truth was never injected:\n  - " + "\n  - ".join(errors)
            )

        return cls(
            rankers=[spec.instantiate(**rk.get(spec.name, {})) for spec in specs],
            **kwargs,
        )

    def run(
        self,
        metric_set: MetricSet,
        clean_volumes: Iterable[tuple[str, torch.Tensor]],
        assets_by_content: Mapping[str, MetricContext] | None = None,
    ) -> MetaEvaluationOutput:
        # Fail loud on an unimplemented dimensionality BEFORE any simulator /
        # metric work (pitfall #15). For "3d" this raises NotImplementedError.
        eval_mode = resolve_eval_mode(self.eval_mode)
        samples = run_simulator(
            clean_volumes, self.simulator_config, assets_by_content=assets_by_content
        )
        if not samples:
            raise ValueError("simulator produced no samples — empty input?")
        metric_values = precompute_metric_values(samples, metric_set.metrics)
        dataset = MetricEvaluationDataset(samples=samples, metric_values=metric_values)

        # Pool hygiene: restrict voting to the curated (de-cloned, non-broken) pool.
        screening: PoolScreening | None = None
        ranking_metric_set = metric_set
        if self.screen_pool_enabled:
            screening = screen_pool(
                dataset,
                clone_threshold=self.clone_threshold,
                signal_threshold=self.signal_threshold,
            )
            voting = set(screening.voting_metrics)
            ranking_metric_set = MetricSet(
                metrics={k: v for k, v in metric_set.metrics.items() if k in voting},
                higher_is_better={
                    k: v for k, v in metric_set.higher_is_better.items() if k in voting
                },
                differentiable={k: v for k, v in metric_set.differentiable.items() if k in voting},
            )

        per_method_results: list[RankingResult] = []
        for ranker in self.rankers:
            per_method_results.append(ranker.rank(ranking_metric_set, dataset))

        # Ranker-pool hygiene (issue #256): the metric pool has always been
        # de-cloned; the RANKER pool never was. Measure and report it, so a
        # redundant voter can never again inflate an axis's weight in silence.
        ranker_redundancy = _inter_ranker_spearman(per_method_results)

        # Cross-axis aggregation. The default "z" combiner is Pareto-consistent
        # only; "kemeny" is the Condorcet-consistent L3⁺ combiner (Lean
        # ``kemeny_optimal_ranks_pareto_winner_first``).
        if self.aggregator == "kemeny":
            aggregate_result = kemeny_consensus(per_method_results, self.kemeny_config)
        elif self.aggregator == "z":
            aggregate_result = aggregate(per_method_results, self.aggregator_config)
        else:
            raise ValueError(f"unknown aggregator {self.aggregator!r}; expected 'z' or 'kemeny'")
        # Powered certification: which per-axis pairwise verdicts are statistically
        # supported at N = dataset.n_samples (Thm 5.1(3) / Lean powered_sample_size).
        # Rescale each method's raw scores to the [-1, 1] bounded scale the powered
        # Hoeffding gate assumes before certifying (critique A3).
        powered = {
            r.method: certify_partial_order(
                _bounded_scores(r.scores), dataset.n_samples, self.powered_alpha
            )
            for r in per_method_results
        }

        # L2⁺ betting certification (off unless use_betting). Shared with the
        # legacy ``scripts/sim2rank/sim2rank.py --novel-meta`` path via
        # :meth:`compute_betting` so both entry points use identical math.
        betting = self.compute_betting(dataset)

        return MetaEvaluationOutput(
            dataset=dataset,
            per_method=per_method_results,
            aggregate=aggregate_result,
            powered=powered,
            screening=screening,
            betting=betting,
            eval_mode=eval_mode,
            ranker_redundancy=ranker_redundancy,
            degradation_provenance={
                # Which degradation SURFACE produced this run. The core
                # ``DEGRADATION_LIBRARY`` and an injected physics ``operator_library``
                # implement overlapping families with DIFFERENT severity→parameter
                # maps (keep-fraction vs acceleration R, image-sigma vs k-space
                # SNR-dB), so two bundles are only comparable when this matches
                # (critique C2). Stamped so the bundle self-describes.
                "library": (
                    "injected" if self.simulator_config.operator_library is not None else "core"
                ),
                "families": sorted({s.degradation_family for s in dataset.samples}),
                "n_severities_per_family": self.simulator_config.n_severities_per_family,
                "seed": self.simulator_config.seed,
            },
        )

    def compute_betting(
        self,
        dataset: MetricEvaluationDataset,
    ) -> dict[str, BettingPartialOrder]:
        """Variance-adaptive, anytime-valid betting partial order (Lean Sim2Rank.Betting).

        Returns ``{}`` when ``use_betting`` is off. Otherwise certifies a partial
        order over the metric pool from each metric's **severity-trajectory
        concordance increments** — the bounded ``[-1, 1]`` sequence the betting
        confidence sequence and Lean's Kendall U-statistic actually consume
        (:func:`betting.kendall_concordance_increments`). For every
        ``(content, degradation_family)`` sub-sweep the samples are ordered by
        severity ``theta`` and the adjacent-pair concordance signs are computed,
        then concatenated across sub-sweeps into one long increment sequence per
        metric. All metrics share an identical increment length and per-position
        alignment (the per-sample index, *not* the ranker-method index), so the
        paired difference in :func:`certify_betting_order` is a genuine sequential,
        anytime-valid statistic.

        This replaces the earlier ``sign(z - median z)`` proxy, which indexed the
        "increments" by *ranker method* (~6 fixed values, no temporal order,
        median-centred): that had no filtration, so the anytime-valid / early-stop
        guarantee was void (a length-6 vector cannot be a confidence *sequence*).
        Reused by the CLI ``meta-evaluate`` pipeline and the legacy novel-meta
        entry point so the certification is identical regardless of caller.
        """
        if not self.use_betting:
            return {}
        per_metric_inc = self._severity_concordance_increments(dataset)
        return {"aggregate": certify_betting_order(per_metric_inc, self.betting_alpha)}

    @staticmethod
    def _severity_concordance_increments(
        dataset: MetricEvaluationDataset,
    ) -> dict[str, list[float]]:
        """Per-metric ``[-1, 1]`` concordance increments along each severity sub-sweep.

        Groups sample indices by ``(content_id, degradation_family)``, orders each
        group by severity ``theta``, and concatenates the adjacent-pair concordance
        signs of every metric's value trajectory. A crashed (NaN) value yields a
        ``0.0`` (neutral) increment at that step rather than dropping it — so every
        metric keeps the *same* increment length and position alignment, which the
        paired confidence sequence relies on.
        """
        groups: dict[tuple[str, str], list[int]] = {}
        for idx, s in enumerate(dataset.samples):
            groups.setdefault((s.content_id, s.degradation_family), []).append(idx)
        ordered_groups: list[list[int]] = [
            sorted(idxs, key=lambda i: dataset.samples[i].severity.get("theta", 0.0))
            for idxs in groups.values()
        ]
        per_metric_inc: dict[str, list[float]] = {}
        for name, values in dataset.metric_values.items():
            inc: list[float] = []
            for idxs in ordered_groups:
                if len(idxs) < 2:
                    continue
                vals = [values[i] for i in idxs]
                sev = [dataset.samples[i].severity.get("theta", 0.0) for i in idxs]
                inc.extend(kendall_concordance_increments(vals, sev))
            per_metric_inc[name] = inc
        return per_metric_inc

    def transfer_certificates(
        self,
        output: MetaEvaluationOutput,
        real_metric_values: dict[str, list[float]],
        *,
        n_bins: int | None = None,
        debias: bool = True,
    ) -> list[TransferCertificate]:
        """L4 sim-to-real ranking-transfer certificates for a completed run.

        ``real_metric_values`` is a held-out *real* cohort's per-metric scalar values
        (same metric names as the simulated run). The simulated/real law TV is the
        **max** of two lower bounds on the joint-law TV the transfer theorem needs:
        (a) the per-metric *marginal* score-law TV (:func:`transfer.tv_distance_samples`,
        max over metrics), and (b) the *joint*-law TV
        (:func:`transfer.tv_distance_joint_samples`, a 1-NN classifier two-sample test
        on the multivariate metric vectors). The joint estimate is a tighter lower
        bound — it catches a sim→real shift that hides in the dependence structure even
        when every marginal matches — so the certificate is less optimistic. The
        per-pair ``TV < Δ/8`` certificate (Lean ``ranking_transfer``) is issued against
        the aggregate simulated scores. A pair that ``transfers`` keeps its simulated
        ordering under the real law.
        """
        sim_scores = output.aggregate.final_scores
        shared = [m for m in sim_scores if m in real_metric_values]
        tv = 0.0
        usable: list[str] = []
        for m in shared:
            sim_vals = output.dataset.metric_values.get(m)
            real_vals = real_metric_values.get(m)
            if not sim_vals or not real_vals:
                continue
            # crash→NaN contract (simulator.precompute_metric_values): a metric that
            # crashed on its whole cohort is all-NaN and has no estimable sim→real TV.
            # Skip it — otherwise it would (a) crash tv_distance_samples on int(NaN)
            # and (b) if "fixed" to TV=1, dominate this worst-case max-pool and
            # spuriously void every transfer certificate. A partly-NaN metric still
            # contributes its finite-subsample TV (masked inside tv_distance_samples).
            if not any(math.isfinite(v) for v in sim_vals) or not any(
                math.isfinite(v) for v in real_vals
            ):
                continue
            tv = max(
                tv,
                tv_distance_samples(sim_vals, real_vals, n_bins=n_bins, debias=debias),
            )
            usable.append(m)

        # Joint-law TV: the multivariate two-sample test catches a dependence-
        # structure shift the per-metric marginals miss (the marginal TV under-
        # estimates the joint TV the theorem needs). Take the max — the tightest
        # available lower bound on the true joint TV.
        if usable:
            n_real = min(len(real_metric_values[m]) for m in usable)
            sim_matrix = [
                [output.dataset.metric_values[m][i] for m in usable]
                for i in range(output.dataset.n_samples)
            ]
            real_matrix = [[real_metric_values[m][i] for m in usable] for i in range(n_real)]
            # The joint estimator is an enhancement; never abort the certificate,
            # but the fallback must be AUDITABLE (#298). Silently dropping the
            # joint TV leaves only the marginal-max, which under-estimates the
            # joint TV and so certifies MORE pairs than intended — an optimistic
            # certificate. Log the failure named rather than swallow it.
            try:
                tv = max(
                    tv,
                    tv_distance_joint_samples(sim_matrix, real_matrix, debias=debias),
                )
            except Exception as exc:
                logger.warning(
                    "transfer certificate: joint-TV estimator failed (%s: %s); "
                    "falling back to the marginal-max TV=%.4f, which may be "
                    "optimistic (certifies more pairs than the joint bound). #298",
                    type(exc).__name__,
                    exc,
                    tv,
                )
        return tv_transfer_certificate(sim_scores, tv)


@dataclass
class MetaEvaluationOutput:
    dataset: MetricEvaluationDataset
    per_method: list[RankingResult]
    aggregate: AggregateRankingResult
    # method -> powered partial order at N = n_samples (empty for older callers).
    powered: dict[str, PoweredPartialOrder] = field(default_factory=dict)
    # Pool-hygiene screening, when screen_pool_enabled (None otherwise).
    screening: PoolScreening | None = None
    # L2⁺ betting (anytime-valid) certification, when use_betting (empty otherwise).
    betting: dict[str, BettingPartialOrder] = field(default_factory=dict)
    # Evaluation dimensionality this run was produced under ("2d" today). Stamped
    # into the summary so the bundle is self-describing (pitfall #15).
    eval_mode: str = "2d"
    # Inter-ranker Spearman matrix + flagged near-clone pairs (issue #256). Empty
    # for callers that construct the output directly.
    ranker_redundancy: dict[str, object] = field(default_factory=dict)
    # Which degradation surface / families produced the run (critique C2). Empty
    # for callers that construct the output directly without provenance.
    degradation_provenance: dict[str, object] = field(default_factory=dict)

    def to_summary(self) -> dict[str, object]:
        return {
            "eval_mode": self.eval_mode,
            "ranker_redundancy": self.ranker_redundancy,
            "degradation_provenance": self.degradation_provenance,
            "final_ranks": self.aggregate.final_ranks,
            "final_scores": self.aggregate.final_scores,
            "defensibility_flags": self.aggregate.defensibility_flags,
            "defensibility_semantics": self.aggregate.defensibility_semantics,
            "weights": self.aggregate.weights,
            "per_method_z": self.aggregate.per_method_z,
            "per_method_ranks": {r.method: r.ranks for r in self.per_method},
            "per_method_scores": {r.method: r.scores for r in self.per_method},
            "n_samples": self.dataset.n_samples,
            "powered": {
                method: {
                    "gap_threshold": po.gap_threshold,
                    "n_certified": po.n_certified,
                    "n_ambiguous": po.n_ambiguous,
                    "certified": po.certified,
                    "ambiguous": po.ambiguous,
                }
                for method, po in self.powered.items()
            },
            "calibration": (
                {
                    "null_fpr": self.aggregate.calibration.null_fpr,
                    "pvalues": self.aggregate.calibration.pvalues,
                    "calibrated_flags": self.aggregate.calibration.calibrated_flags,
                    "n_perm": self.aggregate.calibration.n_perm,
                    "alpha": self.aggregate.calibration.alpha,
                }
                if self.aggregate.calibration is not None
                else None
            ),
            "screening": (
                {
                    "voting_metrics": self.screening.voting_metrics,
                    "clone_clusters": self.screening.clone_clusters,
                    "redundant": self.screening.redundant,
                    "broken": self.screening.broken,
                }
                if self.screening is not None
                else None
            ),
            "betting": {
                key: {
                    "alpha": bo.alpha,
                    "n_certified": bo.n_certified,
                    "n_ambiguous": bo.n_ambiguous,
                    "certified": bo.certified,
                    "ambiguous": bo.ambiguous,
                }
                for key, bo in self.betting.items()
            },
        }
