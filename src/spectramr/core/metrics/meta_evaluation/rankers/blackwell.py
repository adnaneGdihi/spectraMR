"""Gen-5 Blackwell consistency ranker: does a metric respect the information order?

On a Fourier-diagonal catalogue (noise, blur and truncation before noise), Blackwell's
order is the elementwise order of the per-coefficient information. A metric violates a
certified relation ``a > b`` when it prefers the less informative experiment. Raw
violations mix information with the way the data are displayed (Theorem C1.3). The
display-canonicalised metric ``q#`` takes the best Gaussian post-filter, which removes the
display term (Theorem C1.5).

Outputs per metric:

* **score**: the concordance of the metric, as displayed, with the certified strict
  relations: the share of matched (content, draw) pairs, across all relations, on which
  it orders the two experiments as the information order does (ties count one half).
  Draw ``r`` shares its noise across experiments, so the pair is matched, and a strictly
  monotone change of the metric leaves the score unchanged (PSNR and RMSE score alike).
  The same concordance of ``q#`` is reported as ``concordance_sharp``. It saturates at 1
  for standard metrics, which is the cycle-6 finding that their failures are display
  regret, so it cannot carry the ranking.
* **gate**: raw and canonicalised violation rates, and mixture-probe violations
  (Theorem C1.8), each with an e-BH-certified count. Sign e-values test a majority
  statement, and with few contents they certify little, so the counts sit beside the
  point rates rather than inside the score.
* **coherence**: the revealed-design LP (Theorem C1.7) on preferences screened by a
  paired t-rule, and the conditioning index against the oracle-LSI control. The
  preferences are not simultaneously certified, so an infeasible system is a diagnostic,
  not a falsification.

The ranker is ``opt_in``: it never joins a default pool. It calls the metrics itself on
its own catalogue, so its cost does not scale with the dataset's sample count. It reads
only the dataset's clean references (one per ``content_id``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..blackwell import (
    DiagonalExperiment,
    build_catalogue,
    certify,
    group_profiles,
    info_groups,
    metric_table,
    mixture,
    oracle_lsi_control,
    preferences_by_t,
    revealed_design_lp,
    split_preferences,
)
from ..blackwell.certificate import incoherence_certificate
from ..blackwell.evaluate import wiener_envelope
from ..blackwell.gate import (
    MIN_FINITE_FRACTION,
    certifiable,
    probe_violations,
    relation_gate,
)
from ..blackwell.inputs import metric_contexts, no_signal, read_contents, split_metrics
from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .registry import register_ranker

logger = logging.getLogger(__name__)

#: Post-filter classes for the canonicalised rate. ``oracle_wiener`` needs diagonal
#: experiments (see :func:`~..blackwell.evaluate.wiener_envelope`).
ENVELOPES = ("gaussian_blur", "oracle_wiener")

_INCOHERENCE_NOTE = (
    "coherence preferences are screened by a paired t-rule, not simultaneously "
    "certified: an infeasible system is a diagnostic, not a falsification. Preferences "
    "that split an equivalent pair or reverse a certified relation are left out (they "
    "are the equivalence and violation rates)"
)


@dataclass(frozen=True)
class BlackwellConfig:
    n_draws: int = 4
    max_contents: int = 16
    envelope_widths: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    max_probes: int = 60
    q_fdr: float = 0.1
    preference_t: float = 3.0
    max_groups: int = 2048
    seed: int = 0
    envelope: str = "gaussian_blur"
    wiener_lambdas: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        if self.n_draws < 2 or self.max_contents < 2 or self.max_probes < 0:
            raise ValueError("need n_draws >= 2, max_contents >= 2, max_probes >= 0")
        if not 0.0 < self.q_fdr < 1.0:
            raise ValueError(f"q_fdr must be in (0, 1), got {self.q_fdr}")
        if not self.envelope_widths or self.envelope_widths[0] != 0.0:
            raise ValueError("envelope_widths must start with 0.0 (the identity display)")
        if self.envelope not in ENVELOPES:
            raise ValueError(f"envelope must be one of {ENVELOPES}, got {self.envelope!r}")
        if self.preference_t <= 0 or self.max_groups < 1:
            raise ValueError("preference_t must be > 0 and max_groups >= 1")

    def provenance(self) -> dict[str, object]:
        return asdict(self)


#: Builds the catalogue for the run's contents: ``contents -> (experiments, provenance)``.
CatalogueFactory = Callable[
    [Sequence[torch.Tensor]], tuple[list[DiagonalExperiment], dict[str, object]]
]


def radial_catalogue(
    contents: Sequence[torch.Tensor],
) -> tuple[list[DiagonalExperiment], dict[str, object]]:
    """The default analytic five-family radial catalogue at the contents' shape."""
    shape = (int(contents[0].shape[-2]), int(contents[0].shape[-1]))
    return build_catalogue(shape), {"catalogue": "radial_5x5"}


@register_ranker(
    "blackwell",
    generation=5,
    opt_in=True,
    description="Blackwell-Loewner consistency of metrics with the certified information order",
)
class BlackwellConsistencyRanker(BaseRanker):
    method_name = "blackwell"

    def __init__(
        self,
        config: BlackwellConfig | None = None,
        catalogue_factory: CatalogueFactory = radial_catalogue,
    ) -> None:
        self.config = config or BlackwellConfig()
        self.catalogue_factory = catalogue_factory

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        """:meth:`report`, rejected by the degeneracy guard when the scores tie."""
        result = self.report(metric_set, dataset)
        assert_not_degenerate(list(result.scores.values()), name=self.method_name)
        return result

    def report(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        """Scores and the full gate, without the degeneracy guard.

        Standard metrics often tie at a concordance of 1 on this catalogue. The gate
        diagnostics are still the result, so a caller that reports them can check the
        tie itself rather than lose them to the guard.
        """
        cfg = self.config
        contents, contents_info = read_contents(dataset, cfg.max_contents)
        exps, catalogue_provenance = self.catalogue_factory(contents)
        graph = certify(exps)
        cross = [(i, j) for i, j in graph.incomparable if exps[i].family != exps[j].family]
        pick = np.unique(np.linspace(0, len(cross) - 1, min(cfg.max_probes, len(cross))).round())
        probe_pairs = [cross[int(k)] for k in pick] if cross else []
        probes = [mixture(exps[i], exps[j]) for i, j in probe_pairs]

        names, complex_only = split_metrics(metric_set)
        metrics = {m: metric_set.metrics[m] for m in names}
        orient = {m: 1.0 if metric_set.is_higher_better(m) else -1.0 for m in names}
        ctx = metric_contexts(names, contents)
        raw, sharp = metric_table(
            metrics,
            orient,
            contents,
            exps,
            n_draws=cfg.n_draws,
            envelope_widths=cfg.envelope_widths,
            seed=cfg.seed,
            contexts=ctx,
            envelope_fn=(
                wiener_envelope(cfg.wiener_lambdas) if cfg.envelope == "oracle_wiener" else None
            ),
        )
        probe_raw = (
            metric_table(
                metrics,
                orient,
                contents,
                probes,
                n_draws=cfg.n_draws,
                envelope_widths=(0.0,),
                seed=cfg.seed,
                contexts=ctx,
            )[0]
            if probes
            else {}
        )

        profiles, sizes, exact = group_profiles(exps, info_groups(exps), cfg.max_groups)
        control_prefs = preferences_by_t(oracle_lsi_control(contents, exps), cfg.preference_t)
        control = revealed_design_lp(profiles, sizes, control_prefs, exact=exact)

        scores: dict[str, float] = {}
        per_metric: dict[str, dict[str, object]] = {}
        for m in complex_only:
            scores[m] = float("nan")
            per_metric[m] = no_signal("skipped: requires complex input", None)
        for m in names:
            finite = float(torch.isfinite(raw[m]).double().mean())
            if finite < MIN_FINITE_FRACTION:
                logger.warning("blackwell: %r is finite on %.0f%% of calls", m, 100 * finite)
                scores[m] = float("nan")
                per_metric[m] = no_signal("no signal: NaN", finite)
                continue
            scores[m], relation_fields = relation_gate(raw[m], sharp[m], graph, cfg.q_fdr)
            mean_raw = raw[m].mean(-1)
            probe_point, probe_cert = None, None
            if probes:
                pr = probe_raw[m].mean(-1)
                probe_point, probe_cert = probe_violations(
                    [(mean_raw[:, i] - pr[:, p]).numpy() for p, (i, _) in enumerate(probe_pairs)],
                    [(mean_raw[:, j] - pr[:, p]).numpy() for p, (_, j) in enumerate(probe_pairs)],
                    cfg.q_fdr,
                )
            prefs, n_split_eq, n_reversed = split_preferences(
                preferences_by_t(mean_raw, cfg.preference_t), graph
            )
            coh = revealed_design_lp(profiles, sizes, prefs, exact=exact)
            cert = (
                incoherence_certificate(profiles, prefs, [e.name for e in exps], exact=exact)
                if coh.status == "infeasible"
                else None
            )
            kappa = None
            if coh.n_preferences and coh.mass is not None and control.mass:
                kappa = coh.mass / control.mass
            per_metric[m] = {
                "status": "ok",
                "finite_fraction": finite,
                **relation_fields,
                "probe_violation_rate": probe_point,
                "probe_violations_certified": probe_cert,
                "coherent": coh.coherent if coh.n_preferences else None,
                "coherence_status": coh.status,
                "n_preferences": coh.n_preferences,
                "preferences_set_aside": {"equivalent": n_split_eq, "reversed": n_reversed},
                "incoherence_certificate": cert,
                "kappa": kappa,
            }

        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "per_metric": per_metric,
                "relations": graph.summary(exps),
                "n_contents": len(contents),
                "contents": contents_info,
                "n_probes": len(probes),
                # False means a certified count of 0 is "could not certify", not "none".
                "relations_certifiable": certifiable(len(contents), len(graph.strict), cfg.q_fdr),
                "probes_certifiable": certifiable(len(contents), 2 * len(probes), cfg.q_fdr),
                "coherence_exact": exact,
                "control_coherent": control.coherent,
                "coherence_note": _INCOHERENCE_NOTE,
                "config": cfg.provenance(),
                "catalogue": catalogue_provenance,
            },
        )


__all__ = ["BlackwellConfig", "BlackwellConsistencyRanker", "CatalogueFactory", "radial_catalogue"]
