"""Gen-5 Blackwell consistency ranker for catalogues that certify their own relations.

The companion of :mod:`.blackwell` for experiments without a diagonal information
profile. A factory returns a :class:`~..blackwell.relations.RelationCatalogue`:
multi-coil masks with witnessed relations (the default, built in core), or nuisance
mixtures certified by unitary absorption (built in infrastructure and injected). The
gate is the shared :func:`~..blackwell.gate.relation_gate`: concordance, raw and
canonicalised violations, and equivalences. Mixture probes and the coherence LP are
left out, because they need diagonal profiles.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import torch

from ..blackwell.coil_catalogue import coil_relation_catalogue
from ..blackwell.evaluate import metric_table
from ..blackwell.gate import MIN_FINITE_FRACTION, certifiable, relation_gate
from ..blackwell.inputs import metric_contexts, no_signal, read_contents, split_metrics
from ..blackwell.relations import RelationCatalogue
from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .blackwell import BlackwellConfig
from .registry import register_ranker

logger = logging.getLogger(__name__)

#: ``contents -> RelationCatalogue``.
RelationCatalogueFactory = Callable[[Sequence[torch.Tensor]], RelationCatalogue]


@register_ranker(
    "blackwell_relations",
    generation=5,
    opt_in=True,
    description="Blackwell consistency on catalogues with their own certified relations",
)
class BlackwellRelationRanker(BaseRanker):
    method_name = "blackwell_relations"

    def __init__(
        self,
        config: BlackwellConfig | None = None,
        catalogue_factory: RelationCatalogueFactory = coil_relation_catalogue,
    ) -> None:
        self.config = config or BlackwellConfig()
        self.catalogue_factory = catalogue_factory

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        """:meth:`report`, rejected by the degeneracy guard when the scores tie."""
        result = self.report(metric_set, dataset)
        assert_not_degenerate(list(result.scores.values()), name=self.method_name)
        return result

    def report(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        """Scores and the relation gate, without the degeneracy guard."""
        cfg = self.config
        if cfg.envelope != "gaussian_blur":
            raise ValueError(
                f"envelope {cfg.envelope!r} needs diagonal experiments; the relation "
                "catalogues support gaussian_blur only"
            )
        contents, contents_info = read_contents(dataset, cfg.max_contents)
        catalogue = self.catalogue_factory(contents)
        exps, graph = catalogue.experiments, catalogue.graph

        names, complex_only = split_metrics(metric_set)
        orient = {m: 1.0 if metric_set.is_higher_better(m) else -1.0 for m in names}
        raw, sharp = metric_table(
            {m: metric_set.metrics[m] for m in names},
            orient,
            contents,
            exps,
            n_draws=cfg.n_draws,
            envelope_widths=cfg.envelope_widths,
            seed=cfg.seed,
            contexts=metric_contexts(names, contents),
            realise=catalogue.realiser(cfg.n_draws, cfg.seed),
            noise_dims=catalogue.noise_dims,
        )

        scores: dict[str, float] = {}
        per_metric: dict[str, dict[str, object]] = {}
        for m in complex_only:
            scores[m] = float("nan")
            per_metric[m] = no_signal("skipped: requires complex input", None)
        for m in names:
            finite = float(torch.isfinite(raw[m]).double().mean())
            if finite < MIN_FINITE_FRACTION:
                logger.warning("blackwell_relations: %r finite on %.0f%%", m, 100 * finite)
                scores[m] = float("nan")
                per_metric[m] = no_signal("no signal: NaN", finite)
                continue
            scores[m], fields = relation_gate(raw[m], sharp[m], graph, cfg.q_fdr)
            per_metric[m] = {"status": "ok", "finite_fraction": finite, **fields}

        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "per_metric": per_metric,
                "relations": graph.summary(exps),
                "n_contents": len(contents),
                "contents": contents_info,
                "relations_certifiable": certifiable(len(contents), len(graph.strict), cfg.q_fdr),
                "config": cfg.provenance(),
                "catalogue": catalogue.provenance,
            },
        )


__all__ = ["BlackwellRelationRanker", "RelationCatalogueFactory"]
