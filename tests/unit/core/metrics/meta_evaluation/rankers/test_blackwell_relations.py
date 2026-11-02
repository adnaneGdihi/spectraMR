"""The opt-in relation-catalogue Blackwell ranker (multi-coil default), with a planted violator."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.coil_catalogue import coil_relation_catalogue
from spectramr.core.metrics.meta_evaluation.rankers import (
    BlackwellConfig,
    BlackwellRelationRanker,
    available_rankers,
    get_ranker,
)
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
)

pytestmark = pytest.mark.unit
SHAPE = (24, 24)
FAST = BlackwellConfig(n_draws=2, envelope_widths=(0.0, 1.0))


def _dataset() -> MetricEvaluationDataset:
    g = torch.Generator().manual_seed(0)
    samples = []
    for k in range(3):
        img = torch.zeros(SHAPE)
        img[6:18, 6:18] = 1.0
        img = img + 0.2 * torch.rand(SHAPE, generator=g)
        samples.append(DegradationSample(img[None], img[None], "none", {}, f"c{k}", 0))
    return MetricEvaluationDataset(samples, {})


def _small(contents):
    return coil_relation_catalogue(
        contents, accelerations=(2, 4), densities=(0.6, 0.3), partial_fractions=()
    )


def _smoothness(d: torch.Tensor, c: torch.Tensor) -> float:
    x = d[0]
    return -float((x[1:] - x[:-1]).abs().mean() + (x[:, 1:] - x[:, :-1]).abs().mean())


def test_registered_opt_in_and_never_pooled() -> None:
    spec = get_ranker("blackwell_relations")
    assert spec.opt_in and spec.generation == 5
    assert "blackwell_relations" not in {s.name for s in available_rankers()}


def test_planted_violator_is_flagged_and_relations_are_stamped() -> None:
    ms = MetricSet(
        metrics={"mse": lambda d, c: float((d - c).pow(2).mean()), "smooth": _smoothness},
        higher_is_better={"mse": False, "smooth": True},
    )
    res = BlackwellRelationRanker(FAST, catalogue_factory=_small).report(ms, _dataset())
    per = res.diagnostics["per_metric"]
    assert per["smooth"]["bvr_raw"] > per["mse"]["bvr_raw"]
    assert res.scores["smooth"] < res.scores["mse"]
    rel = res.diagnostics["relations"]
    assert rel["n_strict"] > 0 and "n_uncertified" in rel
    assert res.diagnostics["catalogue"]["catalogue"] == "multicoil_cartesian"


def test_nuisance_catalogue_flags_motion_sensitivity_but_not_an_energy_metric() -> None:
    """Planted: image energy is invariant under any unitary T (||T x + n|| ~ ||x + n||),
    so an energy error respects every absorption relation, while squared error does not."""
    from spectramr.infrastructure.physics.degradations.nuisance_catalogue import (
        nuisance_catalogue,
    )

    def energy(d: torch.Tensor, c: torch.Tensor) -> float:
        return abs(float(d.pow(2).mean() - c.pow(2).mean()))

    ms = MetricSet(
        metrics={"mse": lambda d, c: float((d - c).pow(2).mean()), "energy": energy},
        higher_is_better={"mse": False, "energy": False},
    )

    def motion(contents):
        return nuisance_catalogue(contents, codes=("translation_drift", "sudden_motion"))

    res = BlackwellRelationRanker(FAST, catalogue_factory=motion).report(ms, _dataset())
    per = res.diagnostics["per_metric"]
    assert per["mse"]["bvr_raw"] > 0.1
    assert per["energy"]["bvr_raw"] == 0.0


def test_relation_catalogues_reject_the_oracle_wiener_envelope() -> None:
    ms = MetricSet(metrics={"mse": lambda d, c: float((d - c).pow(2).mean())})
    cfg = BlackwellConfig(n_draws=2, envelope="oracle_wiener")
    with pytest.raises(ValueError, match="diagonal"):
        BlackwellRelationRanker(cfg, catalogue_factory=_small).report(ms, _dataset())
