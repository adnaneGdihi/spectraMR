"""The opt-in Blackwell consistency ranker, with a planted information-order violator."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.rankers import (
    BlackwellConfig,
    BlackwellConsistencyRanker,
    available_rankers,
    get_ranker,
)
from spectramr.core.metrics.meta_evaluation.ranking_guards import DegenerateRankingError
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
    MetricSet,
)

pytestmark = pytest.mark.unit
SHAPE = (24, 24)
FAST = BlackwellConfig(n_draws=2, max_probes=4, envelope_widths=(0.0, 1.0))


def _dataset(n: int = 3, shape: tuple[int, ...] = SHAPE) -> MetricEvaluationDataset:
    g = torch.Generator().manual_seed(0)
    samples = []
    for k in range(n):
        img = torch.zeros(shape)
        img[..., 6:18, 6:18] = 1.0
        img = img + 0.2 * torch.rand(shape, generator=g)
        samples.append(DegradationSample(img[None], img[None], "none", {}, f"c{k}", 0))
    return MetricEvaluationDataset(samples, {})


def _mse(d: torch.Tensor, c: torch.Tensor) -> float:
    return float((d - c).pow(2).mean())


def _smoothness(d: torch.Tensor, c: torch.Tensor) -> float:
    """Planted violator: rewards smooth images, so it prefers blur to information."""
    x = d[0]
    return -float((x[1:] - x[:-1]).abs().mean() + (x[:, 1:] - x[:, :-1]).abs().mean())


def _metric_set() -> MetricSet:
    return MetricSet(
        metrics={
            "mse": _mse,
            "smooth": _smoothness,
            "mae": lambda d, c: float((d - c).abs().mean()),
        },
        higher_is_better={"mse": False, "smooth": True, "mae": False},
    )


def test_ranker_is_registered_opt_in_and_never_pooled() -> None:
    spec = get_ranker("blackwell")
    assert spec.opt_in and spec.generation == 5
    assert "blackwell" not in {s.name for s in available_rankers()}
    assert "blackwell" not in {
        s.name for s in available_rankers(have_likert=True, have_task_net=True)
    }


def test_planted_violator_is_flagged_and_ranked_below_mse() -> None:
    res = BlackwellConsistencyRanker(FAST).report(_metric_set(), _dataset())
    per = res.diagnostics["per_metric"]
    assert per["smooth"]["bvr_raw"] > 0.1
    # Point rates carry Monte Carlo error at two draws (a 3e-6 gap can flip), so MSE is
    # held to no certified violation and to a rate far below the planted violator's.
    assert per["mse"]["bvr_sharp_certified"] == 0
    assert per["mse"]["bvr_sharp"] < 0.05 < per["smooth"]["bvr_raw"] / 2
    assert res.scores["smooth"] < res.scores["mse"]
    assert res.ranks[-1] == "smooth"
    assert res.diagnostics["relations"]["n_strict"] > 0
    assert res.diagnostics["config"]["n_draws"] == 2


def test_always_nan_metric_reports_no_signal_instead_of_a_clean_gate() -> None:
    """Planted: a metric that crashes to NaN everywhere must not pass as perfect."""
    ms = _metric_set()
    ms.metrics["broken"] = lambda d, c: float("nan")
    ms.higher_is_better["broken"] = True
    res = BlackwellConsistencyRanker(FAST).report(ms, _dataset())
    gate = res.diagnostics["per_metric"]["broken"]
    assert gate["status"] == "no signal: NaN" and gate["finite_fraction"] == 0.0
    assert gate["bvr_raw"] is None and gate["coherent"] is None and gate["kappa"] is None
    assert res.scores["broken"] != res.scores["broken"]  # NaN ranks last
    assert res.ranks[-1] == "broken"


def test_certifiability_is_reported_for_small_content_counts() -> None:
    diag = BlackwellConsistencyRanker(FAST).report(_metric_set(), _dataset()).diagnostics
    assert diag["relations_certifiable"] is False  # 3 contents cannot clear e >= m / q
    assert diag["probes_certifiable"] is False


def test_rank_applies_the_degeneracy_guard_that_report_skips() -> None:
    clones = MetricSet(
        metrics=dict.fromkeys(("a", "b", "c"), _mse),
        higher_is_better=dict.fromkeys(("a", "b", "c"), False),
    )
    ranker = BlackwellConsistencyRanker(FAST)
    assert len(set(ranker.report(clones, _dataset()).scores.values())) == 1
    with pytest.raises(DegenerateRankingError):
        ranker.rank(clones, _dataset())


def test_needs_two_non_zero_contents() -> None:
    ranker = BlackwellConsistencyRanker(FAST)
    with pytest.raises(ValueError, match="two non-zero"):
        ranker.report(_metric_set(), _dataset(n=1))


def test_volumes_are_sliced_and_mixed_sizes_cropped() -> None:
    from spectramr.core.metrics.meta_evaluation.blackwell.inputs import read_contents

    g = torch.Generator().manual_seed(0)
    vol = torch.rand((1, 10, 24, 24), generator=g) + 0.1
    small = torch.rand((1, 20, 22), generator=g) + 0.1
    empty = torch.zeros((1, 24, 24))
    ds = MetricEvaluationDataset(
        [
            DegradationSample(vol, vol, "none", {}, "vol", 0),
            DegradationSample(small, small, "none", {}, "small", 0),
            DegradationSample(empty, empty, "none", {}, "empty", 0),
        ],
        {},
    )
    contents, info = read_contents(ds, 16)
    assert len(contents) == 5 and all(tuple(c.shape) == (20, 22) for c in contents)
    assert info["cropped"] is True and info["skipped_zero"] == 1 and info["shape"] == [20, 22]


@pytest.mark.parametrize(
    "kwargs",
    [{"n_draws": 1}, {"q_fdr": 1.0}, {"envelope_widths": (1.0,)}, {"max_contents": 1}],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BlackwellConfig(**kwargs)


def test_injected_catalogue_is_used_and_stamped() -> None:
    from spectramr.core.metrics.meta_evaluation.blackwell import build_catalogue

    def factory(contents):
        cat = build_catalogue(tuple(contents[0].shape))
        # an exact copy of the first experiment makes one equivalence
        return [*cat, cat[0]], {"catalogue": "planted"}

    res = BlackwellConsistencyRanker(FAST, catalogue_factory=factory).report(
        _metric_set(), _dataset()
    )
    assert res.diagnostics["catalogue"] == {"catalogue": "planted"}
    assert res.diagnostics["relations"]["n_equivalent"] == 1
    gate = res.diagnostics["per_metric"]["mse"]
    # identical experiments under common noise draws: no display preference at all
    assert gate["equivalence_display_rate"] == 0.0


def test_oracle_wiener_envelope_is_wired_and_validated() -> None:
    res = BlackwellConsistencyRanker(
        BlackwellConfig(n_draws=2, max_probes=2, envelope="oracle_wiener", wiener_lambdas=(1.0,))
    ).report(_metric_set(), _dataset())
    assert res.diagnostics["config"]["envelope"] == "oracle_wiener"
    # squared error canonicalised by its own optimal LSI filter respects the order
    assert res.diagnostics["per_metric"]["mse"]["bvr_sharp_certified"] == 0
    with pytest.raises(ValueError, match="envelope"):
        BlackwellConfig(envelope="bogus")
