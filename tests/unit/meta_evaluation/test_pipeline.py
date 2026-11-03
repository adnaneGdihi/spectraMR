"""End-to-end pipeline + aggregator behavior."""

from __future__ import annotations

import torch

from spectramr.core.metrics.meta_evaluation import (
    AggregatorConfig,
    MetaEvaluationPipeline,
    MetricSet,
    SimulatorConfig,
    aggregate,
)
from spectramr.core.metrics.meta_evaluation.rankers import CDSCRRanker, FSPDRanker


def _mse(p, t):
    return float(((p.float() - t.float()) ** 2).mean().item())


def _const(p, t):
    return 0.0


def test_aggregator_combines_z_scores() -> None:
    from spectramr.core.metrics.meta_evaluation.types import RankingResult

    r1 = RankingResult(
        method="CDSCR",
        scores={"mse": 0.9, "ssim": 0.4, "noise": 0.0},
        ranks=["mse", "ssim", "noise"],
    )
    r2 = RankingResult(
        method="LGDR",
        scores={"mse": 0.7, "ssim": 0.5, "noise": 0.1},
        ranks=["mse", "ssim", "noise"],
    )
    out = aggregate(
        [r1, r2],
        AggregatorConfig(
            weights={"CDSCR": 1.0, "LGDR": 1.0}, min_positive_axes=2
        ),
    )
    # MSE has highest z on both axes -> top of final ranks.
    assert out.final_ranks[0] == "mse"
    # noise should have negative z on both axes; defensibility flag should be False.
    assert out.defensibility_flags["noise"] is False


def _phantom(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij"
    )
    r = torch.sqrt(xx**2 + yy**2)
    return (torch.exp(-r**2 / 0.5) + 0.05 * torch.randn((32, 32), generator=gen)).unsqueeze(0).float()


def test_full_pipeline_smoke() -> None:
    metric_set = MetricSet(
        metrics={"mse": _mse, "constant": _const},
        higher_is_better={"mse": False, "constant": False},
        differentiable={"mse": True, "constant": False},
    )
    clean = [("a", _phantom(0)), ("b", _phantom(1))]
    # Use just two rankers for speed.
    pipeline = MetaEvaluationPipeline(
        rankers=[CDSCRRanker(), FSPDRanker()],
        simulator_config=SimulatorConfig(
            families=["gaussian_noise", "gaussian_blur"],
            n_severities_per_family=3,
            seed=1,
        ),
        aggregator_config=AggregatorConfig(
            weights={"CDSCR": 1.0, "FSPD": 1.0}, min_positive_axes=2
        ),
    )
    output = pipeline.run(metric_set, clean)
    summary = output.to_summary()
    assert "mse" in summary["final_scores"]
    assert summary["final_ranks"][0] == "mse"
    assert isinstance(summary["defensibility_flags"], dict)
    # Powered certification gate is attached per method.
    assert set(output.powered.keys()) == {"CDSCR", "FSPD"}
    for po in output.powered.values():
        # Certified at the independent contents, not the correlated samples.
        assert po.n_samples == summary["n_eff"] == len({s.content_id for s in output.dataset.samples})
        assert po.correction == "holm"
        # Every pair is either certified or ambiguous, never both.
        assert po.n_certified + po.n_ambiguous == 1  # C(2, 2) over {mse, constant}
    assert "powered" in summary
    assert summary["degradation_provenance"]["seed_scheme"] == "blake2b_v1"
    assert set(summary["powered"]) == {"CDSCR", "FSPD"}
