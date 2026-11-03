"""Tests for MetaEvaluationPipeline.run asset threading.

Focus: ``run(..., assets_by_content=...)`` forwards the real acquisition assets
down to ``run_simulator`` (which attaches them per content). Kept light — the
ranker/aggregation stack is exercised elsewhere; here we only assert the seam.
"""

from __future__ import annotations

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.meta_evaluation import pipeline as pipeline_mod
from spectramr.core.metrics.meta_evaluation.pipeline import MetaEvaluationPipeline
from spectramr.core.metrics.meta_evaluation.types import MetricSet


def test_run_forwards_assets_by_content_to_simulator(monkeypatch):
    captured = {}

    def _fake_run_simulator(clean_volumes, config, assets_by_content=None, subject_of=None):
        captured["assets_by_content"] = assets_by_content
        raise RuntimeError("short-circuit after capturing the seam")

    monkeypatch.setattr(pipeline_mod, "run_simulator", _fake_run_simulator)

    pipe = MetaEvaluationPipeline.from_defaults()
    metric_set = MetricSet(metrics={"noop": lambda p, t, **k: 0.0})
    assets = {"subj0": MetricContext(coil_maps=torch.ones(1, 4, 8, 8))}

    try:
        pipe.run(
            metric_set, [("subj0", torch.zeros(1, 8, 8))], assets_by_content=assets
        )
    except RuntimeError:
        pass

    assert captured["assets_by_content"] is assets


def _falling_psnr_dataset(n_contents: int = 30, n_levels: int = 6):
    from spectramr.core.metrics.meta_evaluation.types import (
        DegradationSample,
        MetricEvaluationDataset,
    )

    img = torch.zeros(1, 4, 4)
    samples, psnr, blind = [], [], []
    for c in range(n_contents):
        for t in range(n_levels):
            samples.append(
                DegradationSample(
                    clean=img,
                    degraded=img,
                    degradation_family="gaussian_noise",
                    severity={"theta": t / n_levels},
                    content_id=f"c{c}",
                    seed=0,
                )
            )
            psnr.append(40.0 - 3.0 * t)  # falls correctly with severity
            blind.append(1.0)  # does not respond at all
    return MetricEvaluationDataset(samples, {"psnr": psnr, "blind_lower": blind})


class TestBettingOrientation:
    """A correctly falling higher-is-better metric must outrank a blind one."""

    def test_falling_psnr_is_certified_above_a_blind_metric(self):
        pipe = MetaEvaluationPipeline.from_defaults()
        pipe.use_betting = True
        order = pipe.compute_betting(
            _falling_psnr_dataset(), {"psnr": True, "blind_lower": False}
        )["aggregate"]
        assert order.is_certified("psnr", "blind_lower")

    def test_directions_default_to_the_registry_resolver(self):
        import pytest

        pipe = MetaEvaluationPipeline.from_defaults()
        pipe.use_betting = True
        with pytest.raises(Exception, match="blind_lower"):
            pipe.compute_betting(_falling_psnr_dataset())

    def test_a_metric_without_a_direction_raises(self):
        import pytest

        pipe = MetaEvaluationPipeline.from_defaults()
        pipe.use_betting = True
        with pytest.raises(ValueError, match="blind_lower"):
            pipe.compute_betting(_falling_psnr_dataset(), {"psnr": True})


class TestPoweredEffectiveN:
    """Slices of one subject are one independent unit for the powered gate."""

    def test_five_slices_of_one_subject_give_n_eff_one(self):
        from spectramr.core.metrics.meta_evaluation.pipeline import effective_sample_size

        ds = _falling_psnr_dataset(n_contents=5, n_levels=4)
        for s in ds.samples:
            s.subject_id = "subj0"
        assert effective_sample_size(ds) == (1, "subject")

    def test_without_subjects_n_eff_counts_contents(self):
        from spectramr.core.metrics.meta_evaluation.pipeline import effective_sample_size

        assert effective_sample_size(_falling_psnr_dataset(n_contents=5)) == (5, "content")

    def test_run_certifies_at_n_eff_with_holm_and_stamps_it(self, monkeypatch):
        import torch as _t

        pipe = MetaEvaluationPipeline.from_defaults()
        pipe.simulator_config.families = ["gaussian_noise"]
        pipe.simulator_config.n_severities_per_family = 4
        captured = {}
        real = pipeline_mod.certify_partial_order

        def _spy(scores, n, alpha, **kw):
            captured.setdefault("calls", []).append((n, kw))
            return real(scores, n, alpha, **kw)

        monkeypatch.setattr(pipeline_mod, "certify_partial_order", _spy)
        ms = MetricSet(
            metrics={
                "mse": lambda p, t, **k: float(((p - t) ** 2).mean()),
                "mae": lambda p, t, **k: float((p - t).abs().mean()),
            },
            higher_is_better={"mse": False, "mae": False},
        )
        vols = [(f"s0#{i}", _t.rand(1, 16, 16)) for i in range(5)]
        out = pipe.run(ms, vols, subject_of={c: "s0" for c, _ in vols})
        assert captured["calls"]
        assert {(n, kw.get("correction")) for n, kw in captured["calls"]} == {(1, "holm")}
        summary = out.to_summary()
        assert (summary["n_eff"], summary["n_eff_unit"]) == (1, "subject")
        assert all(p["n_certified"] == 0 for p in summary["powered"].values())


def test_severity_provenance_stamped_only_when_set():
    from spectramr.core.metrics.meta_evaluation.pipeline import MetaEvaluationPipeline
    from spectramr.core.metrics.meta_evaluation.simulator import SimulatorConfig

    pipe = MetaEvaluationPipeline.from_defaults()
    assert pipe._severity_provenance() == {}
    pipe.simulator_config = SimulatorConfig(
        damage_yardsticks=("rel_l2",), severity_provenance={"gaussian_noise": {"start": 1.0}}
    )
    assert pipe._severity_provenance() == {
        "severity_axes": {"gaussian_noise": {"start": 1.0}},
        "damage_yardsticks": ["rel_l2"],
    }
