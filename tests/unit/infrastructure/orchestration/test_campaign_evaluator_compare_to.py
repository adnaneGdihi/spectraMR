"""Campaign ``compare_to`` and seed replicates in the evaluator.

A variant that declares ``compare_to`` is tested only against the named arms.
Arms without it keep the default: tested against every baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.config.schemas.campaign import (
    CampaignConfigSchema,
    CampaignExperimentSchema,
    EvaluationConfigSchema,
)
from spectramr.infrastructure.orchestration.campaign_evaluator import CampaignEvaluator
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)


def _exp(name: str, role: str = "variant", **kw) -> CampaignExperimentSchema:
    return CampaignExperimentSchema(name=name, config=f"{name}.yaml", role=role, **kw)


def _metrics(seed: int, n: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {m: rng.normal(size=n) for m in ("psnr", "ssim", "lpips")}


def _evaluate(
    monkeypatch: pytest.MonkeyPatch,
    arms: dict[str, dict[str, np.ndarray]],
    roles: dict[str, str],
    campaign: CampaignConfigSchema | None,
):
    ev = CampaignEvaluator(
        eval_config=EvaluationConfigSchema(per_sample_inference=False, compute_uncertainty=False)
    )
    state = CampaignState(
        campaign_name="c",
        campaign_dir="/tmp/none",
        experiments=[
            ExperimentStatus(name=n, config_path="x.yaml", role=roles[n], status="completed")
            for n in arms
        ],
    )
    monkeypatch.setattr(ev, "_load_all_metrics", lambda _s: arms)
    return ev.evaluate(state, campaign=campaign)


def _pairs(report) -> set[tuple[str, str]]:
    return {(r.experiment_a, r.experiment_b) for r in report.pairwise_results}


def test_compare_to_restricts_the_variant_to_its_named_arm(monkeypatch):
    # B is the baseline. X declares compare_to=[Y], so X must never be paired
    # with B; the only pairing for X is Y vs X.
    campaign = CampaignConfigSchema(
        name="c",
        experiments=[
            _exp("B", role="baseline"),
            _exp("X", compare_to=["Y"]),
            _exp("Y"),
            _exp("Z"),
        ],
    )
    arms = {n: _metrics(i) for i, n in enumerate(["B", "X", "Y", "Z"])}
    roles = {"B": "baseline", "X": "variant", "Y": "variant", "Z": "variant"}
    report = _evaluate(monkeypatch, arms, roles, campaign)
    assert _pairs(report) == {("Y", "X"), ("B", "Y"), ("B", "Z")}


def test_arms_without_compare_to_keep_the_baseline_pairing(monkeypatch):
    campaign = CampaignConfigSchema(
        name="c",
        experiments=[_exp("B", role="baseline"), _exp("X"), _exp("Y")],
    )
    arms = {n: _metrics(i) for i, n in enumerate(["B", "X", "Y"])}
    roles = {"B": "baseline", "X": "variant", "Y": "variant"}
    report = _evaluate(monkeypatch, arms, roles, campaign)
    assert _pairs(report) == {("B", "X"), ("B", "Y")}


def test_without_a_campaign_every_arm_is_paired_with_every_baseline(monkeypatch):
    arms = {n: _metrics(i) for i, n in enumerate(["B", "X"])}
    roles = {"B": "baseline", "X": "variant"}
    report = _evaluate(monkeypatch, arms, roles, None)
    assert _pairs(report) == {("B", "X")}


def test_seed_replicates_pool_into_one_arm(monkeypatch):
    # Two seeds of the baseline are one arm, so X is paired once with it.
    campaign = CampaignConfigSchema(
        name="c",
        experiments=[
            _exp("B_seed42", role="baseline", tags={"seed": "42"}),
            _exp("B_seed43", role="baseline", tags={"seed": "43"}),
            _exp("X"),
        ],
    )
    arms = {
        "B_seed42": _metrics(1),
        "B_seed43": _metrics(2),
        "X": _metrics(3),
    }
    roles = {"B_seed42": "baseline", "B_seed43": "baseline", "X": "variant"}
    report = _evaluate(monkeypatch, arms, roles, campaign)
    assert _pairs(report) == {("B", "X")}
    assert report.leaderboard is not None
    assert set(report.leaderboard["experiment"]) == {"B", "X"}


def test_compare_to_must_name_a_declared_arm():
    with pytest.raises(ValueError, match="unknown experiments"):
        CampaignConfigSchema(
            name="c",
            experiments=[_exp("B", role="baseline"), _exp("X", compare_to=["nope"])],
        )
