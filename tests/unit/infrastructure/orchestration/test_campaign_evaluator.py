"""Unit tests for the pure-Python data layer of
:mod:`spectramr.infrastructure.orchestration.campaign_evaluator`.

The evaluator's statistical paths (Wilcoxon, bootstrap CIs, multiple-
comparisons correction) need a synthetic per-sample-metrics fixture
and are deferred to a follow-up. This file pins only the dataclasses
and their dict/JSON serialisation contract, which is the entry point
consumed by the CLI:

* ``PairwiseResult`` exposes ``p_value`` and ``is_significant``
  derived properties.
* ``UncertaintyReport`` accepts a name-only constructor.
* ``CampaignReport.to_dict()`` produces a JSON-serialisable payload.
* ``CampaignReport.save()`` writes the JSON to disk via atomic open.
* ``CampaignEvaluator.evaluate()`` returns a "no evaluable experiments"
  report when the campaign state has fewer than two completed arms.

No statistical or numeric assertions are made here — they belong to
the slow / integration lane.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch as _torch
import yaml as _yaml

from spectramr.infrastructure.orchestration.campaign_evaluator import (
    CampaignEvaluator,
    CampaignReport,
    PairwiseResult,
    UncertaintyReport,
)
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)
from spectramr.models.registry import register_model as _register_model
from tests.fixtures.run_state_files import write_run_state

# ── PairwiseResult ─────────────────────────────────────────────────


class TestPairwiseResult:
    def test_default_p_value_is_nan(self) -> None:
        """``paired_ttest`` dict empty → ``p_value`` is NaN, not zero."""
        r = PairwiseResult(experiment_a="a", experiment_b="b", metric="psnr")
        assert math.isnan(r.p_value)
        # NaN compared with < 0.05 is False — not significant.
        assert r.is_significant is False

    def test_p_value_read_from_ttest_dict(self) -> None:
        r = PairwiseResult(
            experiment_a="a",
            experiment_b="b",
            metric="psnr",
            paired_ttest={"p_value": 0.03, "t_stat": 2.5},
        )
        assert r.p_value == 0.03
        assert r.is_significant is True

    def test_significance_boundary_excludes_005(self) -> None:
        """``is_significant`` is strict less-than 0.05 — exactly 0.05 is NOT significant."""
        r = PairwiseResult(
            experiment_a="a",
            experiment_b="b",
            metric="psnr",
            paired_ttest={"p_value": 0.05},
        )
        # Strict inequality.
        assert r.is_significant is False

    def test_significance_just_under_005_significant(self) -> None:
        r = PairwiseResult(
            experiment_a="a",
            experiment_b="b",
            metric="psnr",
            paired_ttest={"p_value": 0.04999},
        )
        assert r.is_significant is True

    def test_default_factories_do_not_leak(self) -> None:
        """Two PairwiseResults share no mutable state — field(default_factory)
        guarantees a fresh dict per instance."""
        r1 = PairwiseResult(experiment_a="a", experiment_b="b", metric="psnr")
        r2 = PairwiseResult(experiment_a="c", experiment_b="d", metric="ssim")
        r1.paired_ttest["p_value"] = 0.01
        # r2's dict must still be empty.
        assert r2.paired_ttest == {}
        assert r2.wilcoxon == {}
        assert r2.effect_size == {}
        assert r2.bootstrap_ci == {}


# ── UncertaintyReport ──────────────────────────────────────────────


class TestUncertaintyReport:
    def test_name_only_constructor(self) -> None:
        ur = UncertaintyReport(experiment_name="alpha")
        assert ur.experiment_name == "alpha"
        # Defaults are None — caller must check before use.
        assert ur.ece is None
        assert ur.sharpness is None
        assert ur.uncertainty_error_correlation is None

    def test_constructor_with_all_metrics(self) -> None:
        ur = UncertaintyReport(
            experiment_name="alpha",
            ece=0.03,
            sharpness=0.85,
            uncertainty_error_correlation=0.4,
        )
        assert ur.ece == 0.03
        assert ur.sharpness == 0.85
        assert ur.uncertainty_error_correlation == 0.4


# ── CampaignReport ─────────────────────────────────────────────────


class TestCampaignReport:
    def test_default_factories_no_leak(self) -> None:
        a = CampaignReport(campaign_name="A")
        b = CampaignReport(campaign_name="B")
        a.pairwise_results.append(PairwiseResult(experiment_a="x", experiment_b="y", metric="psnr"))
        # Each instance owns its own list.
        assert b.pairwise_results == []
        assert b.significance_matrix == {}
        assert b.corrected_p_values == {}
        assert b.uncertainty_reports == []

    def test_to_dict_returns_json_serialisable(self) -> None:
        rep = CampaignReport(campaign_name="alpha")
        rep.pairwise_results.append(
            PairwiseResult(
                experiment_a="a",
                experiment_b="b",
                metric="psnr",
                paired_ttest={"p_value": 0.01, "t_stat": 3.2},
                effect_size={"cohens_d": 0.8},
            )
        )
        rep.uncertainty_reports.append(UncertaintyReport(experiment_name="a", ece=0.05))
        rep.summary = "two arms compared"

        out = rep.to_dict()
        # Top-level keys.
        assert out["campaign_name"] == "alpha"
        assert out["summary"] == "two arms compared"
        # Each PairwiseResult is flattened to a dict.
        assert len(out["pairwise_results"]) == 1
        pr_dict = out["pairwise_results"][0]
        assert pr_dict["experiment_a"] == "a"
        assert pr_dict["experiment_b"] == "b"
        assert pr_dict["paired_ttest"]["p_value"] == 0.01
        # Uncertainty rows preserve all fields.
        assert out["uncertainty"][0]["experiment"] == "a"
        assert out["uncertainty"][0]["ece"] == 0.05

        # Whole payload is JSON-serialisable.
        assert json.dumps(out)

    def test_to_dict_handles_empty_pairwise_and_uncertainty(self) -> None:
        rep = CampaignReport(campaign_name="empty")
        out = rep.to_dict()
        assert out["pairwise_results"] == []
        assert out["uncertainty"] == []

    def test_save_writes_json_to_disk(self, tmp_path: Path) -> None:
        rep = CampaignReport(campaign_name="saved")
        out_path = tmp_path / "report.json"
        rep.save(out_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["campaign_name"] == "saved"

    def test_save_creates_missing_parent(self, tmp_path: Path) -> None:
        rep = CampaignReport(campaign_name="nested")
        out_path = tmp_path / "a" / "b" / "report.json"
        rep.save(out_path)
        assert out_path.exists()


# ── Leaderboard rank direction (metric SSOT) ──────────────────────


class TestLeaderboardRankDirection:
    """The leaderboard must rank each metric in its true optimization direction,
    resolved from the metric SSOT — not a hand-rolled lower-better literal set
    that missed rmse / fid / hfen / prefixed names (which then ranked the WORST
    arm at rank 1)."""

    def _leaderboard(self, metric: str, arm_means: dict[str, float]):
        import numpy as np

        ev = CampaignEvaluator()
        all_metrics = {arm: {metric: np.full(3, mean)} for arm, mean in arm_means.items()}
        return ev._build_leaderboard(all_metrics)

    def _rank_of(self, df, arm: str) -> float:
        return float(df.loc[df["experiment"] == arm, "rank"].iloc[0])

    def test_lower_is_better_metric_ranks_smallest_first(self) -> None:
        # rmse (lower is better) was NOT in the old literal set → mis-ranked.
        df = self._leaderboard("rmse", {"good": 0.10, "bad": 0.90})
        assert self._rank_of(df, "good") == 1.0
        assert self._rank_of(df, "bad") == 2.0

    def test_prefixed_lower_is_better_metric(self) -> None:
        # A prefixed column (val_lpips) must still resolve as lower-is-better.
        df = self._leaderboard("val_lpips", {"good": 0.05, "bad": 0.50})
        assert self._rank_of(df, "good") == 1.0

    def test_higher_is_better_metric_ranks_largest_first(self) -> None:
        df = self._leaderboard("psnr", {"good": 38.0, "bad": 22.0})
        assert self._rank_of(df, "good") == 1.0
        assert self._rank_of(df, "bad") == 2.0


# ── CampaignEvaluator — short-circuit branch ───────────────────────


class TestCampaignEvaluatorShortCircuit:
    def test_evaluate_with_zero_evaluable_arms_returns_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no arm has a completed-with-checkpoint state, the
        evaluator returns a placeholder report instead of crashing."""
        ev = CampaignEvaluator()

        # Build a state with only-failed experiments — none evaluable.
        state = CampaignState(
            campaign_name="all_failed",
            campaign_dir=str(tmp_path),
            experiments=[
                ExperimentStatus(name="a", config_path="x.yaml", status="failed"),
                ExperimentStatus(name="b", config_path="x.yaml", status="failed"),
            ],
        )

        # Stub out per-sample inference and metric loading so the
        # evaluator stays CPU-only and never touches torch.
        called: dict[str, int] = {"inference": 0, "metrics": 0}

        def _no_inference(state_: CampaignState) -> None:
            called["inference"] += 1

        def _empty_metrics(state_: CampaignState) -> dict:
            called["metrics"] += 1
            return {}

        monkeypatch.setattr(ev, "_run_inference_all", _no_inference)
        monkeypatch.setattr(ev, "_load_all_metrics", _empty_metrics)

        rep = ev.evaluate(state)
        assert isinstance(rep, CampaignReport)
        assert rep.campaign_name == "all_failed"
        # Summary explains why nothing was compared.
        assert "Insufficient evaluable experiments" in rep.summary
        # Default config has per_sample_inference=True → stubbed inference
        # was called exactly once before the early return.
        assert called["inference"] == 1
        assert called["metrics"] == 1
        # No pairwise comparisons possible.
        assert rep.pairwise_results == []

    def test_evaluate_skips_inference_when_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from spectramr.config.schemas.campaign import EvaluationConfigSchema

        # Disable per_sample_inference; verify _run_inference_all never fires.
        cfg = EvaluationConfigSchema(per_sample_inference=False)
        ev = CampaignEvaluator(eval_config=cfg)

        state = CampaignState(
            campaign_name="no_inf",
            campaign_dir=str(tmp_path),
            experiments=[],
        )

        ran: list[int] = []
        monkeypatch.setattr(
            ev,
            "_run_inference_all",
            lambda s: ran.append(1),  # type: ignore[arg-type]
        )
        monkeypatch.setattr(ev, "_load_all_metrics", lambda s: {})

        ev.evaluate(state)
        assert ran == [], "_run_inference_all must not run when disabled"


class TestCampaignEvaluatorAcceleratedRunContract:
    """Per-sample evaluation is a heavy pipeline (model forward + LPIPS net).

    It used to resolve its device with a bare
    ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``, so a node
    that lost its GPU produced CPU-computed PSNR/SSIM/LPIPS that were then
    tabulated in the campaign leaderboard alongside GPU-run arms.
    """

    def test_evaluator_routes_through_the_ssot(self) -> None:
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "spectramr"
            / "infrastructure"
            / "orchestration"
            / "campaign_evaluator.py"
        ).read_text()
        assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in src
        assert "resolve_torch_device" in src
        assert 'pipeline="evaluate"' in src


class TestResolveMetricColumn:
    """``_resolve_metric_column`` -- the fix for the silent wrong-metric bug (#1062).

    The old rule was ``[c for c in df.columns if mn in c.lower()]`` then ``cols[0]``:
    unanchored substring, resolved by DataFrame column order. These tests pin the
    three properties that failure violated -- it must not pick a training column, it
    must not depend on column order, and it must refuse rather than guess.
    """

    RESOLVE = staticmethod(CampaignEvaluator._resolve_metric_column)

    # The exact CSV from the issue's reproduction. Tuple, not list: RUF012 forbids a
    # mutable class attribute, and immutability is right here anyway — several tests
    # permute these columns and must not be able to disturb the shared fixture.
    COLUMNS = ("iteration", "train_ssim", "val_ms_ssim", "val_ssim", "val_psnr")

    def test_the_reported_bug_ssim_no_longer_resolves_to_the_training_column(self):
        """The headline regression: 'ssim' used to pick 'train_ssim'."""
        col, _ = self.RESOLVE(list(self.COLUMNS), "ssim")
        assert col == "val_ssim"

    def test_ssim_does_not_resolve_to_the_longer_ms_ssim_metric(self):
        """'ssim' is a substring of 'ms_ssim'; exact 'val_ssim' must win."""
        col, _ = self.RESOLVE(list(self.COLUMNS), "ssim")
        assert col != "val_ms_ssim"

    def test_ms_ssim_still_resolves_to_its_own_column(self):
        col, _ = self.RESOLVE(list(self.COLUMNS), "ms_ssim")
        assert col == "val_ms_ssim"

    def test_resolution_is_independent_of_column_order(self):
        """The old code's answer changed with DataFrame column order."""
        import itertools

        answers = {
            self.RESOLVE(list(perm), "ssim")[0] for perm in itertools.permutations(self.COLUMNS)
        }
        assert answers == {"val_ssim"}

    def test_bare_metric_names_resolve_when_there_is_no_val_prefix(self):
        """MetricsTracker writes keys verbatim into a validation-specific FILE, so a
        validation CSV legitimately carries bare column names."""
        col, _ = self.RESOLVE(["step", "psnr", "ssim"], "ssim")
        assert col == "ssim"

    def test_val_prefixed_column_is_preferred_over_the_bare_one(self):
        col, reason = self.RESOLVE(["ssim", "val_ssim"], "ssim")
        assert col == "val_ssim"
        assert "exact" in reason

    def test_a_training_only_match_is_refused_with_a_reason(self):
        """Campaign metrics are held-out by definition."""
        col, reason = self.RESOLVE(["step", "train_ssim"], "ssim")
        assert col is None
        assert "TRAINING" in reason

    def test_ambiguity_is_refused_rather_than_resolved_by_picking_one(self):
        """A wrong metric that looks right is worse than a missing column."""
        col, reason = self.RESOLVE(["val_psnr_4x", "val_psnr_8x"], "psnr")
        assert col is None
        assert "ambiguous" in reason

    def test_a_unique_decorated_column_is_still_reachable(self):
        """An exact-only rule would have silently dropped these from every campaign
        that reports per-acceleration metrics."""
        col, _ = self.RESOLVE(["step", "val_psnr_4x"], "psnr")
        assert col == "val_psnr_4x"

    def test_a_missing_metric_reports_why(self):
        col, reason = self.RESOLVE(["step", "val_psnr"], "lpips")
        assert col is None
        assert "no column" in reason

    def test_matching_is_case_insensitive(self):
        col, _ = self.RESOLVE(["Step", "VAL_SSIM"], "ssim")
        assert col == "VAL_SSIM"


# ── checkpoint loading and model reconstruction ────────────────────────────
#
# `campaign evaluate` rebuilds each arm's generator from its YAML and loads the
# run's checkpoint into it. Both halves were wrong and both failed quietly:
#
#   * the envelope was read as `model_state_dict` then `model` -- a fourth
#     private vocabulary that misses the `generator` key every GAN checkpoint is
#     written with, so the evaluator returned None on ordinary runs (#1310);
#   * the model was built via `create_generator(**model_kwargs)`, so the
#     *evaluated* architecture was not the *trained* one -- no
#     `acceleration_config`, no `kspace_log_scaled`, no data-consistency block --
#     and channel widths fell back to a hardcoded 2.
#
# The whole point of a leaderboard is that the arms on it are comparable, so
# these are load-bearing, not cosmetic.



@_register_model("_campaign_eval_witness", "reconstruction")
class _CampaignEvalWitness(_torch.nn.Module):
    """Records the SSOT kwargs the evaluator constructed it with."""

    seen: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        acceleration_config: Any = None,
        kspace_log_scaled: Any = None,
        use_dc: Any = None,
        **kwargs: Any,
    ):
        super().__init__()
        type(self).seen = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "acceleration_config": acceleration_config,
            "kspace_log_scaled": kspace_log_scaled,
            "use_dc": use_dc,
        }
        self.conv = _torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self.conv(x)


def _witness_arm_yaml(tmp_path: Path) -> str:
    """A minimal arm declaring the three SSOT blocks the evaluator dropped."""
    cfg = {
        "config_version": "1.0",
        "model": {
            "model_type": "_campaign_eval_witness",
            "in_channels": 3,
            "out_channels": 3,
        },
        "data": {
            "sampling": {"patch_size": [16, 16]},
            "loader": {"batch_size": 1},
            "processing": {"enable_log_scaling": True},
        },
        "undersampling": {"enabled": True},
        "physics": {"data_consistency": {"enabled": True, "method": "soft"}},
        "optimization": {},
        "logging": {},
    }
    path = tmp_path / "arm.yaml"
    path.write_text(_yaml.safe_dump(cfg))
    return str(path)


class TestCampaignEvaluatorRebuildsTheTrainedModel:
    """The evaluated architecture must be the trained architecture."""

    @staticmethod
    def _reconstruct(tmp_path: Path, models: dict[str, Any]):
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        _CampaignEvalWitness.seen = {}
        (tmp_path / "ckpt").mkdir(exist_ok=True)
        path = write_run_state(tmp_path / "ckpt" / "best.pt", models)
        return CampaignEvaluator._reconstruct_model_from_config(
            _witness_arm_yaml(tmp_path), str(path), _torch.device("cpu")
        )

    def test_the_generator_part_loads(self, tmp_path: Path) -> None:
        """The RunState's generator part is the evaluated model's weights."""
        ref = _CampaignEvalWitness(in_channels=3, out_channels=3)
        model = self._reconstruct(tmp_path, {"generator": ref.state_dict()})
        assert isinstance(model, _torch.nn.Module)
        for k, v in ref.state_dict().items():
            assert _torch.equal(model.state_dict()[k], v), f"{k} did not load"

    def test_the_ssot_blocks_reach_the_rebuilt_model(self, tmp_path: Path) -> None:
        """The arm declares undersampling, log scaling and data consistency."""
        ref = _CampaignEvalWitness(in_channels=3, out_channels=3)
        self._reconstruct(tmp_path, {"generator": ref.state_dict()})
        seen = _CampaignEvalWitness.seen
        assert seen.get("acceleration_config") is not None, (
            "the evaluated model was built without the arm's acceleration "
            "block — it is not the model that produced the checkpoint"
        )
        assert seen.get("kspace_log_scaled") is True
        assert seen.get("use_dc") is True

    def test_declared_channel_widths_are_not_defaulted(self, tmp_path: Path) -> None:
        """Sensitivity pair: the arm declares 3, the old code hardcoded 2."""
        ref = _CampaignEvalWitness(in_channels=3, out_channels=3)
        self._reconstruct(tmp_path, {"generator": ref.state_dict()})
        assert _CampaignEvalWitness.seen.get("in_channels") == 3
        assert _CampaignEvalWitness.seen.get("out_channels") == 3

    def test_a_checkpoint_that_shares_nothing_raises(self, tmp_path: Path) -> None:
        """The facade this closes: ``load_state_dict(strict=False)`` against a
        foreign checkpoint loads *nothing* and reports success, so the arm is
        scored on random initial weights (pitfalls #9/#16)."""
        with pytest.raises(ValueError, match="shares zero parameter names"):
            self._reconstruct(tmp_path, {"generator": {"totally.unrelated": _torch.zeros(1)}})

    def test_a_pre_v1_checkpoint_names_the_converter(self, tmp_path: Path) -> None:
        """A director envelope from before schema 1 is refused, not guessed at."""
        from spectramr.infrastructure.checkpointing.payload import SchemaVersionError
        from spectramr.infrastructure.orchestration.campaign_evaluator import (
            CampaignEvaluator,
        )

        ref = _CampaignEvalWitness(in_channels=3, out_channels=3)
        legacy = tmp_path / "best.pt"
        _torch.save({"generator": ref.state_dict(), "global_step": 7}, legacy)
        with pytest.raises(SchemaVersionError, match="convert_checkpoints_v1"):
            CampaignEvaluator._reconstruct_model_from_config(
                _witness_arm_yaml(tmp_path), str(legacy), _torch.device("cpu")
            )

    def test_the_runs_resolved_config_wins_over_the_yaml(self, tmp_path: Path, monkeypatch) -> None:
        """An edited YAML is reported, and the run's own artifact rebuilds the model."""
        import spectramr.infrastructure.orchestration.campaign_evaluator as ce
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.validation.resolved_config_artifact import (
            write_resolved_config,
        )

        seen: dict = {}
        real = ce.resolve_inference_settings

        def spy(config_path, checkpoint_path, **kw):
            settings, source = real(config_path, checkpoint_path, **kw)
            seen.update(checkpoint=checkpoint_path, source=source)
            return settings, source

        monkeypatch.setattr(ce, "resolve_inference_settings", spy)
        trained = _witness_arm_yaml(tmp_path)
        run = tmp_path / "ckpt"
        run.mkdir()
        write_resolved_config(run, TrainingSettings.from_yaml(trained), run_id="t")
        doc = _yaml.safe_load(Path(trained).read_text())
        doc["optimization"] = {"optimizer": {"learning_rate": 0.31337}}
        edited = tmp_path / "edited.yaml"
        edited.write_text(_yaml.safe_dump(doc))
        ref = _CampaignEvalWitness(in_channels=3, out_channels=3)
        path = write_run_state(run / "best.pt", {"generator": ref.state_dict()})

        ce.CampaignEvaluator._reconstruct_model_from_config(
            str(edited), str(path), _torch.device("cpu")
        )
        assert seen["checkpoint"] == str(path)
        assert seen["source"]["kind"] == "resolved_config"
        assert "optimization" in seen["source"]["diverging_blocks"]
