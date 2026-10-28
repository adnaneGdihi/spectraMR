"""``campaign infer``: which arms are graded, on what, and what is refused."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.orchestration import campaign_inference as ci
from spectramr.infrastructure.orchestration.campaign_state import CampaignState, ExperimentStatus


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    arms = []
    for stem, ckpt in (
        ("arm_a", "best.safetensors"),
        ("arm_b", "step_00000010.safetensors"),
        ("arm_c", None),
    ):
        cfg = tmp_path / f"{stem}.yaml"
        cfg.write_text("x: 1\n")
        run = tmp_path / stem
        (run / "checkpoints").mkdir(parents=True)
        (run / "test_metrics.json").write_text('{"manifest": "data/manifests/test.json"}')
        if ckpt:
            (run / "checkpoints" / ckpt).write_bytes(b"x")
        arms.append(
            ExperimentStatus(
                name=stem,
                config_path=str(cfg),
                output_dir=f"/cluster/elsewhere/{stem}",
                status="completed",
            )
        )
    CampaignState(campaign_name="c", campaign_dir=str(tmp_path), experiments=arms).save()
    return tmp_path


def _plans(campaign: Path, **kw):
    return {p.name: p for p in ci.plan_inference(campaign, campaign, **kw)}


def test_best_is_preferred_then_latest_and_no_checkpoint_is_refused_with_a_reason(campaign):
    plans = _plans(campaign)
    assert (plans["arm_a"].rule, Path(plans["arm_a"].checkpoint).name) == (
        "best",
        "best.safetensors",
    )
    assert plans["arm_b"].rule == "latest"
    assert "no checkpoint" in plans["arm_c"].skip_reason


def test_arms_are_located_under_the_given_dir_not_the_recorded_cluster_path(campaign):
    assert _plans(campaign)["arm_a"].run_dir == str(campaign / "arm_a")


def test_an_existing_result_needs_force(campaign):
    (campaign / "arm_a" / "inference").mkdir()
    (campaign / "arm_a" / "inference" / "final_eval.json").write_text("{}")
    assert "--force" in _plans(campaign)["arm_a"].skip_reason
    assert _plans(campaign, force=True)["arm_a"].skip_reason is None


def test_only_filters_and_an_unknown_name_raises(campaign):
    assert list(_plans(campaign, only=["arm_b"])) == ["arm_b"]
    with pytest.raises(ValueError, match="no arm"):
        ci.plan_inference(campaign, campaign, only=["nope"])


def test_a_missing_config_is_a_skip_not_a_shorter_array(campaign):
    state = CampaignState.load(campaign / "campaign_state.json")
    state.experiments[0].config_path = "/gone/arm_a.yaml"
    state.save()
    assert "config not found" in _plans(campaign)["arm_a"].skip_reason


def test_submit_freezes_aligned_lists_and_asks_the_array_for_inference_mode(campaign, monkeypatch):
    seen = {}
    snap = campaign / "snap.json"
    snap.write_text("{}")
    monkeypatch.setattr(ci, "write_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(
        ci,
        "_campaign_config",
        lambda *a: SimpleNamespace(
            slurm_defaults=SimpleNamespace(model_dump=lambda: {"gpus": 1}), array_concurrency=4
        ),
    )
    monkeypatch.setattr(
        ci.SLURMBackend,
        "generate_array_job_script",
        staticmethod(lambda **kw: seen.update(kw) or "#!/bin/bash\n"),
    )
    plans = [p for p in ci.plan_inference(campaign, campaign) if p.skip_reason is None]
    assert (
        ci.submit_inference(
            plans, campaign, campaign, dry_run=True, slurm_overrides={"time": "02:00:00"}
        )
        is None
    )
    manifest, ckpts = (
        seen["manifest_text"].splitlines(),
        seen["inference_checkpoints_text"].splitlines(),
    )
    assert len(manifest) == len(ckpts) == seen["n_tasks"] == 2
    assert not (campaign / ci.MANIFEST).exists()  # a dry run writes nothing
    assert ckpts[0].endswith("best.safetensors") and seen["inference_checkpoints"].endswith(
        ci.CHECKPOINTS
    )
    assert seen["slurm_params"] == {"gpus": 1, "time": "02:00:00"} and seen["concurrency"] == 4


def test_nothing_gradable_raises(campaign):
    with pytest.raises(ValueError, match="nothing to submit"):
        ci.submit_inference([], campaign, campaign, dry_run=True)


def test_a_run_that_scored_its_final_weights_is_graded_on_the_latest_step(campaign):
    run = campaign / "arm_a"
    (run / "checkpoints" / "step_00000020.safetensors").write_bytes(b"x")
    (run / "test_metrics.json").write_text('{"weights": "final", "manifest": "t.json"}')
    plan = _plans(campaign)["arm_a"]
    assert Path(plan.checkpoint).name == "step_00000020.safetensors" and "final" in plan.rule


def test_an_output_dir_that_is_not_the_config_stem_is_refused(campaign):
    state = CampaignState.load(campaign / "campaign_state.json")
    state.experiments[0].config_path = str(campaign / "arm_b.yaml")
    state.save()
    assert "config stem" in _plans(campaign)["arm_a"].skip_reason


def _set_status(campaign, status):
    state = CampaignState.load(campaign / "campaign_state.json")
    state.experiments[0].status = status
    state.save()


def test_a_stale_submitted_arm_that_finished_is_graded(campaign):
    _set_status(campaign, "submitted")
    (campaign / "arm_a" / "run_summary.json").write_text('{"success": true}')
    assert _plans(campaign)["arm_a"].skip_reason is None


def test_a_submitted_arm_with_no_finished_run_is_not_graded(campaign):
    _set_status(campaign, "running")
    assert "run_summary.json is missing" in _plans(campaign)["arm_a"].skip_reason
    (campaign / "arm_a" / "run_summary.json").write_text(
        '{"success": true, "wall_clock_yield": true}'
    )
    assert "does not report a finished run" in _plans(campaign)["arm_a"].skip_reason
    (campaign / "arm_a" / "run_summary.json").write_text('{"success": false}')
    assert "does not report a finished run" in _plans(campaign)["arm_a"].skip_reason


def test_a_failed_arm_is_not_graded_whatever_is_on_disk(campaign):
    _set_status(campaign, "failed")
    (campaign / "arm_a" / "run_summary.json").write_text('{"success": true}')
    assert "not completed" in _plans(campaign)["arm_a"].skip_reason


def test_an_arm_whose_training_run_scored_no_test_split_is_not_graded(campaign):
    (campaign / "arm_a" / "test_metrics.json").unlink()
    assert "scored no test split" in _plans(campaign)["arm_a"].skip_reason


def test_the_plan_names_the_test_manifest_the_training_pass_scored(campaign):
    assert _plans(campaign)["arm_a"].test_manifest == "data/manifests/test.json"
