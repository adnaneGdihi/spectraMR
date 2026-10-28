"""``campaign infer`` is reachable from the parser and from the per-task dispatcher."""

from __future__ import annotations

import pytest

import spectramr.cli.manifest_dispatch as md
from spectramr.cli.inference_dispatch import checkpoint_for_task, inference_verb_args


def test_the_subcommand_is_registered_with_its_flags():
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(
        ["campaign", "infer", "/c", "--only", "a", "--force", "--dry-run"]
    )
    assert (args.campaign_action, args.only, args.force, args.dry_run) == (
        "infer",
        ["a"],
        True,
        True,
    )
    assert callable(args.func)


def test_a_task_reads_its_own_checkpoint_line(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("/a/best.safetensors\n/b/best.safetensors\n")
    assert checkpoint_for_task(f, 1) == "/b/best.safetensors"
    with pytest.raises(IndexError, match="no checkpoint for task 2"):
        checkpoint_for_task(f, 2)


def test_inference_flags_name_the_checkpoint_explicitly():
    assert inference_verb_args("/x/best.safetensors") == [
        "--validate-only",
        "--resume",
        "/x/best.safetensors",
    ]


def test_dispatch_routes_into_the_inference_dir_and_grades_the_listed_checkpoint(tmp_path, capsys):
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  max_iterations: 10\nparallel:\n  strategy: none\n")
    (tmp_path / "m.txt").write_text(f"{cfg}\n")
    (tmp_path / "c.txt").write_text("/run/checkpoints/best.safetensors\n")
    rc = md.main(
        [
            "--manifest",
            str(tmp_path / "m.txt"),
            "--index",
            "0",
            "--output-base",
            "/camp",
            "--inference-checkpoints",
            str(tmp_path / "c.txt"),
            "--dry-run",
            "--no-audit",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "training.output_dir=/camp/arm/inference" in out
    assert "--validate-only --resume /run/checkpoints/best.safetensors" in out
    assert "if-present" not in out


@pytest.mark.parametrize("extra", [["--prod"], ["--resume"], ["--train-iters", "5"]])
def test_inference_mode_refuses_training_knobs(tmp_path, extra):
    (tmp_path / "m.txt").write_text("x.yaml\n")
    (tmp_path / "c.txt").write_text("c\n")
    rc = md.main(
        [
            "--manifest",
            str(tmp_path / "m.txt"),
            "--index",
            "0",
            "--output-base",
            "/camp",
            "--inference-checkpoints",
            str(tmp_path / "c.txt"),
            "--dry-run",
            *extra,
        ]
    )
    assert rc == 2


def test_inference_mode_needs_an_output_base(tmp_path):
    (tmp_path / "m.txt").write_text("x.yaml\n")
    (tmp_path / "c.txt").write_text("c\n")
    assert (
        md.main(
            [
                "--manifest",
                str(tmp_path / "m.txt"),
                "--index",
                "0",
                "--inference-checkpoints",
                str(tmp_path / "c.txt"),
                "--dry-run",
            ]
        )
        == 2
    )


def _args(campaign, **kw):
    import argparse

    base = {
        "campaign_dir": campaign,
        "base_dir": campaign,
        "only": [],
        "force": False,
        "concurrency": None,
        "slurm": [],
        "time": None,
        "dry_run": True,
        "check": False,
        "rtol": 1e-3,
        "atol": 1e-6,
    }
    return argparse.Namespace(**{**base, **kw})


def _campaign(tmp_path, arms):
    from spectramr.infrastructure.orchestration.campaign_state import (
        CampaignState,
        ExperimentStatus,
    )

    exps = []
    for stem, has_ckpt in arms:
        (tmp_path / f"{stem}.yaml").write_text("x: 1\n")
        ck = tmp_path / stem / "checkpoints"
        ck.mkdir(parents=True)
        (tmp_path / stem / "test_metrics.json").write_text('{"manifest": "t.json"}')
        if has_ckpt:
            (ck / "best.safetensors").write_bytes(b"x")
        exps.append(
            ExperimentStatus(
                name=stem,
                config_path=str(tmp_path / f"{stem}.yaml"),
                output_dir=f"/c/{stem}",
                status="completed",
            )
        )
    CampaignState(campaign_name="c", campaign_dir=str(tmp_path), experiments=exps).save()


def test_a_partial_request_is_refused_before_anything_is_submitted(tmp_path, monkeypatch):
    from spectramr.cli.campaign_infer import campaign_infer
    from spectramr.infrastructure.orchestration import campaign_inference as ci

    _campaign(tmp_path, [("a", True), ("b", False)])
    monkeypatch.setattr(
        ci, "submit_inference", lambda *a, **k: pytest.fail("submitted a partial array")
    )
    assert campaign_infer(_args(tmp_path)) == 1


def test_only_asks_for_a_subset_explicitly(tmp_path, monkeypatch):
    from spectramr.cli.campaign_infer import campaign_infer
    from spectramr.infrastructure.orchestration import campaign_inference as ci

    _campaign(tmp_path, [("a", True), ("b", False)])
    sent = {}
    monkeypatch.setattr(ci, "submit_inference", lambda plans, *a, **k: sent.update(n=len(plans)))
    assert campaign_infer(_args(tmp_path, only=["a"])) == 0 and sent["n"] == 1


def test_check_stamps_a_verdict_the_report_names(tmp_path):
    import json

    from spectramr.cli.campaign_infer import campaign_infer
    from spectramr.core.run_artifacts import read_parity

    _campaign(tmp_path, [("a", True)])
    (tmp_path / "a" / "inference").mkdir()
    (tmp_path / "a" / "final_eval.json").write_text(json.dumps({"psnr": {"s/0": 25.0}}))
    (tmp_path / "a" / "inference" / "final_eval.json").write_text(
        json.dumps({"psnr": {"s/0": 26.0}})
    )
    assert campaign_infer(_args(tmp_path, check=True)) == 1
    assert read_parity(tmp_path / "a") == "mismatch"
