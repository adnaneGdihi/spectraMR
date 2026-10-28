"""``campaign profile`` is reachable from the parser and from the per-task dispatcher."""

from __future__ import annotations

import argparse

import pytest

import spectramr.cli.manifest_dispatch as md


def test_the_subcommand_is_registered_and_takes_profile_flags_after_the_separator():
    from spectramr.cli.app import build_parser

    args = build_parser().parse_args(
        ["campaign", "profile", "c.yaml", "--only", "a", "--dry-run", "--", "--mode", "full"]
    )
    assert (args.campaign_action, args.only, args.dry_run) == ("profile", ["a"], True)
    assert args.profile_args == ["--mode", "full"]
    assert callable(args.func)


def test_a_task_profiles_into_its_own_run_dir_line(tmp_path, capsys):
    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  max_iterations: 10\nparallel:\n  strategy: none\n")
    (tmp_path / "m.txt").write_text(f"{cfg}\n")
    (tmp_path / "r.txt").write_text("/camp/arm\n")
    (tmp_path / "a.txt").write_text("--mode\ncpu-only\n")
    rc = md.main(
        [
            "--manifest", str(tmp_path / "m.txt"),
            "--index", "0",
            "--verb", "profile",
            "--profile-run-dirs", str(tmp_path / "r.txt"),
            "--verb-args-file", str(tmp_path / "a.txt"),
            "--dry-run",
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0
    assert "profile --config" in out and "--mode cpu-only --output-dir /camp/arm" in out
    assert "training.output_dir" not in out


@pytest.mark.parametrize(
    "extra", [["--prod"], ["--resume"], ["--output-base", "/camp"], ["--verb", "train"]]
)
def test_profile_mode_refuses_training_knobs(tmp_path, extra):
    (tmp_path / "m.txt").write_text("x.yaml\n")
    (tmp_path / "r.txt").write_text("/camp/x\n")
    verb = [] if "--verb" in extra else ["--verb", "profile"]
    argv = ["--manifest", str(tmp_path / "m.txt"), "--index", "0", *verb]
    argv += ["--profile-run-dirs", str(tmp_path / "r.txt"), "--dry-run", *extra]
    assert md.main(argv) == 2


def _args(config, **kw):
    base = {
        "config": config,
        "base_dir": config.parent,
        "only": [],
        "concurrency": None,
        "slurm": [],
        "time": None,
        "dry_run": True,
        "profile_args": [],
    }
    return argparse.Namespace(**{**base, **kw})


def _campaign(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\n")  # b.yaml is missing
    path = tmp_path / "camp.yaml"
    path.write_text(
        "name: c\nexecution: array\nexperiments:\n- name: a\n  config: a.yaml\n"
        "- name: b\n  config: b.yaml\n"
    )
    return path


def test_a_partial_request_is_refused_before_anything_is_submitted(tmp_path, monkeypatch):
    from spectramr.cli.campaign_profile import campaign_profile
    from spectramr.infrastructure.orchestration import campaign_profile as cp

    monkeypatch.setattr(
        cp, "submit_profile", lambda *a, **k: pytest.fail("submitted a partial array")
    )
    assert campaign_profile(_args(_campaign(tmp_path))) == 1


def test_passed_flags_follow_the_preset_so_they_win(tmp_path, monkeypatch):
    from spectramr.cli.campaign_profile import campaign_profile
    from spectramr.infrastructure.orchestration import campaign_profile as cp

    sent = {}

    def fake(cfg, campaign_dir, plans, base, **kw):
        sent.update(n=len(plans), args=kw["profile_args"])
        return None, ""

    monkeypatch.setattr(cp, "submit_profile", fake)
    rc = campaign_profile(_args(_campaign(tmp_path), only=["a"], profile_args=["--mode", "full"]))
    assert rc == 0 and sent["n"] == 1
    assert sent["args"] == [*cp.DEFAULT_PROFILE_ARGS, "--mode", "full"]
