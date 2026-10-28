"""Tests for the unified ``spectramr launch`` dispatcher (WS-D).

``launch`` resolves a backend from ``--where``, builds a pipeline-agnostic
``SpectraMRInvocation`` from ``--pipeline``/``--config``, and runs it. The
in-process ``LocalBackend`` lives here (it calls the CLI dispatch); the other
backends are the pure infra ones.
"""

from __future__ import annotations

import argparse

import pytest

import spectramr.cli.launch as launch_mod
from spectramr.cli.launch import LocalBackend, resolve_backend
from spectramr.infrastructure.execution import (
    ApptainerBackend,
    DockerBackend,
    SpectraMRInvocation,
    ResourceSpec,
    SlurmBackend,
)


def _args(**over):
    base = {
        "pipeline": "train",
        "config": "x.yaml",
        "where": "local",
        "fanout": "single",
        "dry_run": True,
        "account": None,
        "partition": None,
        "mem": None,
        "gpus": None,
        "time": None,
        "nodes": None,
        "extra": [],
    }
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _clean_launch_env():
    """Isolate SPECTRAMR_LAUNCH_* (launch() writes them into os.environ)."""
    import os

    saved = {k: v for k, v in os.environ.items() if k.startswith("SPECTRAMR_LAUNCH_")}
    for k in saved:
        del os.environ[k]
    yield
    for k in [k for k in os.environ if k.startswith("SPECTRAMR_LAUNCH_")]:
        del os.environ[k]
    os.environ.update(saved)


def test_resolve_backend():
    assert isinstance(resolve_backend("local"), LocalBackend)
    assert isinstance(resolve_backend("docker"), DockerBackend)
    assert isinstance(resolve_backend("apptainer"), ApptainerBackend)
    assert isinstance(resolve_backend("slurm"), SlurmBackend)
    with pytest.raises(ValueError, match="Unknown"):
        resolve_backend("nope")


def test_launch_exports_launch_env_for_provenance(capsys):
    """launch() stamps SPECTRAMR_LAUNCH_* (backend + resolved resources) so the
    child run can record them in run_summary.json (pitfall #15c)."""
    import os

    rc = launch_mod.launch(_args(where="local", gpus=2, account="acct"))
    assert rc == 0
    assert os.environ["SPECTRAMR_LAUNCH_BACKEND"] == "local"
    assert os.environ["SPECTRAMR_LAUNCH_GPUS"] == "2"
    assert os.environ["SPECTRAMR_LAUNCH_ACCOUNT"] == "acct"
    capsys.readouterr()  # swallow the dry-run print


def test_local_backend_dry_run():
    handle = LocalBackend().run(SpectraMRInvocation("train", "x.yaml"), ResourceSpec(), dry_run=True)
    assert handle.command == ["spectramr", "train", "--config", "x.yaml"]
    assert handle.returncode is None


def test_local_backend_runs_in_process(monkeypatch):
    captured: dict = {}
    import spectramr.cli.app as app

    monkeypatch.setattr(app, "main", lambda argv: captured.__setitem__("argv", argv) or 0)
    handle = LocalBackend().run(SpectraMRInvocation("train", "x.yaml"), ResourceSpec())
    assert captured["argv"] == ["train", "--config", "x.yaml"]
    assert handle.returncode == 0


def test_launch_single_local_dry_run_prints_command(capsys):
    rc = launch_mod.launch(_args(where="local"))
    assert rc == 0
    assert "spectramr train --config x.yaml" in capsys.readouterr().out


def test_launch_single_slurm_dry_run_emits_script(capsys):
    rc = launch_mod.launch(_args(where="slurm", account="acct", gpus=2, mem="64G"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "#SBATCH --account=acct" in out
    assert "#SBATCH --gpus=2" in out


def test_launch_campaign_local_raises():
    # campaigns can't run on the in-process backend; fail loudly (pitfall #9).
    with pytest.raises(NotImplementedError):
        launch_mod.launch(_args(fanout="campaign", where="local", dry_run=False))


def test_launch_campaign_docker_delegates_with_where(monkeypatch):
    import spectramr.cli.app as app

    captured: dict = {}
    monkeypatch.setattr(app, "main", lambda argv: captured.__setitem__("argv", argv) or 0)
    rc = launch_mod.launch(_args(fanout="campaign", where="docker", config="C.yaml", dry_run=False))
    assert rc == 0
    assert captured["argv"][:3] == ["campaign", "submit", "C.yaml"]
    assert "--where" in captured["argv"] and "docker" in captured["argv"]


# ---------------------------------------------------------------------------
# Verb compatibility (WS-D / PR #130 N1+N3). ``launch`` injects the config as a
# ``--config`` FLAG, so every advertised PIPELINE_VERB must accept ``--config``
# when parsed by the REAL CLI parser. Pre-fix only ``train`` was exercised, so
# ``audit`` (positional config), ``report``/``meta-evaluate`` (no config), and
# ``predict`` (``--model``/``--input``, no config) were advertised but would
# fail at argparse time. These tests parse the constructed argv against
# ``build_parser()`` without executing the command.
# ---------------------------------------------------------------------------

#: Minimal verb-specific required args (beyond the injected ``--config``) so the
#: constructed argv parses; these are what a real caller passes via ``-- …``.
_VERB_REQUIRED_EXTRAS = {
    "train": (),
    "sanity_check": (),
    "ablation": (),
    "infer": ("--checkpoint", "c.pt", "--input", "in/"),
    "hpo": ("--model-type", "unet"),
    "experiment": ("--experiment", "e1"),
    "profile": (),
}


@pytest.mark.parametrize("verb", launch_mod.PIPELINE_VERBS)
def test_every_advertised_verb_accepts_injected_config(verb):
    """Each ``PIPELINE_VERBS`` entry's launcher-constructed argv parses against
    the real parser — i.e. the verb actually accepts the ``--config`` flag the
    launcher injects (the regression that exposed audit/report/predict)."""
    from spectramr.cli.app import build_parser

    extras = _VERB_REQUIRED_EXTRAS[verb]  # KeyError => a new verb lacks coverage
    argv = SpectraMRInvocation(verb, "cfg.yaml", extra_args=extras).to_cli_args()
    parsed = build_parser().parse_args(argv)  # must NOT SystemExit
    assert parsed.command == verb


@pytest.mark.parametrize("verb", ["predict", "audit", "report", "meta-evaluate", "infer-dataset"])
def test_config_incompatible_or_deprecated_verbs_not_advertised(verb):
    """Verbs whose config is positional/absent (or that are deprecated) must NOT
    be in PIPELINE_VERBS — advertising them is a CLI lie (pitfall #15)."""
    assert verb not in launch_mod.PIPELINE_VERBS


def test_audit_with_injected_config_is_an_argparse_error():
    """Documents WHY audit is excluded: it takes the config positionally, so the
    launcher's ``--config`` injection is rejected by the real parser."""
    from spectramr.cli.app import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["audit", "--config", "x.yaml"])
