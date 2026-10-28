"""The submit-time status screen applies the pipeline's launch refusal (#2049).

One planted arm per refused status and one per status the pipeline runs, so the
screen is shown to separate the two sets rather than to drop everything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.cli import manifest_status_screen as screen
from spectramr.config.schemas.base import EXPERIMENT_STATUSES, LAUNCH_REFUSED_STATUSES


def _arm(path: Path, status: str | None) -> Path:
    meta = "metadata:\n  name: probe\n" + (f"  status: {status}\n" if status else "")
    path.write_text("config_version: '1.0'\n" + meta)
    return path


@pytest.mark.parametrize("status", sorted(LAUNCH_REFUSED_STATUSES))
def test_each_refused_status_is_dropped(tmp_path, status):
    arm = _arm(tmp_path / f"{status}.yaml", status)
    kept, refused = screen.screen_manifest([str(arm)], allow_status=None, repo_root=tmp_path)
    assert kept == []
    assert refused == [(status, str(arm))]


@pytest.mark.parametrize(
    "status", [*sorted(set(EXPERIMENT_STATUSES) - LAUNCH_REFUSED_STATUSES), None]
)
def test_statuses_the_pipeline_runs_are_kept(tmp_path, status):
    arm = _arm(tmp_path / "arm.yaml", status)
    kept, refused = screen.screen_manifest([str(arm)], allow_status=None, repo_root=tmp_path)
    assert kept == [str(arm)]
    assert refused == []


def test_allow_status_keeps_only_the_status_it_names(tmp_path):
    stub = _arm(tmp_path / "stub.yaml", "needs_implementation")
    inert = _arm(tmp_path / "inert.yaml", "inert")
    kept, refused = screen.screen_manifest(
        [str(stub), str(inert)], allow_status="needs_implementation", repo_root=tmp_path
    )
    assert kept == [str(stub)]
    assert refused == [("inert", str(inert))]


def test_a_relative_line_resolves_against_the_repo_root(tmp_path):
    _arm(tmp_path / "blocked.yaml", "blocked")
    _, refused = screen.screen_manifest(["blocked.yaml"], allow_status=None, repo_root=tmp_path)
    assert refused == [("blocked", "blocked.yaml")]


def test_a_missing_file_is_left_for_the_dispatcher_to_report(tmp_path):
    kept, _ = screen.screen_manifest(["nope.yaml"], allow_status=None, repo_root=tmp_path)
    assert kept == ["nope.yaml"]


@pytest.mark.parametrize(
    ("verb_args", "expected"),
    [
        ([], None),
        (["--debug"], None),
        (["--allow-status", "inert"], "inert"),
        (["--allow-status=blocked"], "blocked"),
        (["--allow-status", "inert", "--allow-status", "blocked"], "blocked"),
    ],
)
def test_allow_status_is_read_the_way_argparse_reads_it(verb_args, expected):
    assert screen.allow_status_from_verb_args(verb_args) == expected


def _files(tmp_path, *arms):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("".join(f"{a}\n" for a in arms))
    return manifest, tmp_path / "refused.txt"


def test_main_rewrites_the_manifest_and_lists_the_refused(tmp_path):
    ok = _arm(tmp_path / "ok.yaml", "ready")
    stub = _arm(tmp_path / "stub.yaml", "needs_implementation")
    manifest, refused = _files(tmp_path, ok, stub)
    assert screen.main(["--manifest", str(manifest), "--refused", str(refused)]) == 0
    assert manifest.read_text() == f"{ok}\n"
    assert refused.read_text() == f"needs_implementation\t{stub}\n"


def test_main_reads_allow_status_from_the_frozen_verb_args(tmp_path):
    stub = _arm(tmp_path / "stub.yaml", "needs_implementation")
    manifest, refused = _files(tmp_path, stub)
    verb_args = tmp_path / "verb_args.txt"
    verb_args.write_text("--allow-status\nneeds_implementation\n")
    screen.main(
        ["--manifest", str(manifest), "--refused", str(refused), "--verb-args-file", str(verb_args)]
    )
    assert manifest.read_text() == f"{stub}\n"
    assert refused.read_text() == ""


def test_a_verb_that_never_reads_the_status_is_not_screened(tmp_path):
    stub = _arm(tmp_path / "stub.yaml", "needs_implementation")
    manifest, refused = _files(tmp_path, stub)
    screen.main(["--manifest", str(manifest), "--refused", str(refused), "--verb", "infer"])
    assert manifest.read_text() == f"{stub}\n"
    assert refused.read_text() == ""
