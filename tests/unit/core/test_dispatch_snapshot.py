"""The submit-time snapshot an array task checks itself against (#2662).

Each drift test plants exactly one difference between the record and the task
and asserts the refusal names it with both values; the unchanged twin proves the
check does not refuse everything. Runs follow ``dev``: dev advancing past the
submitted commit is not drift, a checkout off dev is.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from spectramr.core import dispatch_snapshot as ds

_GIT = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *_GIT, *args], check=True, capture_output=True)


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo), *_GIT, "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), *_GIT, "commit", "-q", "--allow-empty", "-m", message],
        check=True,
    )
    return ds.git_head(repo)


@pytest.fixture
def array(tmp_path: Path) -> tuple[Path, Path, str]:
    """A checkout holding one arm, its manifest (relative entry) and the record."""
    subprocess.run(["git", "init", "-q", "-b", "dev", str(tmp_path)], check=True)
    arm = tmp_path / "arms" / "theta30.yaml"
    arm.parent.mkdir()
    arm.write_text("training:\n  output_dir: results/n2n_v2\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg.py").write_text("x = 0\n")
    head = _commit(tmp_path, "v2")
    manifest = tmp_path / "dispatch" / "manifest.txt"
    manifest.parent.mkdir()
    manifest.write_text("arms/theta30.yaml\n")
    ds.write_snapshot(manifest, tmp_path)
    return manifest, arm, head


def _verify(manifest: Path, *, allow: bool = False, env: dict[str, str] | None = None):
    return ds.verify_snapshot(manifest, "arms/theta30.yaml", allow_drift=allow, env=env or {})


def test_the_record_sits_beside_the_manifest(array) -> None:
    manifest, arm, head = array
    record = json.loads((manifest.parent / "manifest.snapshot.json").read_text())
    assert record["head"] == head
    assert record["branch"] == "dev"
    assert record["dirty"] is False
    assert record["configs"] == {"arms/theta30.yaml": ds.sha256_file(arm)}


def test_an_unchanged_task_is_verified(array) -> None:
    manifest, _arm, head = array
    verdict = _verify(manifest)
    assert verdict["status"] == "verified"
    assert verdict["current_head"] == head
    assert verdict["drift"] == []


def test_a_yaml_edited_after_submit_is_refused_by_name(array) -> None:
    manifest, arm, _head = array
    before = ds.sha256_file(arm)
    arm.write_text("training:\n  output_dir: results/n2n\n")
    with pytest.raises(ds.DispatchSnapshotError) as exc:
        _verify(manifest)
    message = str(exc.value)
    assert "config arms/theta30.yaml changed since submit" in message
    assert before in message and ds.sha256_file(arm) in message


def test_dev_advancing_past_the_submitted_commit_runs_the_task_on_the_new_one(array) -> None:
    """Runs follow dev: a pull that changes src/ is the code the task should run."""
    manifest, _arm, head = array
    (manifest.parents[1] / "src" / "pkg.py").write_text("x = 1\n")
    moved = _commit(manifest.parents[1], "pull")
    verdict = _verify(manifest)
    assert verdict["status"] == "verified"
    assert verdict["recorded_head"] == head and verdict["current_head"] == moved


def test_a_yaml_change_committed_on_dev_is_run(array) -> None:
    manifest, arm, _head = array
    arm.write_text("training:\n  output_dir: results/n2n_v3\n")
    _commit(manifest.parents[1], "arm edit merged into dev")
    assert _verify(manifest)["status"] == "verified"


def test_a_checkout_off_dev_is_refused_by_branch_name(array) -> None:
    manifest, _arm, _head = array
    _git(manifest.parents[1], "checkout", "-q", "-b", "feat/x")
    with pytest.raises(ds.DispatchSnapshotError, match=r"on 'feat/x'.*must run on 'dev'"):
        _verify(manifest)


def test_a_detached_head_is_refused_even_at_the_submitted_commit(array) -> None:
    manifest, _arm, head = array
    _git(manifest.parents[1], "checkout", "-q", "--detach", head)
    with pytest.raises(ds.DispatchSnapshotError, match=r"detached HEAD.*must run on 'dev'"):
        _verify(manifest)


def test_a_dev_that_no_longer_contains_the_submitted_commit_is_refused(array) -> None:
    manifest, _arm, head = array
    _git(manifest.parents[1], "commit", "-q", "--amend", "-m", "rewritten")
    with pytest.raises(ds.DispatchSnapshotError, match=f"{head} is not in the history"):
        _verify(manifest)


def test_a_pull_that_leaves_src_alone_does_not_refuse_the_task(array) -> None:
    """Docs and other arms' YAMLs move HEAD but not what a task imports (#2691)."""
    manifest, _arm, head = array
    (manifest.parents[1] / "README.md").write_text("docs\n")
    moved = _commit(manifest.parents[1], "docs only")
    verdict = _verify(manifest)
    assert verdict["status"] == "verified" and verdict["current_head"] == moved != head


def test_a_record_from_before_the_branch_rule_also_follows_dev(array) -> None:
    """Arrays queued before this rule carry no branch or src tree; the live checkout decides."""
    manifest, _arm, _head = array
    path = manifest.parent / "manifest.snapshot.json"
    record = json.loads(path.read_text())
    del record["src_tree"], record["branch"]
    path.write_text(json.dumps(record))
    (manifest.parents[1] / "src" / "pkg.py").write_text("x = 3\n")
    _commit(manifest.parents[1], "dev moved on")
    assert _verify(manifest)["status"] == "verified"


def test_the_refusal_says_the_saved_checkpoints_are_intact(array) -> None:
    manifest, arm, _head = array
    arm.write_text("changed\n")
    with pytest.raises(ds.DispatchSnapshotError, match="checkpoint an earlier link"):
        _verify(manifest)


def test_an_unresolvable_head_is_refused(array, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _arm, _head = array

    def _no_git(_root):
        raise ds.DispatchSnapshotError("git rev-parse failed")

    monkeypatch.setattr(ds, "git_head", _no_git)
    with pytest.raises(ds.DispatchSnapshotError, match="cannot be resolved"):
        _verify(manifest)


def test_a_config_missing_from_the_record_is_refused(array) -> None:
    manifest, _arm, _head = array
    with pytest.raises(ds.DispatchSnapshotError, match="not in the submit-time snapshot"):
        ds.verify_snapshot(manifest, "arms/other.yaml", allow_drift=False, env={})


def test_a_yaml_deleted_after_submit_is_refused_by_name(array) -> None:
    """A pull that renames or deletes an arm leaves the manifest pointing at nothing."""
    manifest, arm, _head = array
    arm.unlink()
    with pytest.raises(ds.DispatchSnapshotError, match=r"arms/theta30\.yaml is no longer readable"):
        _verify(manifest)


def test_opting_in_proceeds_and_keeps_every_drift(array) -> None:
    manifest, arm, head = array
    _git(manifest.parents[1], "checkout", "-q", "-b", "feat/x")
    (manifest.parents[1] / "src" / "pkg.py").write_text("x = 2\n")
    moved = _commit(manifest.parents[1], "feature work")
    arm.write_text("changed: true\n")
    verdict = _verify(manifest, allow=True)
    assert verdict["status"] == "drift_allowed"
    assert verdict["recorded_head"] == head and verdict["current_head"] == moved
    assert verdict["current_sha256"] == ds.sha256_file(arm)
    assert len(verdict["drift"]) == 2  # off dev, and an uncommitted YAML edit


def test_a_requeued_link_is_checked_like_a_first_start(array) -> None:
    manifest, arm, _head = array
    requeued = {"SLURM_RESTART_COUNT": "1"}
    assert _verify(manifest, env=requeued)["restart_count"] == 1
    arm.write_text("changed between links\n")
    with pytest.raises(ds.DispatchSnapshotError, match="changed since submit"):
        _verify(manifest, env=requeued)


def test_a_manifest_without_a_record_is_reported_absent(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a.yaml\n")
    verdict = ds.verify_snapshot(manifest, "a.yaml", allow_drift=False, env={})
    assert verdict["status"] == "absent"


def test_a_record_of_another_schema_is_refused(array) -> None:
    manifest, _arm, _head = array
    path = ds.snapshot_path(manifest)
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema": 99}))
    with pytest.raises(ds.DispatchSnapshotError, match="schema 99"):
        _verify(manifest)


@pytest.mark.parametrize(("raw", "expected"), [("", False), ("0", False), ("1", True)])
def test_the_opt_in_reads_three_spellings(raw: str, expected: bool) -> None:
    assert ds.allow_drift_requested({ds.SPECTRAMR_DISPATCH_ALLOW_DRIFT: raw}) is expected


@pytest.mark.parametrize("raw", ["yes", "true", "2"])
def test_any_other_opt_in_spelling_raises(raw: str) -> None:
    with pytest.raises(ValueError, match="must be unset"):
        ds.allow_drift_requested({ds.SPECTRAMR_DISPATCH_ALLOW_DRIFT: raw})


def test_recording_outside_a_checkout_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "a.yaml").write_text("x: 1\n")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a.yaml\n")
    with pytest.raises(ds.DispatchSnapshotError, match="rev-parse"):
        ds.write_snapshot(manifest, tmp_path)
    assert not ds.snapshot_path(manifest).exists()


def test_recording_off_dev_raises(array) -> None:
    manifest, _arm, _head = array
    ds.snapshot_path(manifest).unlink()
    _git(manifest.parents[1], "checkout", "-q", "-b", "feat/x")
    with pytest.raises(ds.DispatchSnapshotError, match=r"on 'feat/x'.*must run on 'dev'"):
        ds.write_snapshot(manifest, manifest.parents[1])
    assert not ds.snapshot_path(manifest).exists()


def test_recording_a_missing_yaml_raises(array) -> None:
    manifest, _arm, _head = array
    manifest.write_text("arms/gone.yaml\n")
    with pytest.raises(ds.DispatchSnapshotError, match=r"arms/gone\.yaml"):
        ds.write_snapshot(manifest, manifest.parents[1])


def test_the_cli_writes_the_record_or_exits_nonzero(array, tmp_path: Path) -> None:
    manifest, _arm, _head = array
    ds.snapshot_path(manifest).unlink()
    assert ds.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 0
    assert ds.snapshot_path(manifest).is_file()
    manifest.write_text("arms/gone.yaml\n")
    assert ds.main(["--manifest", str(manifest), "--repo-root", str(tmp_path)]) == 1


def test_the_report_is_loud_for_anything_but_verified(array, capsys) -> None:
    manifest, arm, _head = array
    arm.write_text("changed\n")
    ds.report_snapshot(_verify(manifest, allow=True))
    assert "DRIFT ALLOWED" in capsys.readouterr().err


def test_provenance_reads_the_verdict_the_dispatcher_wrote(tmp_path: Path) -> None:
    verdict = tmp_path / "v.json"
    verdict.write_text(json.dumps({"status": "drift_allowed", "drift": ["x"]}))
    env = {ds.SPECTRAMR_DISPATCH_SNAPSHOT_RECORD: str(verdict)}
    assert ds.resolve_dispatch_snapshot_provenance(env)["status"] == "drift_allowed"
    assert ds.resolve_dispatch_snapshot_provenance({}) == {}
