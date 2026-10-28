"""The git questions an array dispatch asks, and the branch rule: runs follow ``dev``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spectramr.core import dispatch_git as dg

_GIT = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *_GIT, *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "dev", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "one")
    return tmp_path


def test_a_dev_checkout_may_run(repo: Path) -> None:
    assert dg.git_branch(repo) == "dev"
    assert dg.off_run_branch(repo) is None


def test_another_branch_is_named_in_the_refusal(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "feat/x")
    assert "is on 'feat/x', but every run must run on 'dev'" in dg.off_run_branch(repo)


def test_a_detached_head_is_not_a_branch(repo: Path) -> None:
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    assert dg.git_branch(repo) is None
    assert "a detached HEAD" in dg.off_run_branch(repo)


def test_contains_follows_history_and_not_its_reverse(repo: Path) -> None:
    first = dg.git_head(repo)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "two")
    second = dg.git_head(repo)
    assert dg.contains(repo, first, second)
    assert not dg.contains(repo, second, first)


def test_contains_raises_on_an_unknown_commit(repo: Path) -> None:
    with pytest.raises(dg.DispatchSnapshotError, match="merge-base"):
        dg.contains(repo, "0" * 40, dg.git_head(repo))


def test_committed_tells_an_edit_from_a_commit(repo: Path) -> None:
    path = repo / "a.txt"
    assert dg.committed(path)
    path.write_text("edited\n")
    assert not dg.committed(path)
    _git(repo, "commit", "-q", "-am", "edit")
    assert dg.committed(path)


def test_a_file_outside_any_checkout_is_never_committed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    loose = tmp_path / "a.yaml"
    loose.write_text("x: 1\n")
    assert not dg.committed(loose)


def test_the_run_branch_is_dev_unless_a_fixture_names_another() -> None:
    assert dg.run_branch({}) == "dev"
    assert dg.run_branch({dg.SPECTRAMR_DISPATCH_RUN_BRANCH: "feat/x"}) == "feat/x"
    with pytest.raises(ValueError, match="not a branch name"):
        dg.run_branch({dg.SPECTRAMR_DISPATCH_RUN_BRANCH: " dev"})


def test_the_refusal_names_the_branch_runs_follow(repo: Path) -> None:
    env = {dg.SPECTRAMR_DISPATCH_RUN_BRANCH: "release"}
    assert "must run on 'release'" in dg.off_run_branch(repo, env)


def test_a_failing_command_raises_naming_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(dg.DispatchSnapshotError, match="rev-parse HEAD"):
        dg.git_head(tmp_path)
