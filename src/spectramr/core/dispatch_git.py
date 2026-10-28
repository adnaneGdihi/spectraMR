"""The git questions an array dispatch asks of its checkout, and the branch runs follow.

Array tasks run from ``dev`` (:func:`run_branch`): the submitter refuses another
branch, and every task start refuses a checkout that has left dev or no longer
contains the submitted commit (:mod:`spectramr.core.dispatch_snapshot`).

Stdlib only: the submit wrapper runs this with whatever ``python3`` it finds,
before any venv is guaranteed to exist.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from spectramr.core.env_names import SPECTRAMR_DISPATCH_RUN_BRANCH

#: The one branch array tasks run from.
RUN_BRANCH = "dev"


class DispatchSnapshotError(RuntimeError):
    """The snapshot cannot be written or read, or the task has drifted from it."""


def _run(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DispatchSnapshotError(f"`git {' '.join(args)}` in {repo_root} failed: {exc}") from exc


def git(repo_root: Path, *args: str) -> str:
    """``git -C repo_root <args>``'s stdout; raises naming the command on any failure."""
    out = _run(repo_root, *args)
    if out.returncode != 0:
        raise DispatchSnapshotError(
            f"`git {' '.join(args)}` in {repo_root} exited {out.returncode}: "
            f"{out.stderr.strip() or out.stdout.strip()}"
        )
    return out.stdout.strip()


def git_head(repo_root: str | Path) -> str:
    """The commit ``repo_root`` has checked out; raises when git cannot say."""
    return git(Path(repo_root), "rev-parse", "HEAD")


def git_src_tree(repo_root: str | Path) -> str:
    """The tree hash of ``src/`` at HEAD: equal trees are the same importable code."""
    return git(Path(repo_root), "rev-parse", "HEAD:src")


def git_dirty(repo_root: str | Path) -> bool:
    """Whether tracked files differ from HEAD. Untracked files (run outputs) do not count."""
    return bool(git(Path(repo_root), "status", "--porcelain", "--untracked-files=no"))


def git_branch(repo_root: str | Path) -> str | None:
    """The branch ``repo_root`` has checked out, or ``None`` on a detached HEAD."""
    out = _run(Path(repo_root), "symbolic-ref", "--quiet", "--short", "HEAD")
    return out.stdout.strip() if out.returncode == 0 else None


def run_branch(env: Mapping[str, str] | None = None) -> str:
    """The branch runs follow: :data:`RUN_BRANCH` unless a test fixture names another."""
    raw = (os.environ if env is None else env).get(SPECTRAMR_DISPATCH_RUN_BRANCH, "")
    if raw == "":
        return RUN_BRANCH
    if raw.strip() != raw:
        raise ValueError(f"{SPECTRAMR_DISPATCH_RUN_BRANCH}={raw!r} is not a branch name.")
    return raw


def off_run_branch(repo_root: str | Path, env: Mapping[str, str] | None = None) -> str | None:
    """Why the checkout may not run arrays, or ``None`` when it is on :func:`run_branch`."""
    wanted = run_branch(env)
    branch = git_branch(repo_root)
    if branch == wanted:
        return None
    where = "a detached HEAD" if branch is None else repr(branch)
    return f"{repo_root} is on {where}, but every run must run on {wanted!r}"


def contains(repo_root: str | Path, ancestor: str, head: str) -> bool:
    """Whether ``ancestor`` is in the history of ``head``; raises when git cannot say."""
    out = _run(Path(repo_root), "merge-base", "--is-ancestor", ancestor, head)
    if out.returncode in (0, 1):
        return out.returncode == 0
    raise DispatchSnapshotError(f"`git merge-base` in {repo_root} failed: {out.stderr.strip()}")


def committed(path: Path) -> bool:
    """Whether ``path`` matches HEAD of the checkout holding it, so any change came in
    with a commit. A file outside any checkout cannot have been committed."""
    out = _run(Path(path).parent, "status", "--porcelain", "--", Path(path).name)
    return out.returncode == 0 and not out.stdout.strip()


__all__ = [
    "RUN_BRANCH",
    "DispatchSnapshotError",
    "committed",
    "contains",
    "git",
    "git_branch",
    "git_dirty",
    "git_head",
    "git_src_tree",
    "off_run_branch",
    "run_branch",
]
