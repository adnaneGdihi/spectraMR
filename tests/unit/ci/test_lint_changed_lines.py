"""``scripts/ci/lint_changed_lines.py`` gates NEW debt, not inherited debt.

The gate it replaces ran whole-file ruff on every touched file. On a tree with 13,225
pre-existing findings that made it unsatisfiable: every PR that touched a legacy module
went red for debt it did not write, so ``required`` was red on all ten PRs of the
sim2rank stack while the four lanes that actually test correctness were green. A check
that can never pass is not a gate -- it trains people to merge red.

These tests pin the two halves of the contract:
  * a finding on a line the PR did not touch must NOT fail it, and
  * a finding on a line the PR DID add must.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "lint_changed_lines.py"

# N806: `B0` is a non-lowercase local. It is also the correct symbol for the main
# magnetic field, which is why 146 of these sit in the physics tree and why nobody is
# going to rename them.
_DIRTY = "def f():\n    B0 = 3.0\n    return B0\n"
_CLEAN = "def g():\n    return 1\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "src").mkdir(parents=True)
    _git(r.parent, "init", "-q", "-b", "main", str(r))
    _git(r, "config", "user.email", "t@t.t")
    _git(r, "config", "user.name", "t")
    # ruff needs a config, or it picks up the repo's own pyproject from a parent dir.
    (r / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    (r / "ruff.toml").write_text('line-length = 100\nlint.select = ["E", "F", "N"]\n')
    return r


def _run(repo: Path, base: str, head: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--base", base, "--head", head],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_inherited_debt_on_an_untouched_line_does_not_fail_the_pr(repo: Path) -> None:
    """THE case. Legacy file is already dirty; the PR appends a clean function."""
    (repo / "src" / "legacy.py").write_text(_DIRTY)
    base = _commit(repo, "pre-existing debt")

    (repo / "src" / "legacy.py").write_text(_DIRTY + "\n\n" + _CLEAN)
    head = _commit(repo, "append something clean")

    res = _run(repo, base, head)
    assert res.returncode == 0, f"inherited N806 must not fail the PR:\n{res.stdout}"


def test_a_violation_on_a_line_the_pr_added_fails(repo: Path) -> None:
    (repo / "src" / "legacy.py").write_text(_CLEAN)
    base = _commit(repo, "clean file")

    (repo / "src" / "legacy.py").write_text(_CLEAN + "\n\n" + _DIRTY)
    head = _commit(repo, "add new debt")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "N806" in res.stdout
    assert "line added by this change" in res.stdout


def test_a_new_file_is_gated_end_to_end(repo: Path) -> None:
    """An added file has no inherited debt, so every finding in it is the PR's."""
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "brand_new.py").write_text(_DIRTY)
    head = _commit(repo, "add a dirty new file")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "new file" in res.stdout


def test_a_new_file_must_also_be_formatted(repo: Path) -> None:
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "ugly.py").write_text("def h():\n    return   [1,2,   3]\n")
    head = _commit(repo, "add an unformatted new file")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "not ruff-formatted" in res.stdout


def test_a_modified_file_is_not_conscripted_into_a_reformat(repo: Path) -> None:
    """Reformatting a legacy file to satisfy a two-line diff buries the change."""
    (repo / "src" / "legacy.py").write_text("def h():\n    return   [1,2,   3]\n")
    base = _commit(repo, "badly formatted legacy file")

    (repo / "src" / "legacy.py").write_text("def h():\n    return   [1,2,   3]\n\n\n" + _CLEAN)
    head = _commit(repo, "append a clean function")

    res = _run(repo, base, head)
    assert res.returncode == 0, res.stdout


def test_a_pr_touching_no_python_passes(repo: Path) -> None:
    (repo / "README.md").write_text("hi\n")
    base = _commit(repo, "seed")
    (repo / "README.md").write_text("hi there\n")
    head = _commit(repo, "docs only")

    res = _run(repo, base, head)
    assert res.returncode == 0
    assert "No changed Python files" in res.stdout


# --------------------------------------------------------------------------------
# Staged mode: the same policy, driven from the index instead of a commit range.
# Every case above is mirrored, because the two modes share a code path only from
# `changed_files` down -- the selector itself is the part that differs, so a bug can
# live in one mode while the other stays green.
# --------------------------------------------------------------------------------


def _stage(repo: Path) -> None:
    _git(repo, "add", "-A")


def _run_staged(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--staged"],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_staged_inherited_debt_on_an_untouched_line_does_not_fail(repo: Path) -> None:
    (repo / "src" / "legacy.py").write_text(_DIRTY)
    _commit(repo, "pre-existing debt")

    (repo / "src" / "legacy.py").write_text(_DIRTY + "\n\n" + _CLEAN)
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 0, f"inherited N806 must not fail the commit:\n{res.stdout}"


def test_staged_violation_on_an_added_line_fails(repo: Path) -> None:
    (repo / "src" / "legacy.py").write_text(_CLEAN)
    _commit(repo, "clean file")

    (repo / "src" / "legacy.py").write_text(_CLEAN + "\n\n" + _DIRTY)
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 1
    assert "N806" in res.stdout
    assert "line added by this change" in res.stdout


def test_staged_new_file_is_gated_end_to_end(repo: Path) -> None:
    (repo / "src" / "keep.py").write_text(_CLEAN)
    _commit(repo, "seed")

    (repo / "src" / "brand_new.py").write_text(_DIRTY)
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 1
    assert "new file" in res.stdout


def test_staged_new_file_must_also_be_formatted(repo: Path) -> None:
    (repo / "src" / "keep.py").write_text(_CLEAN)
    _commit(repo, "seed")

    (repo / "src" / "ugly.py").write_text("def h():\n    return   [1,2,   3]\n")
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 1
    assert "not ruff-formatted" in res.stdout


def test_staged_modified_file_is_not_conscripted_into_a_reformat(repo: Path) -> None:
    (repo / "src" / "legacy.py").write_text("def h():\n    return   [1,2,   3]\n")
    _commit(repo, "badly formatted legacy file")

    (repo / "src" / "legacy.py").write_text("def h():\n    return   [1,2,   3]\n\n\n" + _CLEAN)
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 0, res.stdout


def test_staged_commit_touching_no_python_passes(repo: Path) -> None:
    (repo / "README.md").write_text("hi\n")
    _commit(repo, "seed")
    (repo / "README.md").write_text("hi there\n")
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 0
    assert "No changed Python files" in res.stdout


def test_staged_works_on_an_unborn_branch(repo: Path) -> None:
    """The very first commit has no ``HEAD``; every staged file is an addition.

    Without the empty-tree fallback ``git diff --cached HEAD`` exits non-zero and the
    hook crashes on the one commit nobody can retry against a previous state.
    """
    (repo / "src" / "first.py").write_text(_DIRTY)
    _stage(repo)

    res = _run_staged(repo)
    assert res.returncode == 1, f"an unborn branch must still be gated:\n{res.stderr}"
    assert "new file" in res.stdout


def test_staged_reports_a_partially_staged_file_rather_than_answering_wrongly(
    repo: Path,
) -> None:
    """Line numbers come from the index, ruff reads the working tree.

    pre-commit stashes unstaged changes so the two agree under the hook. A hand run
    over a half-staged file is the case where they do not, and a confident wrong
    answer there is worse than a noisy right one.
    """
    (repo / "src" / "legacy.py").write_text(_CLEAN)
    _commit(repo, "seed")

    (repo / "src" / "legacy.py").write_text(_CLEAN + "\n\n" + _CLEAN.replace("g()", "g2()"))
    _stage(repo)
    (repo / "src" / "legacy.py").write_text(_CLEAN + "\n\n\n\n\n" + _DIRTY)

    res = _run_staged(repo)
    assert "has unstaged changes" in res.stdout, res.stdout


def test_staged_rejects_a_range_as_well(repo: Path) -> None:
    (repo / "src" / "a.py").write_text(_CLEAN)
    base = _commit(repo, "seed")
    res = subprocess.run(
        [sys.executable, str(_SCRIPT), "--staged", "--base", base, "--head", base],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "cannot also take a range" in res.stderr


def test_range_mode_still_requires_both_ends(repo: Path) -> None:
    """``pr-required.yml`` passes ``--base``/``--head``; making them optional must not
    turn a typo there into a silent pass."""
    res = subprocess.run([sys.executable, str(_SCRIPT)], cwd=repo, capture_output=True, text=True)
    assert res.returncode == 2
    assert "--base and --head" in res.stderr


# --------------------------------------------------------------------------------
# Scope: every Python file, not just src/ and tests/.
#
# The hook this script replaces linted ALL Python. Held to a src/+tests/ allowlist the
# script would have been a narrower gate than the thing it displaced -- 448 tracked
# files, 362 of them under scripts/, silently losing lint at commit time with no CI
# lane behind them. These two plants are the files that were invisible.
# --------------------------------------------------------------------------------


def test_a_dirty_new_file_outside_src_and_tests_is_gated(repo: Path) -> None:
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "scripts" / "ci" / "new_gate.py").write_text(_DIRTY)
    head = _commit(repo, "add a dirty gate")

    res = _run(repo, base, head)
    assert res.returncode == 1, f"scripts/ is in scope:\n{res.stdout}"
    assert "new file" in res.stdout


def test_a_dirty_added_line_outside_src_and_tests_is_gated(repo: Path) -> None:
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "scripts" / "ci" / "gate.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "scripts" / "ci" / "gate.py").write_text(_CLEAN + "\n\n" + _DIRTY)
    head = _commit(repo, "add new debt to a gate")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "line added by this change" in res.stdout


def test_widening_did_not_turn_it_into_a_whole_file_gate(repo: Path) -> None:
    """The control for the two above: outside src/ the added-line rule still holds."""
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "scripts" / "ci" / "gate.py").write_text(_DIRTY)
    base = _commit(repo, "pre-existing debt in a gate")

    (repo / "scripts" / "ci" / "gate.py").write_text(_DIRTY + "\n\n" + _CLEAN)
    head = _commit(repo, "append something clean")

    res = _run(repo, base, head)
    assert res.returncode == 0, res.stdout


def test_a_path_ruff_excludes_is_skipped_even_when_named_explicitly(repo: Path) -> None:
    """``--force-exclude``. ruff lints a path you hand it even when the config excludes
    it, because ``exclude`` governs directory walking -- so widening scope without the
    flag would start linting vendored trees this repo does not own."""
    (repo / "ruff.toml").write_text(
        'line-length = 100\nlint.select = ["E", "F", "N"]\nextend-exclude = ["src/vendored/"]\n'
    )
    (repo / "src" / "vendored").mkdir()
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "vendored" / "upstream.py").write_text(_DIRTY)
    head = _commit(repo, "vendor an upstream file")

    res = _run(repo, base, head)
    assert res.returncode == 0, f"src/vendored/ is excluded by ruff config:\n{res.stdout}"


# --------------------------------------------------------------------------------
# The shipped pep8-naming list, driven through the gate.
#
# These read `[tool.ruff.lint.pep8-naming]` out of the repo's OWN pyproject.toml
# rather than restating it, so deleting or narrowing the section turns them red
# instead of leaving a copy that agrees with nothing.
# --------------------------------------------------------------------------------

_IGNORED_MRI_SYMBOL = "def f(x):\n    B = x.shape[0]\n    return B\n"
_SUFFIXED_NAME = "def f(x):\n    C_source = x.shape[0]\n    return C_source\n"


def _shipped_ignore_names() -> list[str]:
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    return cfg["tool"]["ruff"]["lint"]["pep8-naming"]["extend-ignore-names"]


@pytest.fixture
def repo_shipped_naming(repo: Path) -> Path:
    """``repo``, but with ruff configured exactly as this repository configures it."""
    (repo / "ruff.toml").write_text(
        'line-length = 100\nlint.select = ["E", "F", "N"]\n'
        "[lint.pep8-naming]\n"
        f"extend-ignore-names = {json.dumps(_shipped_ignore_names())}\n"
    )
    return repo


def test_the_shipped_list_clears_a_bare_mri_symbol(repo_shipped_naming: Path) -> None:
    """`B` is the batch axis, not a style violation, and no rename improves it."""
    repo = repo_shipped_naming
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "model.py").write_text(_IGNORED_MRI_SYMBOL)
    head = _commit(repo, "add a file naming the batch axis")

    res = _run(repo, base, head)
    assert res.returncode == 0, f"`B` must not be gated:\n{res.stdout}"


def test_the_shipped_list_does_not_clear_a_suffixed_name(repo_shipped_naming: Path) -> None:
    """The control. A blanket `N806` disable would pass this too, which is the point:
    the list is narrow, and `C_source` is the case the rule was right about."""
    repo = repo_shipped_naming
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "model.py").write_text(_SUFFIXED_NAME)
    head = _commit(repo, "add a file with a suffixed name")

    res = _run(repo, base, head)
    assert res.returncode == 1, f"`C_source` must still be gated:\n{res.stdout}"
    assert "N806" in res.stdout
