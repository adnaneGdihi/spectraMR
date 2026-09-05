"""The experiment corpus has one enumerator, and it enumerates *tracked* files.

Companion to :mod:`tests.utils.corpus`, whose module docstring carries the
incident these tests pin: cluster job ``8004252`` spent 24 of its 344 failures
on two arms that exist in no git history on any branch, because six modules
enumerated the cohort with ``Path.rglob`` and the cluster's working tree had
untracked generator output sitting in it.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from tests.utils.corpus import repo_root, tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

_COHORT = "experiments/inprogress/kspace_filling"

# A glob over `experiments/` is allowed only where comparing on-disk against
# tracked is the POINT of the test. Keyed by "<path>::<symbol>" so a file may
# hold both an exempt tripwire and ordinary scans that must be repointed.
_GLOB_EXEMPT = {
    (
        "tests/unit/experiments/test_kspace_filling_cohort_consistency.py"
        "::test_no_cohort_arm_is_hidden_from_git"
    ),
    (
        "tests/unit/experiments/test_kspace_filling_cohort_consistency.py"
        "::test_no_cohort_directory_is_covered_by_a_gitignore_rule"
    ),
    "tests/unit/test_corpus_enumeration_ssot.py::test_tracked_yamls_excludes_an_untracked_arm",
    "tests/unit/test_corpus_enumeration_ssot.py::test_no_test_module_globs_the_corpus_directly",
}

_GLOB_EXEMPT_FILES = {
    # The owner itself, whose docstring quotes the pattern it replaces.
    "tests/utils/corpus.py",
    # A standalone fuzz harness, not a pytest module: it `sys.exit(0)`s at import
    # when atheris is absent. Editing it makes `select_impacted_tests.py` hand it
    # to pytest, where that exit kills an xdist worker and the whole run dies with
    # "Unexpectedly no active workers available" -- which is what happened when
    # this PR first repointed it. It gates nothing, so on-disk enumeration is
    # harmless here. The selector/`sys.exit` trap is filed separately.
    "tests/fuzz/audit_ladder_fuzz/atheris_runner.py",
}


def _yaml_glob_calls(src: str) -> list[tuple[int, str]]:
    """``(lineno, attr)`` for every ``x.glob("*.yaml")`` / ``x.rglob(...)`` CALL.

    Parsed, not grepped. A regex over raw lines also matches the pattern quoted
    inside a docstring -- which it promptly did, flagging this ratchet's own
    helper module for *documenting* the anti-pattern it exists to remove.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"glob", "rglob"} or not node.args:
            continue
        arg = node.args[0]
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.endswith((".yaml", ".yml"))
        ):
            found.append((node.lineno, node.func.attr))
    return found


def test_repo_root_is_the_directory_holding_experiments_and_pyproject() -> None:
    """Asks git, so it survives a refactor that moves this file up or down a level.

    A hardcoded ``parents[n]`` is the most-repeated defect in this suite's
    history, and it fails *silently* when the target is globbed: a glob over a
    non-existent directory yields nothing, so the test passes by scanning zero
    files.
    """
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "experiments").is_dir()
    assert (root / "tests" / "utils" / "corpus.py").is_file()


def test_tracked_yamls_agrees_with_git_ls_files() -> None:
    """The helper must return exactly what git reports -- no more, no fewer."""
    skip_if_public_export("experiments/ does not ship; git reports no arms because there are none")
    root = repo_root()
    expected = {
        root / line
        for line in subprocess.run(
            ["git", "ls-files", _COHORT],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if line.endswith(".yaml")
    }
    assert expected, "cohort path is wrong -- git reports no arms"
    assert set(tracked_yamls(root / _COHORT)) == expected


def test_tracked_yamls_excludes_an_untracked_arm() -> None:
    """The discriminating check: without it, this module proves nothing.

    On a clean tree tracked == on-disk, so every assertion here would pass
    against a plain ``rglob`` too. Only an actually-untracked file separates the
    two implementations -- which is precisely why the defect survived until a
    cluster run whose working tree happened to be dirty.
    """
    skip_if_public_export(
        "experiments/ does not ship; there is no cohort directory to plant the probe in"
    )
    probe = repo_root() / _COHORT / "_corpus_ssot_probe.yaml"
    assert not probe.exists(), "stale probe from an interrupted run -- delete it"
    try:
        probe.write_text("config_version: '1.0'\n")

        on_disk = sorted((repo_root() / _COHORT).rglob("*.yaml"))
        tracked = tracked_yamls(repo_root() / _COHORT)

        assert probe in on_disk, "probe not visible to the glob -- test is broken"
        assert probe not in tracked, "tracked_yamls returned an UNTRACKED arm"
        assert len(on_disk) == len(tracked) + 1
    finally:
        probe.unlink(missing_ok=True)


def test_tracked_yamls_accepts_relative_paths_and_non_recursive_mode() -> None:
    """Both call shapes the repointed sites use."""
    skip_if_public_export("experiments/ does not ship; both call shapes see an empty cohort")
    root = repo_root()
    assert tracked_yamls(_COHORT) == tracked_yamls(root / _COHORT)

    top_only = tracked_yamls(_COHORT, recursive=False)
    everything = tracked_yamls(_COHORT)
    assert top_only, "non-recursive mode found nothing"
    assert set(top_only) < set(everything), "non-recursive should be a strict subset"


def test_no_test_module_globs_the_corpus_directly() -> None:
    """Ratchet: the second enumerator must not grow back.

    Scoped to modules that mention ``experiments``, so a glob over ``tmp_path``
    or a fixture directory is untouched. Exemptions are per-function, not
    per-file: ``test_kspace_filling_cohort_consistency.py`` legitimately holds
    both a tripwire that must glob and ordinary scans that must not.
    """
    root = repo_root()
    offenders: list[str] = []

    for path in sorted((root / "tests").rglob("*.py")):
        src = path.read_text(errors="ignore")
        if "experiments" not in src:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in _GLOB_EXEMPT_FILES:
            continue
        calls = _yaml_glob_calls(src)
        if not calls:
            continue
        lines = src.splitlines()

        # Map each line to its enclosing def/class, so the exemption can name a
        # function rather than blanket-approving a whole file.
        owner_of: dict[int, str] = {}
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    owner_of.setdefault(ln, node.name)

        for lineno, _attr in calls:
            if f"{rel}::{owner_of.get(lineno, '<module>')}" in _GLOB_EXEMPT:
                continue
            offenders.append(f"{rel}:{lineno}  {lines[lineno - 1].strip()}")

    assert not offenders, (
        "these enumerate the experiment corpus with a filesystem glob, so their "
        "subject differs on every machine (cluster job 8004252 lost 24 failures "
        "to this). Route through tests.utils.corpus.tracked_yamls, or add an "
        "explicit _GLOB_EXEMPT entry if comparing on-disk vs tracked IS the "
        "point:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("bad", ["/nonexistent/corpus/dir", "experiments/does_not_exist"])
def test_missing_directory_returns_empty_rather_than_raising(bad: str) -> None:
    """Matches ``Path.glob`` semantics, so repointing cannot change behaviour here.

    Note this is the one property that makes a *stale path constant* silent --
    it is inherited from the glob it replaces deliberately, because the ratchet
    above plus ``test_repo_root_...`` are what guard the path itself.
    """
    assert tracked_yamls(bad) == []


# ---------------------------------------------------------------------------
# The two absences of git
#
# ``repo_root`` distinguishes "this was never a checkout" (an sdist, a release
# tarball, a downloaded ZIP -- visible skip) from "this is a checkout whose git
# is broken" (raise). Both branches are planted here rather than reasoned about,
# because the raise branch used to be unconditional and that made a *collection*
# error out of three shipped modules -- and pytest answers a collection error by
# discarding the entire session:
#
#     43278 tests collected, 3 errors in 715.06s
#     !!!!!!!! Interrupted: 3 errors during collection !!!!!!!!
#
# measured 2026-08-28 on the exported public tree with no ``.git``. Zero of the
# 43,278 ran. One module is enough; there were three.
#
# Each plant is executed in a throwaway tree, not asserted about, and each
# harness first proves the plant actually landed -- a heredoc that silently
# wrote nothing would score green.
# ---------------------------------------------------------------------------

_PLANT_MODULE_LEVEL = """\
from tests.utils.corpus import tracked_yamls

ARMS = tracked_yamls("experiments/inprogress")   # PLANT: import time, relative

def test_uses_the_arms() -> None:
    assert ARMS is not None
"""

_PLANT_FUNCTION_LOCAL = """\
from tests.utils.corpus import repo_root

def test_reaches_the_corpus() -> None:
    assert repo_root()                           # PLANT: inside a test

def test_unrelated_must_still_run() -> None:
    assert 1 + 1 == 2
"""


def _plant_tree(tmp_path, plant_name: str, plant_body: str):
    """A minimal tree carrying the real ``corpus.py`` plus one planted module."""
    pkg = tmp_path / "tests" / "utils"
    pkg.mkdir(parents=True)
    (tmp_path / "tests" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    owner = pkg / "corpus.py"
    owner.write_text((repo_root() / "tests" / "utils" / "corpus.py").read_text())

    planted = tmp_path / "tests" / plant_name
    planted.write_text(plant_body)

    # Prove the plant landed. A truncated heredoc, a bad path or a copy that
    # wrote zero bytes all produce a green run for the wrong reason.
    assert "PLANT" in planted.read_text(), f"plant did not land in {planted}"
    assert "allow_module_level" in owner.read_text(), "copied the wrong corpus.py"
    return tmp_path


def _run_pytest(tree, *extra: str, env_overrides: dict | None = None):
    import os

    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)  # the outer run's addopts must not leak
    env["PYTHONPATH"] = str(tree)
    env.update(env_overrides or {})
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-p",
            "no:isolate",
            *extra,
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        env=env,
    )


def test_a_non_git_tree_skips_a_module_level_caller_instead_of_killing_the_session(
    tmp_path,
) -> None:
    """The shape that discarded 43,278 collected tests. Watched red before the fix."""
    tree = _plant_tree(tmp_path, "test_planted_module_level.py", _PLANT_MODULE_LEVEL)
    assert not (tree / ".git").exists(), "the plant requires a tree with no .git"

    out = _run_pytest(tree)
    combined = out.stdout + out.stderr

    assert "Interrupted" not in combined, f"session was discarded:\n{combined}"
    assert "error" not in combined.lower() or "0 error" in combined, combined
    assert "skipped" in combined, f"expected a visible skip, got:\n{combined}"
    assert "no .git" in combined, f"the skip must name the cause:\n{combined}"


def test_a_non_git_tree_skips_one_test_and_leaves_its_siblings_running(tmp_path) -> None:
    """A function-local caller must cost its own test, not the module's coverage."""
    tree = _plant_tree(tmp_path, "test_planted_function_local.py", _PLANT_FUNCTION_LOCAL)

    out = _run_pytest(tree)
    combined = out.stdout + out.stderr

    assert "1 passed" in combined, f"the unrelated sibling must still run:\n{combined}"
    assert "1 skipped" in combined, f"the corpus test must skip:\n{combined}"


def test_a_broken_checkout_still_raises_rather_than_skipping_quietly(tmp_path) -> None:
    """The other absence. Relaxing this would make 18 modules silent no-ops.

    ``.git`` exists, so the tree IS a checkout; git is simply unreachable. That
    is a defect in the checkout and must stay loud.
    """
    tree = _plant_tree(tmp_path, "test_planted_module_level.py", _PLANT_MODULE_LEVEL)
    (tree / ".git").mkdir()

    # git off PATH -> OSError from subprocess.run, the branch that must raise.
    out = _run_pytest(tree, env_overrides={"PATH": str(tmp_path / "no-such-bin")})
    combined = out.stdout + out.stderr

    assert "RuntimeError" in combined, f"a broken checkout must raise:\n{combined}"
    assert "IS a checkout" in combined, f"the message must name which absence:\n{combined}"
    assert "no .git" not in combined, f"must NOT take the skip branch:\n{combined}"
