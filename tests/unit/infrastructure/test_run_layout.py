"""Paired with ``src/spectramr/infrastructure/run_layout.py``.

The invariant under test is "a profiling run's throwaway output is not the arm's
own output". It had three independent spellings before this module existed, so
the tests below check both halves of electing one owner: that the predicate is
correct, and that the writer and the readers are actually reading *it*.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.infrastructure import run_layout
from spectramr.infrastructure.run_layout import (
    PROFILE_SUBDIR,
    drop_profiling_artifacts,
    is_profiling_artifact,
)


def test_the_writer_uses_this_constant_rather_than_its_own_spelling() -> None:
    """One owner (non-negotiable 17): the path the writer builds must match.

    A literal ``"profiles"`` re-typed in ``profile_paths`` would keep every test
    here green while the two drifted, which is exactly the failure mode a shared
    constant is supposed to remove. So this asserts against the *real* resolver.
    """
    from spectramr.cli.profile_paths import resolve_profile_paths

    paths = resolve_profile_paths("exp_11", "run-abc", results_root=Path("R"))
    assert PROFILE_SUBDIR in paths.profile_dir.parts
    assert paths.profile_dir == Path("R/exp_11") / PROFILE_SUBDIR / "run-abc"
    assert is_profiling_artifact(paths.child_run_dir)


def test_the_discovery_skip_list_uses_this_constant() -> None:
    from spectramr.infrastructure.reporting.batch import _SKIP_DIRS

    assert PROFILE_SUBDIR in _SKIP_DIRS
    assert "profiles" in _SKIP_DIRS  # trees written before the rename still carry it


@pytest.mark.parametrize(
    "path",
    [
        "experiments/results/exp_11/profiles",
        "experiments/results/exp_11/profiles/run-abc/run",
        "experiments/results/exp_11/profiles/run-abc/run/metrics/real_images",
        Path("/abs/experiments/results/exp_11/profiles/run-abc"),
        # The current segment; the `profiles` cases above are trees written before the rename.
        "experiments/results/campaigns/c/arm/profile/run-abc/run/metrics/real_images",
    ],
)
def test_paths_inside_a_profiling_run_are_recognised(path) -> None:
    assert is_profiling_artifact(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # The substring plant. `"profiles" in str(path)` calls all three of these
        # profiling artifacts and erases a legitimately-named experiment from
        # every report. Matching on the path SEGMENT is the whole point.
        "experiments/results/profiles_ablation",
        "experiments/results/profiles_ablation/logs",
        "experiments/results/exp_profiles/metrics/real_images",
        "experiments/results/profile_sweep/metrics/real_images",
        # ...and the ordinary case.
        "experiments/results/exp_11",
        "experiments/results/exp_11/metrics/real_images",
    ],
)
def test_lookalike_names_are_not_profiling_artifacts(path) -> None:
    assert is_profiling_artifact(path) is False


def test_drop_filters_only_profiling_paths_and_preserves_order() -> None:
    given = [
        Path("experiments/results/exp_11/profiles/r1/run/metrics/real_images"),
        Path("experiments/results/exp_11/metrics/real_images"),
        Path("experiments/results/profiles_ablation/metrics/real_images"),
    ]
    kept = drop_profiling_artifacts(given)
    assert kept == [given[1], given[2]]


def test_drop_on_a_clean_list_is_the_identity() -> None:
    """Guards against a filter that quietly empties every caller's input."""
    given = [Path("a/metrics/real_images"), Path("b/metrics/real_images")]
    assert drop_profiling_artifacts(given) == given


def test_importing_this_module_stays_free_of_heavy_dependencies() -> None:
    """It sits at ``infrastructure/`` root *because* of import weight.

    Moved back under ``infrastructure/reporting/`` it would pull torch, scipy,
    pandas and matplotlib into the ``spectramr --help`` path through that
    package's eager ``__init__``. Pinned here so the placement cannot be
    "tidied" without a red test explaining why it is where it is.
    """
    import ast

    # Read through the imported module's own ``__file__``, never a
    # CWD-relative path: under a worktree the repo-relative spelling reads a
    # *different* checkout than the one on ``sys.path``, so the assertion
    # would pass on a file that is not the one under test.
    src = Path(run_layout.__file__).read_text(encoding="utf-8")
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported <= {"__future__", "pathlib"}, f"new dependency: {imported}"
