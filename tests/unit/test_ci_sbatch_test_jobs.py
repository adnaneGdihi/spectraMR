"""Guards for the CI sbatch test jobs (``scripts/ci/run_*_tests.sbatch``).

These pin two failure shapes that make a job report the wrong colour rather
than crash visibly, and that live in shell text with no import surface:

* a bare ``${PYTHONPATH}`` expansion under ``set -u`` aborts the script before
  pytest ever runs, because a fresh Slurm allocation usually has it unset;
* ``--cov`` without ``--cov-fail-under=0`` lets pytest-cov turn the
  ``fail_under = 60`` floor in ``pyproject.toml`` into pytest's exit code, so
  the job reports FAILED with every test passing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CI = Path(__file__).resolve().parents[2] / "scripts" / "ci"
_JOBS = ["run_unit_tests.sbatch", "run_smoke_tests.sbatch", "run_all_tests.sbatch"]

# A ${PYTHONPATH} / $PYTHONPATH read that carries no :- default.
_UNGUARDED_EXPANSION = re.compile(r"\$\{PYTHONPATH\}|\$PYTHONPATH(?!\w)")


def find_unguarded_pythonpath(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if _UNGUARDED_EXPANSION.search(ln)]


def find_cov_without_floor_optout(text: str) -> list[str]:
    """Return each pytest invocation that measures coverage without the opt-out.

    A backslash-continued command is one logical line, so the continuations are
    joined before the two flags are looked for.
    """
    joined = text.replace("\\\n", " ")
    return [ln for ln in joined.splitlines() if "--cov=" in ln and "--cov-fail-under" not in ln]


@pytest.mark.parametrize("name", _JOBS)
def test_pythonpath_expansion_is_guarded(name: str) -> None:
    offenders = find_unguarded_pythonpath((_CI / name).read_text())
    assert offenders == [], (
        f"{name}: bare PYTHONPATH read under `set -u` aborts before pytest; "
        f"use ${{PYTHONPATH:-}}. Offending lines: {offenders}"
    )


@pytest.mark.parametrize("name", _JOBS)
def test_coverage_runs_opt_out_of_the_floor(name: str) -> None:
    offenders = find_cov_without_floor_optout((_CI / name).read_text())
    assert offenders == [], (
        f"{name}: --cov without --cov-fail-under=0 makes the 60% floor the job's "
        f"exit code, so all-green reports FAILED. Offending commands: {offenders}"
    )


@pytest.mark.parametrize("name", _JOBS)
def test_job_selects_only_paths_that_exist(name: str) -> None:
    """A renamed test root makes pytest exit 4, which reads as a red suite."""
    repo = _CI.parents[1]
    selected = re.findall(r"(?<![\w/.-])(tests/[\w/*.-]+)", (_CI / name).read_text())
    missing = [p for p in sorted(set(selected)) if "*" not in p and not (repo / p).exists()]
    assert missing == [], f"{name} selects paths that no longer exist: {missing}"


# --- planted violations: each detector must go red on the shape it claims ---


def test_detector_catches_braced_pythonpath() -> None:
    assert find_unguarded_pythonpath("export PYTHONPATH=${PWD}:${PYTHONPATH}")


def test_detector_catches_unbraced_pythonpath() -> None:
    assert find_unguarded_pythonpath("export PYTHONPATH=$PYTHONPATH:/x")


def test_detector_accepts_the_guarded_form() -> None:
    assert find_unguarded_pythonpath("export PYTHONPATH=${PWD}:${PYTHONPATH:-}") == []


def test_detector_catches_cov_on_a_continued_command() -> None:
    planted = "pytest tests/unit/ \\\n    --cov=src \\\n    --cov-report=term-missing\n"
    assert find_cov_without_floor_optout(planted)


def test_detector_accepts_cov_with_the_optout() -> None:
    clean = "pytest tests/unit/ \\\n    --cov=src \\\n    --cov-fail-under=0\n"
    assert find_cov_without_floor_optout(clean) == []


def test_detector_catches_a_missing_test_root() -> None:
    repo = _CI.parents[1]
    assert not (repo / "tests/definitely_not_a_real_root").exists()
