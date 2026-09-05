"""The SLURM all-tests job must be able to fail (issue #639).

Every phase ran ``pytest ... 2>&1 | tee <log>`` and then captured ``$?``. In a
pipeline ``$?`` is the *last* command's status, which is ``tee``'s, and ``tee``
succeeds whatever pytest did. All three phase exit codes were therefore 0 on
every run: the ``if [ $UNIT_EXIT -ne 0 ]`` early-abort was unreachable and the
job reported success over a red suite.

Two halves, and the second is what makes this worth having:

* **Shape** -- no phase may go back to bare ``$?`` after a ``tee``.
* **Behaviour** -- the replacement idiom is executed in a real ``bash`` and must
  actually surface the failure. A grep alone would pass on a script that read
  ``${PIPESTATUS[0]}`` three commands too late, because PIPESTATUS is reset by
  every subsequent command -- which is precisely how this is mis-fixed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.utils.repo_scripts import require_repo_file

REPO = Path(__file__).resolve().parents[3]
_SBATCH_REL = "scripts/ci/run_all_tests.sbatch"

PHASE_VARS = ["UNIT_EXIT", "INTEGRATION_EXIT", "SPECIALIZED_EXIT"]


@pytest.fixture(scope="module")
def script() -> str:
    return require_repo_file(_SBATCH_REL).read_text()


def test_script_is_valid_bash() -> None:
    """A syntax error here is invisible until the job is queued on the cluster."""
    result = subprocess.run(
        ["bash", "-n", str(require_repo_file(_SBATCH_REL))], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_it_is_a_bash_script_not_sh(script: str) -> None:
    """Anti-vacuity for the fix itself: PIPESTATUS is a bashism.

    Under ``#!/bin/sh`` the array is unset, ``${PIPESTATUS[0]}`` expands empty,
    and the comparison below becomes a syntax error -- a worse bug than #639.
    """
    assert script.splitlines()[0].strip() == "#!/bin/bash"


@pytest.mark.parametrize("var", PHASE_VARS)
def test_phase_exit_code_is_not_captured_from_the_pipeline_status(var: str, script: str) -> None:
    """``VAR=$?`` after a ``tee`` is the #639 defect exactly."""
    assert f"{var}=$?" not in script, (
        f"{var} captures `tee`'s status, not pytest's; use ${{PIPESTATUS[0]}} "
        "read immediately after the pipeline"
    )


@pytest.mark.parametrize("var", PHASE_VARS)
def test_phase_exit_code_comes_from_pipestatus(var: str, script: str) -> None:
    assert f"{var}=${{PIPESTATUS[0]}}" in script


@pytest.mark.parametrize("var", PHASE_VARS)
def test_nothing_executes_between_the_pipeline_and_the_capture(var: str, script: str) -> None:
    """PIPESTATUS is reset by the next command, so the read must be adjacent.

    Blank lines and comments are not commands and are allowed; anything else
    silently reinstates the bug while leaving the grep above satisfied.
    """
    lines = script.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == f"{var}=${{PIPESTATUS[0]}}")
    preceding = [
        ln for ln in reversed(lines[:idx]) if ln.strip() and not ln.strip().startswith("#")
    ]
    assert preceding, f"no command precedes {var}"
    assert preceding[0].strip().startswith("2>&1 | tee "), (
        f"{var} is not read immediately after its pipeline; nearest preceding "
        f"command is {preceding[0].strip()!r}, which resets PIPESTATUS"
    )


def test_the_specialized_phase_stays_non_blocking(script: str) -> None:
    """Phase 3 was ``|| true``. Making it fail the job is a behaviour change.

    ``|| true`` had to go because it clobbers PIPESTATUS, so the non-blocking
    property now rests on ``set +e`` plus nothing exiting on the variable.
    """
    assert "set +e" in script
    assert not re.search(r"exit \$SPECIALIZED_EXIT", script)


# ---------------------------------------------------------------------------
# Behaviour: the idiom, executed
# ---------------------------------------------------------------------------


def _run_bash(body: str) -> str:
    return subprocess.run(
        ["bash", "-c", body], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_the_old_idiom_really_did_swallow_the_failure() -> None:
    """Anti-vacuity: proves the bug was real, so the fix below is not a no-op."""
    assert _run_bash("(exit 7) 2>&1 | tee /dev/null\nE=$?\necho $E") == "0"


def test_the_new_idiom_surfaces_the_failure() -> None:
    assert _run_bash("(exit 7) 2>&1 | tee /dev/null\nE=${PIPESTATUS[0]}\necho $E") == "7"


def test_comments_between_pipeline_and_capture_are_harmless() -> None:
    """The committed phase 1 carries two comment lines in that gap."""
    body = "(exit 7) 2>&1 | tee /dev/null\n\n# a comment\n# another\nE=${PIPESTATUS[0]}\necho $E"
    assert _run_bash(body) == "7"


def test_a_command_between_pipeline_and_capture_destroys_it() -> None:
    """The failure mode the adjacency test above exists to catch."""
    body = "(exit 7) 2>&1 | tee /dev/null\necho ignored >/dev/null\nE=${PIPESTATUS[0]}\necho $E"
    assert _run_bash(body) == "0"


def test_the_set_plus_e_wrapper_still_reports_the_real_code() -> None:
    """Phase 3's exact shape: non-blocking, but the code is still recoverable."""
    body = "set -e\nset +e\n(exit 9) 2>&1 | tee /dev/null\nE=${PIPESTATUS[0]}\nset -e\necho $E"
    assert _run_bash(body) == "9"
