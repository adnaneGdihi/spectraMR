"""The per-arm campaign script must stop at the wall clock, save, and requeue.

Only the array path and ``train_distributed.sbatch`` had yield handling, so a per-arm job was
killed at the wall and lost everything since its last periodic save. The yield block of the
generated script is executed here with a stub training command (which writes the marker the
way the loop does) and a stub ``scontrol``, because a text match cannot show that the branches
behave.
"""

from __future__ import annotations

import stat
import subprocess
import time
from pathlib import Path

import pytest

from spectramr.infrastructure.orchestration.slurm_backend import SLURMBackend

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _account(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)


def _base(tmp_path: Path) -> Path:
    """A campaign base_dir whose scripts/common is the real helper directory."""
    base = tmp_path / "base"
    (base / "scripts").mkdir(parents=True)
    (base / "scripts" / "common").symlink_to(REPO / "scripts" / "common")
    return base


def _script(base: Path, *, resume: bool) -> str:
    return SLURMBackend.generate_job_script(
        experiment_name="exp",
        config_path="c.yaml",
        output_dir=str(base / "out"),
        base_dir=str(base),
        slurm_params={"gpus": 1},
        resume=resume,
    )


def _yield_section(script: str) -> str:
    """From the wall-clock block to just before the inference branch."""
    start = script.index("# ── Wall-clock yield ──")
    end = script.index("if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then", start)
    return script[start:end]


def _run(
    tmp_path: Path,
    *,
    resume: bool,
    train_rc: int,
    write_marker: bool,
    deadline: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    base = _base(tmp_path)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    calls = tmp_path / "scontrol_calls.txt"
    scontrol = stubs / "scontrol"
    scontrol.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nexit 0\n')
    scontrol.chmod(scontrol.stat().st_mode | stat.S_IEXEC)

    # Stand-in for the training command: writes the marker the way the loop does.
    train = stubs / "train.sh"
    marker_line = 'echo "{}" > "$SPECTRAMR_WALL_CLOCK_MARKER"\n' if write_marker else ""
    train.write_text(f"#!/bin/sh\n{marker_line}exit {train_rc}\n")
    train.chmod(train.stat().st_mode | stat.S_IEXEC)

    body = (
        f'TRAIN_CMD="{train}"\n'
        + _yield_section(_script(base, resume=resume))
        + 'echo "REACHED_INFERENCE"\n'
    )
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "SLURM_JOB_ID": "4242"}
    if deadline:
        env["SLURM_JOB_END_TIME"] = str(int(time.time()) + 3600)
    result = subprocess.run(
        ["/usr/bin/bash", "-c", body], env=env, capture_output=True, text=True, check=False
    )
    return result, calls.read_text().splitlines() if calls.exists() else []


class TestYield:
    def test_a_clean_yield_with_resume_requeues_and_skips_inference(self, tmp_path) -> None:
        result, calls = _run(tmp_path, resume=True, train_rc=0, write_marker=True)
        assert result.returncode == 0, result.stderr
        assert calls == ["requeue 4242"]
        assert "REACHED_INFERENCE" not in result.stdout

    def test_a_fresh_script_yields_and_requeues_too(self, tmp_path) -> None:
        """A fresh arm is the common campaign case. The yield used to require resume=True, so
        it could not chain at all; now the requeued link resumes (RESUME_ARG, per link)."""
        result, calls = _run(tmp_path, resume=False, train_rc=0, write_marker=True)
        assert result.returncode == 0, result.stderr
        assert calls == ["requeue 4242"]
        assert "REACHED_INFERENCE" not in result.stdout

    def test_a_nonzero_exit_with_a_marker_is_not_a_yield(self, tmp_path) -> None:
        result, calls = _run(tmp_path, resume=True, train_rc=7, write_marker=True)
        assert result.returncode == 7
        assert calls == []
        assert "not requeueing" in result.stderr

    def test_no_marker_falls_through_to_the_normal_path(self, tmp_path) -> None:
        result, calls = _run(tmp_path, resume=True, train_rc=0, write_marker=False)
        assert "REACHED_INFERENCE" in result.stdout
        assert calls == []

    def test_without_a_deadline_there_is_no_marker_and_no_yield(self, tmp_path) -> None:
        """No Slurm end time means train to max_iterations, never an error."""
        result, calls = _run(tmp_path, resume=True, train_rc=0, write_marker=True, deadline=False)
        assert result.returncode == 0
        assert calls == []
        assert "REACHED_INFERENCE" in result.stdout

    def test_a_missing_helper_fails_loudly(self, tmp_path) -> None:
        script = SLURMBackend.generate_job_script(
            experiment_name="exp",
            config_path="c.yaml",
            output_dir=str(tmp_path / "out"),
            base_dir=str(tmp_path / "nowhere"),
            slurm_params={"gpus": 1},
            resume=True,
        )
        body = "TRAIN_CMD=true\n" + _yield_section(script)
        result = subprocess.run(
            ["/usr/bin/bash", "-c", body],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 3 and "missing" in result.stderr


def test_the_marker_is_on_shared_storage_not_tmpdir(tmp_path) -> None:
    section = _yield_section(_script(_base(tmp_path), resume=True))
    assert "TMPDIR" not in section.split("SPECTRAMR_WALL_CLOCK_MARKER=")[1].split("\n")[0]
    assert "experiments/results/dispatch/yield_" in section


def test_the_generated_script_is_still_valid_bash(tmp_path) -> None:
    base = _base(tmp_path)
    for resume in (True, False):
        checked = subprocess.run(
            ["/usr/bin/bash", "-n"],
            input=_script(base, resume=resume),
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stderr


class TestRequeueDirectives:
    """A yield is a requeue of this job: Slurm refuses it without the directive, and append
    keeps the earlier link's log. Only a resuming script can yield."""

    def test_a_resuming_script_is_requeue_eligible_and_appends(self, tmp_path) -> None:
        header = _script(_base(tmp_path), resume=True)
        assert "#SBATCH --requeue" in header
        assert "#SBATCH --open-mode=append" in header

    def test_a_fresh_script_is_requeue_eligible_too(self, tmp_path) -> None:
        """Its requeued link resumes, so a node failure no longer restarts it at iteration 0."""
        header = _script(_base(tmp_path), resume=False)
        assert "#SBATCH --requeue" in header
        assert "#SBATCH --open-mode=append" in header
