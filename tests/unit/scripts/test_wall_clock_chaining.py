"""The 120 h wall-clock chain, end to end across bash and Python.

A production arm is allowed 120 h per job and needs more, so the run has to
survive being cut in half. Three things have to hold together for that, and
each one is silently satisfiable on its own:

1. the launcher must tell the run when its allocation ends,
2. the run must stop and save a margin ahead of it,
3. the launcher must ask Slurm to run the task again.

Break any one and the other two still look correct: a job that never derives a
deadline trains normally and dies at the wall; a job that yields and is never
requeued leaves a checkpoint nobody continues from. So every test here plants
the broken shape rather than the working one (non-negotiable 15).

The bash half is executed rather than grepped, following
``test_train_distributed_sbatch.py``: asserting on the script's *text* is how a
derivation that no longer runs keeps passing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.utils.repo_scripts import require_repo_file

_HELPER_REL = "scripts/common/wall_clock.sh"
_LAUNCHERS = (
    "scripts/training/dispatch_experiments.sbatch",
    "scripts/training/train_distributed.sbatch",
)

#: Ordinary utilities the helper calls, so the harness can run under ``env -i``
#: without inheriting the developer's shell.
_BASE_PATH = "/usr/local/bin:/usr/bin:/bin"


def _run_helper(env: dict[str, str], fake_bin: Path | None = None) -> subprocess.CompletedProcess:
    """Source the real helper, call it, and print what it exported."""
    helper = require_repo_file(_HELPER_REL)
    path = f"{fake_bin}:{_BASE_PATH}" if fake_bin else _BASE_PATH
    script = (
        f"source {helper}\n"
        "derive_wall_clock_deadline\n"
        'echo "RESULT=${SPECTRAMR_WALL_CLOCK_DEADLINE:-unset}"\n'
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": path, **env},
    )


def _result_of(proc: subprocess.CompletedProcess) -> str:
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT="):
            return line.split("=", 1)[1]
    raise AssertionError(f"helper printed no RESULT line:\n{proc.stdout}\n{proc.stderr}")


# ---------------------------------------------------------------------------
# 1. The launcher derives the deadline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_helper_is_valid_bash() -> None:
    """A syntax error here is invisible until the job is queued on the cluster."""
    proc = subprocess.run(
        ["bash", "-n", str(require_repo_file(_HELPER_REL))], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.unit
def test_slurm_job_end_time_is_used_directly() -> None:
    """Slurm >= 21.08 hands the epoch second over; no RPC needed."""
    proc = _run_helper({"SLURM_JOB_END_TIME": "1893456000"})

    assert _result_of(proc) == "1893456000"
    assert "SLURM_JOB_END_TIME" in proc.stdout


@pytest.mark.unit
def test_an_explicit_deadline_is_honoured() -> None:
    """An operator stating the deadline must win over any derivation."""
    proc = _run_helper({"SPECTRAMR_WALL_CLOCK_DEADLINE": "42", "SLURM_JOB_END_TIME": "1893456000"})

    assert _result_of(proc) == "42"


@pytest.mark.unit
def test_no_slurm_means_no_budget() -> None:
    """An interactive run has no wall: train to max_iterations."""
    proc = _run_helper({})

    assert _result_of(proc) == "unset"
    assert proc.returncode == 0, "absence of a limit is not an error"


@pytest.mark.unit
def test_deadline_falls_back_to_squeue(tmp_path: Path) -> None:
    """Older Slurm does not export the end time, so it is asked for."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text('#!/bin/bash\necho "2030-01-01T00:00:00"\n')
    squeue.chmod(0o755)

    proc = _run_helper({"SLURM_JOB_ID": "12345"}, fake_bin=fake_bin)

    expected = subprocess.run(
        ["date", "-d", "2030-01-01T00:00:00", "+%s"], capture_output=True, text=True
    ).stdout.strip()
    assert _result_of(proc) == expected
    assert "squeue" in proc.stdout


@pytest.mark.unit
@pytest.mark.parametrize("reported", ["UNLIMITED", "N/A", ""])
def test_an_unbounded_allocation_yields_no_budget(tmp_path: Path, reported: str) -> None:
    """`UNLIMITED` must read as "no wall", never as a date that fails to parse."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text(f'#!/bin/bash\necho "{reported}"\n')
    squeue.chmod(0o755)

    proc = _run_helper({"SLURM_JOB_ID": "12345"}, fake_bin=fake_bin)

    assert _result_of(proc) == "unset"
    assert proc.returncode == 0


@pytest.mark.unit
def test_an_unparseable_end_time_does_not_export_a_garbage_deadline(tmp_path: Path) -> None:
    """Exporting an empty or malformed value would make the run raise at startup."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text('#!/bin/bash\necho "not-a-date"\n')
    squeue.chmod(0o755)

    proc = _run_helper({"SLURM_JOB_ID": "12345"}, fake_bin=fake_bin)

    assert _result_of(proc) == "unset"


# ---------------------------------------------------------------------------
# 2. The launchers are set up to be requeued
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", _LAUNCHERS, ids=lambda p: Path(p).stem)
@pytest.mark.unit
def test_launcher_allows_requeue_and_appends_its_log(rel: str) -> None:
    """Both are required and neither announces its absence.

    Without ``--requeue`` Slurm refuses the ``scontrol requeue`` the dispatcher
    issues, and the chain stops after one link. Without ``--open-mode=append``
    the requeued task truncates its predecessor's ``.out`` -- so the evidence of
    why the first link stopped is destroyed by the link that continues it.
    """
    text = require_repo_file(rel).read_text()

    assert "#SBATCH --requeue" in text
    assert "#SBATCH --open-mode=append" in text


@pytest.mark.parametrize("rel", _LAUNCHERS, ids=lambda p: Path(p).stem)
@pytest.mark.unit
def test_launcher_delegates_to_the_one_derivation(rel: str) -> None:
    """A second copy of the derivation is the shape non-negotiable 17 forbids."""
    text = require_repo_file(rel).read_text()
    executable = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

    assert "scripts/common/wall_clock.sh" in executable
    assert "derive_wall_clock_deadline" in executable
    assert "squeue" not in executable, "the launcher re-derives instead of delegating"


# ---------------------------------------------------------------------------
# 3. The dispatcher requeues the task -- but only when requeueing can help
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_launcher_names_the_marker_and_the_run_obeys(tmp_path: Path, monkeypatch) -> None:
    """One owner for the location (non-negotiable 17).

    The writer used to derive it from `training.output_dir` read out of raw YAML
    while the run read the same field through pydantic. Any validator or root
    prefix that rewrote the path would leave the two looking in different
    places, and the chain would stop without a word.
    """
    from spectramr.cli import manifest_dispatch as md
    from spectramr.core.env_names import SPECTRAMR_WALL_CLOCK_MARKER
    from spectramr.core.wall_clock import resolve_yield_marker_path

    chosen = md.yield_marker_path(tmp_path, "exp_11a", 3)
    assert chosen.name == "yield_exp_11a_3.json"

    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARKER, str(chosen))
    assert resolve_yield_marker_path("/some/unrelated/run/dir") == chosen


@pytest.mark.unit
def test_an_unnamed_marker_falls_back_into_the_run_dir(monkeypatch) -> None:
    """A hand-run `spectramr train` has no launcher to name one."""
    from spectramr.core.env_names import SPECTRAMR_WALL_CLOCK_MARKER
    from spectramr.core.wall_clock import DEFAULT_MARKER_NAME, resolve_yield_marker_path

    monkeypatch.delenv(SPECTRAMR_WALL_CLOCK_MARKER, raising=False)
    assert resolve_yield_marker_path("/run") == Path("/run") / DEFAULT_MARKER_NAME


@pytest.mark.unit
def test_requeue_invokes_scontrol_with_this_job(monkeypatch, tmp_path: Path) -> None:
    from spectramr.cli import manifest_dispatch as md

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(md.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "99887_3")

    assert md.requeue_this_task(tmp_path / "m.json") is True
    assert calls == [["scontrol", "requeue", "99887_3"]]


@pytest.mark.unit
def test_a_refused_requeue_is_reported_not_swallowed(monkeypatch, tmp_path, capsys) -> None:
    """A yielded run that is not requeued has stopped for good, and looks
    exactly like one that finished -- so the refusal must be loud."""
    from spectramr.cli import manifest_dispatch as md

    monkeypatch.setattr(
        md.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "Requeue not supported"),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "555")

    assert md.requeue_this_task(tmp_path / "m.json") is False
    err = capsys.readouterr().err
    assert "NOT finished" in err
    assert "Requeue not supported" in err
    assert "--dependency=afterany:555" in err, "no manual route offered"


@pytest.mark.unit
def test_outside_slurm_the_yield_is_explained_rather_than_requeued(
    monkeypatch, tmp_path, capsys
) -> None:
    from spectramr.cli import manifest_dispatch as md

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    assert md.requeue_this_task(tmp_path / "m.json") is False
    assert "nothing to requeue" in capsys.readouterr().err


@pytest.mark.unit
def test_a_yield_without_resume_refuses_to_requeue(monkeypatch, tmp_path, capsys) -> None:
    """THE infinite-loop shape.

    The deadline is derived, and the loop yields, whether or not the launch
    carried --resume. A requeued task re-runs the identical command: without
    resume it starts at iteration 0, yields at the next wall, and repeats --
    burning the allocation forever without ever passing 120h of training.
    """
    from spectramr.cli import manifest_dispatch as md

    cfg = tmp_path / "arm.yaml"
    cfg.write_text("training:\n  output_dir: /tmp/x\nparallel:\n  strategy: none\n")
    dispatch = tmp_path / "d"

    requeued: list[list[str]] = []

    def fake_run(argv, **kwargs):
        # The training launch: simulate a yield by leaving the marker behind.
        if any("spectramr.cli" in str(a) for a in argv):
            Path(os.environ[md.SPECTRAMR_WALL_CLOCK_MARKER]).write_text("{}")
            return subprocess.CompletedProcess(argv, 0)
        requeued.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(md.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    man = tmp_path / "m.txt"
    man.write_text(str(cfg) + "\n")

    rc = md._dispatch(
        manifest=str(man),
        index=0,
        train_iters=None,
        resume=False,
        no_audit=True,
        dispatch_dir=str(dispatch),
        dry_run=False,
    )

    assert rc == 1, "a no-resume yield must fail the task, not requeue it"
    assert requeued == [], "requeueing here loops forever at iteration 0"
    assert "loop forever" in capsys.readouterr().err


@pytest.mark.unit
def test_prod_implies_resume() -> None:
    """A production arm does not fit in one 120h allocation, so a production run
    is a chain by definition and every link must continue the last."""
    import inspect

    from spectramr.cli import manifest_dispatch as md

    src = inspect.getsource(md._dispatch)
    prod_at = src.index('print("[MODE] PROD')
    assert "resume = True" in src[prod_at : prod_at + 700]


# ---------------------------------------------------------------------------
# 4. The marker's lifecycle is what makes it trustworthy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_run_clears_a_stale_marker_before_training() -> None:
    """Left behind, it requeues a run that has already finished -- forever."""
    import inspect

    from spectramr.pipelines import train

    src = inspect.getsource(train.run_training_pipeline)
    assert "resolve_yield_marker_path(run_dir).unlink(missing_ok=True)" in src


@pytest.mark.unit
def test_the_dispatcher_also_clears_it_before_launching() -> None:
    """Belt and braces on the side that reads it: a marker from the previous
    link must never be mistaken for this link's verdict."""
    import inspect

    from spectramr.cli import manifest_dispatch as md

    src = inspect.getsource(md._dispatch)
    assert "marker.unlink(missing_ok=True)" in src
    assert src.index("marker.unlink") < src.index("[TRAIN]")


@pytest.mark.unit
def test_the_marker_records_where_to_resume(tmp_path: Path, monkeypatch) -> None:
    from spectramr.core.env_names import SPECTRAMR_WALL_CLOCK_MARKER
    from spectramr.pipelines.train import _write_yield_marker

    marker = tmp_path / "nested" / "yield.json"
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARKER, str(marker))

    _write_yield_marker(tmp_path, {"resume_from_iteration": 40123, "iterations_completed": 40123})

    payload = json.loads(marker.read_text())
    assert payload["resume_from_iteration"] == 40123
    assert payload["yielded_at"]


@pytest.mark.unit
def test_the_whole_chain_hands_off(tmp_path: Path, monkeypatch) -> None:
    """Link 1 yields where the launcher said to; link 2 finds it there."""
    from spectramr.cli import manifest_dispatch as md
    from spectramr.core.env_names import SPECTRAMR_WALL_CLOCK_MARKER
    from spectramr.pipelines.train import _write_yield_marker

    marker = md.yield_marker_path(tmp_path / "dispatch", "arm", 0)
    monkeypatch.setenv(SPECTRAMR_WALL_CLOCK_MARKER, str(marker))

    _write_yield_marker(tmp_path / "run", {"resume_from_iteration": 7000})
    assert marker.is_file()

    calls: list[list[str]] = []
    monkeypatch.setattr(
        md.subprocess,
        "run",
        lambda argv, **kw: (calls.append(argv), subprocess.CompletedProcess(argv, 0, "", ""))[1],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "4242")

    assert md.requeue_this_task(marker) is True
    assert calls[0][:2] == ["scontrol", "requeue"]


# ---------------------------------------------------------------------------
# 5. The budget and the launcher agree about the environment
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_shell_export_is_the_name_python_reads() -> None:
    """The one seam nothing else covers: bash exports, Python reads, and a typo
    on either side is a budget that silently never applies."""
    from spectramr.core.env_names import SPECTRAMR_WALL_CLOCK_DEADLINE
    from spectramr.core.wall_clock import resolve_wall_clock_budget

    deadline = time.time() + 10_000
    proc = _run_helper({"SLURM_JOB_END_TIME": str(int(deadline))})
    exported = _result_of(proc)
    assert exported != "unset"

    os.environ[SPECTRAMR_WALL_CLOCK_DEADLINE] = exported
    try:
        budget = resolve_wall_clock_budget()
        assert budget is not None
        assert budget.deadline == pytest.approx(float(exported))
    finally:
        del os.environ[SPECTRAMR_WALL_CLOCK_DEADLINE]


@pytest.mark.unit
def test_a_failing_squeue_does_not_kill_the_job(tmp_path: Path) -> None:
    """The launchers run under `set -euo pipefail`.

    A busy controller returning nonzero propagates through the pipe into the
    assignment and, without `|| true`, exits the whole 120h job before it trains
    a step. The other tests here run `bash -c` WITHOUT `-e`, so they are
    structurally blind to this -- hence the explicit flags.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text("#!/bin/bash\nexit 1\n")
    squeue.chmod(0o755)

    helper = require_repo_file(_HELPER_REL)
    proc = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            f"source {helper}\nderive_wall_clock_deadline\necho SURVIVED",
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{fake_bin}:{_BASE_PATH}", "SLURM_JOB_ID": "1"},
    )

    assert proc.returncode == 0, f"set -e killed the job: {proc.stderr}"
    assert "SURVIVED" in proc.stdout


@pytest.mark.unit
def test_the_distributed_launcher_requeues_too() -> None:
    """It derives a deadline and yields, but does not use manifest_dispatch --
    so without its own requeue it saves, exits 0, and the chain silently stops
    (the non-negotiable 16 facade shape)."""
    text = require_repo_file("scripts/training/train_distributed.sbatch").read_text()
    executable = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

    assert "scontrol requeue" in executable
    assert "SPECTRAMR_WALL_CLOCK_MARKER" in executable
    # And it must refuse the same no-resume infinite loop the dispatcher does.
    assert 'if [[ -z "${RESUME:-}" ]]; then' in executable


@pytest.mark.unit
@pytest.mark.parametrize(
    ("declared", "expected"),
    [(5000, 100), (500, 100), (100, 100), (25, 25), (10, 10), (0, 100), (-5, 100)],
)
def test_the_check_cadence_is_capped_independently_of_logging(declared, expected) -> None:
    """65 corpus arms declare `logging.intervals.log: 5000`. Riding that cadence
    inspects the deadline every ~83 min at 1 s/step, against a 900 s margin --
    the job is killed before the check fires, on exactly the long-running arms
    this exists for."""
    from spectramr.core.wall_clock import resolve_check_interval

    assert resolve_check_interval(declared) == expected


@pytest.mark.unit
def test_the_distributed_marker_lives_on_shared_storage() -> None:
    """Never ${TMPDIR}.

    This launcher supports --nodes=2, and under c10d rendezvous global rank 0 is
    whichever node arrives first -- not necessarily the one running the batch
    script. A node-local marker is written where nothing reads it and the chain
    stops silently, which is the facade shape this feature exists to close.
    """
    text = require_repo_file("scripts/training/train_distributed.sbatch").read_text()
    executable = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    marker_line = next(ln for ln in executable.splitlines() if "SPECTRAMR_WALL_CLOCK_MARKER=" in ln)

    assert "TMPDIR" not in marker_line, marker_line
    assert "REPO_ROOT" in marker_line, "the marker must sit on the shared checkout"


@pytest.mark.unit
def test_the_resume_stamp_is_rank_gated() -> None:
    """`_stamp_resume_record` is a read-modify-write on one shared path.

    Ungated, every rank appends: four entries per link on a 4-rank arm, or a
    torn read that fails open and records nothing. The process-group arms are
    precisely the ones that chain.
    """
    import inspect

    from spectramr.pipelines import train

    src = inspect.getsource(train.run_training_pipeline)
    assert "if _resume_record and _is_rank_zero:" in src


@pytest.mark.unit
def test_the_distributed_verb_forwards_the_resume_spelling_untouched() -> None:
    """A second interpretation here would make `if-present` dead on exactly the
    arms the chain is for (non-negotiable 17)."""
    import inspect

    from spectramr.pipelines import distributed

    src = inspect.getsource(distributed)
    assert "resume_path=resume_path" in src
    assert '"auto"' not in src, "the distributed path is re-interpreting the spelling"
