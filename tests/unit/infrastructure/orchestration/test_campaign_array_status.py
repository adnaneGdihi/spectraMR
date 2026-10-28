"""Array campaign arms are polled per task, not by their shared job id.

``query_batch_status`` keys statuses by job id and skips ``<job>_<task>`` lines,
so before this every arm of a campaign array fell back to the job's first line,
task 0's state: a TIMEOUT task was recorded ``completed`` once task 0 finished,
``campaign watch`` evaluated, and ``--resume`` skipped it.
"""

from __future__ import annotations

from spectramr.infrastructure.orchestration.campaign_orchestrator import CampaignOrchestrator
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)
from spectramr.infrastructure.orchestration.slurm_backend import (
    JobStatus,
    parse_array_task_lines,
)

_SACCT = "\n".join(
    [
        "500_0|COMPLETED|0:0|2026-10-08T16:00|2026-10-09T10:00|18:00:00|n1|None",
        "500_0.batch|COMPLETED|0:0|2026-10-08T16:00|2026-10-09T10:00|18:00:00|n1|None",
        "500_1|TIMEOUT|0:0|2026-10-08T16:00|2026-10-10T16:00|48:00:00|n2|None",
        "500_2|RUNNING|0:0|2026-10-09T10:00||02:00:00|n1|None",
        "500_[3-5%2]|PENDING|0:0|||00:00:00||JobArrayTaskLimit",
    ]
)


def test_each_task_and_each_pending_range_is_parsed() -> None:
    parsed = parse_array_task_lines(_SACCT)
    states = {task: s.normalised_state for (_job, task), s in parsed.items()}
    assert states == {
        0: "completed",
        1: "timeout",
        2: "running",
        3: "submitted",  # SLURM PENDING normalises to the campaign's "submitted"
        4: "submitted",
        5: "submitted",
    }


def _campaign(tmp_path):
    state = CampaignState(campaign_name="c", campaign_dir=str(tmp_path))
    for task, name in enumerate(["a", "b", "c"]):
        state.experiments.append(
            ExperimentStatus(
                name=name,
                config_path=f"{name}.yaml",
                slurm_job_id=500,
                array_task_id=task,
                status="submitted",
                output_dir=str(tmp_path / name),
            )
        )
    state.save()


def test_check_progress_reads_each_arm_from_its_own_task(tmp_path, monkeypatch) -> None:
    _campaign(tmp_path)
    orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
    monkeypatch.setattr(
        orch.slurm, "query_array_task_status", lambda ids: parse_array_task_lines(_SACCT)
    )
    monkeypatch.setattr(orch.slurm, "query_batch_status", lambda ids: {})
    monkeypatch.setattr(orch, "_discover_results", lambda exp: None)
    state = orch.check_progress(str(tmp_path))
    assert {e.name: e.status for e in state.experiments} == {
        "a": "completed",
        "b": "timeout",
        "c": "running",
    }


def test_planted_job_keyed_status_gives_every_arm_task_zeros_state(tmp_path) -> None:
    """The shape that was wrong: one JobStatus per job id, task 0's line."""
    by_job = {500: JobStatus(job_id=500, state="COMPLETED")}
    _campaign(tmp_path)
    state = CampaignState.load(tmp_path / "campaign_state.json")
    read = {e.name: by_job[e.slurm_job_id].normalised_state for e in state.experiments}
    assert set(read.values()) == {"completed"}  # b timed out and c is running
