"""One arm with a half-published conversion must not stop the poll for the others."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.checkpointing.select import CONVERSION_JOURNAL
from spectramr.infrastructure.orchestration.campaign_orchestrator import CampaignOrchestrator
from spectramr.infrastructure.orchestration.campaign_state import CampaignState, ExperimentStatus
from spectramr.infrastructure.orchestration.slurm_backend import JobStatus
from tests.fixtures.run_state_files import write_run_state


@pytest.fixture(autouse=True)
def _slurm_account(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")


def _run(root: Path, name: str, *, pending: bool) -> Path:
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    write_run_state(
        run / "checkpoints" / "step_00000002.safetensors",
        {"generator": {"w": torch.zeros(1)}},
        role="step",
    )
    if pending:
        (run / "checkpoints" / CONVERSION_JOURNAL).write_text("{}")
    return run


def _campaign(tmp_path: Path, slurm_state: str) -> tuple[Path, CampaignOrchestrator]:
    campaign_dir = tmp_path / "out"
    campaign_dir.mkdir()
    state = CampaignState(campaign_name="c", campaign_dir=str(campaign_dir))
    for job, (name, pending) in enumerate([("stuck", True), ("healthy", False)], start=41):
        state.experiments.append(
            ExperimentStatus(
                name=name,
                config_path=f"{name}.yaml",
                slurm_job_id=job,
                status="running",
                output_dir=str(_run(campaign_dir, name, pending=pending)),
            )
        )
    state.save()
    orch = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True)
    orch.slurm.query_batch_status = lambda ids: {  # type: ignore[method-assign]
        jid: JobStatus(job_id=jid, state=slurm_state) for jid in ids
    }
    return campaign_dir, orch


@pytest.mark.parametrize("slurm_state", ["COMPLETED", "UNKNOWN"])
def test_a_pending_conversion_marks_its_arm_and_the_poll_goes_on(
    tmp_path, caplog, slurm_state
) -> None:
    """COMPLETED reaches it through result discovery, a vanished job through the artifact probe."""
    campaign_dir, orch = _campaign(tmp_path, slurm_state)
    polls = 2 if slurm_state == "UNKNOWN" else 1
    with caplog.at_level(logging.WARNING):
        for _ in range(polls):
            state = orch.check_progress(str(campaign_dir))

    stuck, healthy = state.get_experiment("stuck"), state.get_experiment("healthy")
    assert stuck.status == "conversion_pending"
    assert "convert_checkpoints_v1.py" in stuck.error_message
    assert healthy.status == "completed" and healthy.checkpoint_path is not None
    assert any("stuck" in r.getMessage() and "conversion" in r.getMessage() for r in caplog.records)

    saved = CampaignState.load(campaign_dir / "campaign_state.json")
    assert saved.get_experiment("stuck").status == "conversion_pending"
    assert saved.get_experiment("healthy").status == "completed"
    assert "convert_checkpoints_v1.py" in saved.summary_table()


def test_the_arm_recovers_on_the_next_poll_once_converted(tmp_path) -> None:
    campaign_dir, orch = _campaign(tmp_path, "COMPLETED")
    orch.check_progress(str(campaign_dir))
    saved = CampaignState.load(campaign_dir / "campaign_state.json")
    (Path(saved.get_experiment("stuck").output_dir) / "checkpoints" / CONVERSION_JOURNAL).unlink()

    stuck = orch.check_progress(str(campaign_dir)).get_experiment("stuck")
    assert stuck.status == "completed"
    assert stuck.checkpoint_path is not None
    assert stuck.error_message is None


def test_a_pending_arm_is_not_resubmitted_on_resume(tmp_path) -> None:
    campaign_dir, orch = _campaign(tmp_path, "COMPLETED")
    orch.check_progress(str(campaign_dir))
    resuming = CampaignOrchestrator(base_dir=str(tmp_path), dry_run=True, resume=True)
    assert "stuck" in resuming._prior_handled(campaign_dir)


def test_a_pending_arm_is_not_terminal() -> None:
    """It must be polled again, or it would never be rediscovered after conversion."""
    exp = ExperimentStatus(name="a", config_path="a.yaml", status="conversion_pending")
    assert not exp.is_terminal
