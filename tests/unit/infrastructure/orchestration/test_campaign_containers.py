"""Campaign-in-containers: per-arm execution via ``where`` (punch-list #1, full).

The campaign orchestrator gains a ``where`` knob — ``slurm`` (default, sbatch) or
``docker`` / ``apptainer`` (run each arm in a container, synchronously). SLURM is
the unchanged default (covered by the golden-file test); these tests pin the
container dispatch + the validation/guards.
"""

from __future__ import annotations

import pytest

import spectramr.infrastructure.execution.backends as backends_mod
from spectramr.infrastructure.orchestration.campaign_orchestrator import (
    CampaignOrchestrator,
)


@pytest.fixture(autouse=True)
def _slurm_account(monkeypatch):
    """Supply the SLURM allocation these tests used to inherit from a literal.

    ``ResourceSpec`` ships no account default -- an allocation name is
    site-specific, so the tree carries none (#1146). Tests that render SLURM
    directives must therefore configure one, exactly as a user does.
    """
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)


def test_rejects_unknown_where():
    with pytest.raises(ValueError, match="where"):
        CampaignOrchestrator(where="nonsense")


def test_default_where_is_slurm():
    assert CampaignOrchestrator().where == "slurm"


def test_run_arm_in_container_dry_run(tmp_path):
    orch = CampaignOrchestrator(where="docker", dry_run=True)
    status = orch._run_arm_in_container(
        name="arm1",
        cfg_path="experiments/inprogress/x.yaml",
        exp_output=tmp_path,
        slurm_params={"gpus": 1},
        role="variant",
    )
    # dry-run does not actually run the container.
    assert status.status == "submitted"
    assert status.name == "arm1"
    assert status.output_dir == str(tmp_path)


def test_run_arm_in_container_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backends_mod.DockerBackend,
        "run",
        lambda self, inv, res, *, dry_run=False: backends_mod.RunHandle(
            "docker", returncode=0, command=["docker", "run", "..."]
        ),
    )
    status = CampaignOrchestrator(where="docker")._run_arm_in_container(
        name="a", cfg_path="x.yaml", exp_output=tmp_path, slurm_params={}, role="v"
    )
    assert status.status == "completed"


def test_run_arm_in_container_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backends_mod.DockerBackend,
        "run",
        lambda self, inv, res, *, dry_run=False: backends_mod.RunHandle("docker", returncode=1),
    )
    status = CampaignOrchestrator(where="docker")._run_arm_in_container(
        name="a", cfg_path="x.yaml", exp_output=tmp_path, slurm_params={}, role="v"
    )
    assert status.status == "failed"
    assert "exited 1" in (status.error_message or "")


def test_run_arm_passes_output_dir_override(monkeypatch, tmp_path):
    seen = {}

    def fake_run(self, inv, res, *, dry_run=False):
        seen["args"] = inv.to_cli_args()
        return backends_mod.RunHandle("docker", returncode=0)

    monkeypatch.setattr(backends_mod.DockerBackend, "run", fake_run)
    CampaignOrchestrator(where="docker")._run_arm_in_container(
        name="a", cfg_path="x.yaml", exp_output=tmp_path, slurm_params={}, role="v"
    )
    # the arm runs `train --config x.yaml -O training.output_dir=<dir>` so its
    # checkpoints land in the campaign tree.
    assert "train" in seen["args"]
    assert f"training.output_dir={tmp_path}" in " ".join(seen["args"])


def test_submit_one_arm_slurm_uses_generate_job_script(tmp_path):
    # where=slurm + dry_run -> the existing SLURM path (submit_job dry-run -> 0).
    orch = CampaignOrchestrator(where="slurm", dry_run=True)
    status = orch._submit_one_arm(
        name="a",
        cfg_path="x.yaml",
        exp_output=tmp_path,
        slurm_params={"gpus": 1},
        test_manifest_path=None,
        role="v",
    )
    assert status.status == "submitted"
    assert status.slurm_job_id == 0  # dry-run submit_job returns 0


@pytest.mark.parametrize(
    ("flags", "resumes"), [({}, False), ({"fresh": True}, False), ({"resume": True}, True)]
)
def test_container_and_per_arm_paths_agree_on_resume(monkeypatch, tmp_path, flags, resumes):
    """Only --resume spells a checkpoint resume on the non-array paths (#2542)."""
    seen = {}

    def fake_run(self, inv, res, *, dry_run=False):
        seen["args"] = inv.to_cli_args()
        return backends_mod.RunHandle("docker", returncode=0)

    monkeypatch.setattr(backends_mod.DockerBackend, "run", fake_run)
    CampaignOrchestrator(where="docker", **flags)._run_arm_in_container(
        name="a", cfg_path="x.yaml", exp_output=tmp_path, slurm_params={}, role="v"
    )
    assert ("--resume" in seen["args"]) is resumes

    orch = CampaignOrchestrator(where="slurm", dry_run=True, **flags)
    scripts: list[str] = []
    orch.slurm.submit_job = lambda script, **kw: scripts.append(script) or 0  # type: ignore[method-assign]
    orch._submit_one_arm(
        name="a",
        cfg_path="x.yaml",
        exp_output=tmp_path,
        slurm_params={},
        test_manifest_path=None,
        role="v",
    )
    assert ("--resume auto" in scripts[0]) is resumes
