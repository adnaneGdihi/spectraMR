"""The per-arm script must launch a process-group arm through torchrun even at one GPU.

It chose the verb from the GPU count alone, so a ``parallel.strategy: deepspeed`` arm on one GPU
ran ``spectramr train``, built its model and data, and only then refused at Stage B with
"requires an initialised process group". The launch section of the generated script is executed
here against real YAML files.
"""

from __future__ import annotations

import stat
import subprocess
import sys
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


def _stubs(tmp_path: Path) -> Path:
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    python = stubs / "python"
    # A wrapper, not a symlink: a venv interpreter reached through a symlink elsewhere does
    # not find its pyvenv.cfg and loses the venv's site-packages.
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    scontrol = stubs / "scontrol"
    scontrol.write_text("#!/bin/sh\necho node0\n")
    for stub in (python, scontrol):
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stubs


def _train_cmd(
    tmp_path: Path,
    yaml_text: str,
    *,
    gpus: int = 1,
    nodes: int = 1,
    resume: bool = False,
    restart_count: int | None = None,
):
    config = tmp_path / "arm.yaml"
    config.write_text(yaml_text)
    script = SLURMBackend.generate_job_script(
        experiment_name="exp",
        config_path=str(config),
        output_dir=str(tmp_path / "out"),
        base_dir=str(tmp_path),
        slurm_params={"gpus": gpus, "nodes": nodes},
        resume=resume,
    )
    section = script[script.index("NUM_GPUS=") : script.index('echo "Config:')]
    env = {
        "PATH": f"{_stubs(tmp_path)}:/usr/bin:/bin",
        "PYTHONPATH": str(REPO / "src"),
        "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
    }
    if restart_count is not None:
        env["SLURM_RESTART_COUNT"] = str(restart_count)
    out = subprocess.run(
        ["/usr/bin/bash", "-c", f'GPUS_PER_NODE={gpus}\n{section}\necho "$TRAIN_CMD"'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return out


class TestProcessGroupArmsAtOneGpu:
    @pytest.mark.parametrize("strategy", ["deepspeed", "ddp", "fsdp"])
    def test_a_process_group_strategy_launches_train_distributed(self, tmp_path, strategy) -> None:
        """Planted: the probe arm that failed on the cluster at Stage B."""
        out = _train_cmd(tmp_path, f"parallel:\n  strategy: {strategy}\n")
        assert out.returncode == 0, out.stderr
        assert "train-distributed" in out.stdout
        assert "torchrun --nproc_per_node=1" in out.stdout

    def test_a_plain_arm_keeps_the_single_process_verb(self, tmp_path) -> None:
        out = _train_cmd(tmp_path, "parallel:\n  strategy: none\n")
        assert out.returncode == 0, out.stderr
        assert "python -m spectramr.cli train --config" in out.stdout
        assert "train-distributed" not in out.stdout

    @pytest.mark.parametrize("yaml_text", ["training:\n  max_iterations: 5\n", "parallel:\n"])
    def test_an_absent_or_null_parallel_block_is_single_process(self, tmp_path, yaml_text) -> None:
        """The shapes that broke the old bash grep."""
        out = _train_cmd(tmp_path, yaml_text)
        assert out.returncode == 0, out.stderr
        assert "train-distributed" not in out.stdout

    def test_dp_is_genuinely_single_process(self, tmp_path) -> None:
        """nn.DataParallel needs no group, so a != none shorthand would be wrong."""
        out = _train_cmd(tmp_path, "parallel:\n  strategy: dp\n")
        assert out.returncode == 0, out.stderr
        assert "train-distributed" not in out.stdout

    def test_more_gpus_still_launch_through_torchrun_without_a_strategy(self, tmp_path) -> None:
        out = _train_cmd(tmp_path, "parallel:\n  strategy: none\n", gpus=4)
        assert "torchrun --nproc_per_node=4" in out.stdout

    def test_an_unreadable_config_refuses_instead_of_guessing_the_verb(self, tmp_path) -> None:
        config = tmp_path / "missing.yaml"
        script = SLURMBackend.generate_job_script(
            experiment_name="exp",
            config_path=str(config),
            output_dir=str(tmp_path / "out"),
            base_dir=str(tmp_path),
        )
        section = script[script.index("NUM_GPUS=") : script.index('echo "Config:')]
        out = subprocess.run(
            ["/usr/bin/bash", "-c", f"GPUS_PER_NODE=1\n{section}"],
            env={
                "PATH": f"{_stubs(tmp_path)}:/usr/bin:/bin",
                "PYTHONPATH": str(REPO / "src"),
                "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 3
        assert "refusing to guess the launch verb" in out.stderr


class TestResumeSpellingPerLink:
    """The first link follows the arm's own setting; every requeued link continues the run.

    The cluster probe was generated with resume=True, so its first link used the strict
    ``--resume auto``, which raises when the directory has no checkpoint. And a fresh arm (the
    common campaign case) could not yield at all, because the yield required resume=True.
    """

    PLAIN = "parallel:\n  strategy: none\n"

    def test_a_fresh_first_link_passes_no_resume_flag(self, tmp_path) -> None:
        out = _train_cmd(tmp_path, self.PLAIN, resume=False, restart_count=0)
        assert "--resume" not in out.stdout

    def test_an_unset_restart_count_is_the_first_link(self, tmp_path) -> None:
        out = _train_cmd(tmp_path, self.PLAIN, resume=False)
        assert "--resume" not in out.stdout

    def test_an_explicit_resume_keeps_its_strict_spelling_on_the_first_link(self, tmp_path) -> None:
        out = _train_cmd(tmp_path, self.PLAIN, resume=True, restart_count=0)
        assert "--resume auto" in out.stdout

    @pytest.mark.parametrize("resume", [False, True])
    @pytest.mark.parametrize("restart", [1, 4])
    def test_a_requeued_link_resumes_if_a_checkpoint_is_present(
        self, tmp_path, resume, restart
    ) -> None:
        """Planted: a fresh script's requeued link used to restart at iteration 0."""
        out = _train_cmd(tmp_path, self.PLAIN, resume=resume, restart_count=restart)
        assert "--resume if-present" in out.stdout
        assert "--resume auto" not in out.stdout

    def test_the_requeued_spelling_reaches_a_process_group_launch_too(self, tmp_path) -> None:
        out = _train_cmd(
            tmp_path, "parallel:\n  strategy: deepspeed\n", resume=False, restart_count=2
        )
        assert "train-distributed" in out.stdout and "--resume if-present" in out.stdout
