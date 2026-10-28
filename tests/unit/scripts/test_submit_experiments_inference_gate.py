"""The job ``submit_experiments.sh`` renders evaluates only a run that trained to completion.

The heredoc is rendered with ``cat`` in place of ``sbatch`` and run against a stub
``python`` that logs each call, so no training, resolver or inference really runs.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/training/submit_experiments.sh"


def _heredoc() -> str:
    lines = SCRIPT.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("sbatch <<EOF"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "EOF")
    return "\n".join(["cat <<EOF", *lines[start + 1 : end], "EOF"]) + "\n"


def _render(tmp_path: Path, run: Path) -> Path:
    renderer = tmp_path / "render.sh"
    renderer.write_text(
        f'OUTPUT_DIR="{run}"; CONFIG_TO_USE=c.yaml; EXPERIMENT_NAME=e\n' + _heredoc()
    )
    job = tmp_path / "job.sh"
    job.write_text(
        subprocess.run(["bash", str(renderer)], capture_output=True, text=True, check=True).stdout
    )
    return job


def _stub_python(tmp_path: Path, run: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> "{tmp_path}/calls.log"\n'
        'case "$*" in\n'
        '  *"spectramr.cli train"*) exit "${STUB_TRAIN_RC:-0}";;\n'
        f'  *resolve_checkpoint*) echo "{run}/checkpoints/step_00000010.pt";;\n'
        "esac\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bin_dir


@pytest.mark.parametrize(("train_rc", "infers"), [(1, False), (0, True)])
def test_inference_runs_only_after_a_successful_training(tmp_path, train_rc, infers) -> None:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "step_00000010.pt").write_bytes(b"x")
    (run / "test_split_manifest.txt").write_text("subject\n")
    job = _render(tmp_path, run)
    bin_dir = _stub_python(tmp_path, run)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "STUB_TRAIN_RC": str(train_rc),
    }
    proc = subprocess.run(["bash", str(job)], env=env, capture_output=True, text=True)
    calls = (tmp_path / "calls.log").read_text()
    assert proc.returncode == train_rc
    assert ("spectramr.cli infer" in calls) is infers
    if infers:
        assert f"--checkpoint {run}/checkpoints/step_00000010.pt" in calls
