"""``spectramr.main`` applies SPECTRAMR_CPU_THREADS at import.

The pin runs before ``import torch`` and overwrites the three ``*_NUM_THREADS``
variables, so it is checked in a fresh interpreter with the production import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import spectramr
from spectramr.core.resources import ALLOC_CPU_ENV

pytestmark = pytest.mark.unit

_SCHEDULER_ENV = (*ALLOC_CPU_ENV, "LOCAL_WORLD_SIZE")

_PROBE = (
    "import os, spectramr.main, torch; "
    "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], "
    "os.environ['OPENBLAS_NUM_THREADS'], torch.get_num_threads())"
)


def _import_main(**env: str) -> subprocess.CompletedProcess[str]:
    child_env = {
        k: v
        for k, v in os.environ.items()
        if k != "SPECTRAMR_CPU_THREADS" and k not in _SCHEDULER_ENV
    }
    child_env["PYTHONPATH"] = str(Path(spectramr.__file__).resolve().parents[1])
    child_env["SPECTRAMR_SUPPRESS_CLINICAL_WARNING"] = "1"
    child_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_default_pins_every_library_to_one_thread() -> None:
    result = _import_main()
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-4:] == ["1", "1", "1", "1"]


def test_slurm_grant_reaches_the_env_vars_and_torch() -> None:
    result = _import_main(SLURM_CPUS_PER_TASK="2")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-4:] == ["2", "2", "2", "2"]


def test_override_reaches_the_env_vars_and_torch() -> None:
    result = _import_main(SPECTRAMR_CPU_THREADS="3")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-4:] == ["3", "3", "3", "3"]


def test_a_malformed_override_fails_the_import() -> None:
    result = _import_main(SPECTRAMR_CPU_THREADS="lots")
    assert result.returncode != 0
    assert "SPECTRAMR_CPU_THREADS" in result.stderr
