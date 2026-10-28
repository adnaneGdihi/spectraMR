"""Tests for scripts/data/register_m4raw_contrasts.sbatch, run with bash as the job would.

The guards stop before any Python runs; end-to-end runs on a synthetic split walk
all five steps (repeat floor, NEX average, check, register, re-check) with the local
venv, on repetitions carrying their own receiver phase.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.fixtures import m4raw_phantom as ph

pytest.importorskip("SimpleITK")
REPO = Path(__file__).resolve().parents[3]
SBATCH = REPO / "scripts/data/register_m4raw_contrasts.sbatch"
SPLIT = "multicoil_train/multicoil_train"


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "SLURM_SUBMIT_DIR": str(REPO),
        "VENV_PATH": sys.prefix,
        "SRC_ROOT": str(tmp_path / "data"),
        "AVG_ROOT": str(tmp_path / "averaged"),
        "OUT_ROOT": str(tmp_path / "registered"),
        "SPLITS": SPLIT,
        "MAX_PAIRS": "10",
    }


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SBATCH), *args],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )


def _acquired(tmp_path: Path, split: str = SPLIT) -> Path:
    """Two repetitions per contrast, each at its own global phase; T1 displaced from T2."""
    src = tmp_path / "data" / split
    src.mkdir(parents=True)
    for rep, phase in (("01", 0.4), ("02", 2.5)):
        for contrast, image in (
            ("T1", ph.phantom(-1.0, (0.0, 3.0, -2.0))),
            ("T2", ph.phantom(1.0)),
            ("FLAIR", ph.phantom(0.7)),
        ):
            path = ph.write(src / f"2022000001_{contrast}{rep}.h5", image)
            with h5py.File(path, "a") as f:
                f["kspace"][...] = f["kspace"][()] * np.exp(1j * phase)
    return src


def _steps(stdout: str) -> dict[str, str]:
    """Each step's output, keyed by its ``n/5`` label."""
    parts = stdout.split("\n=== ")[1:]
    return {part.split()[1]: part for part in parts}


def test_all_five_steps_register_the_averages(tmp_path: Path) -> None:
    _acquired(tmp_path)
    result = _run(tmp_path, "T1", "T2")
    assert result.returncode == 0, result.stdout + result.stderr
    steps = _steps(result.stdout)
    assert list(steps) == ["1/5", "2/5", "3/5", "4/5", "5/5", "done"]
    assert "T101 vs T102" in steps["1/5"]  # the repeat floor reads the acquisitions
    averaged = sorted(p.name for p in (tmp_path / "averaged" / SPLIT).iterdir())
    assert averaged == ["2022000001_FLAIR00.h5", "2022000001_T100.h5", "2022000001_T200.h5"]
    assert "T1 vs T2" in steps["3/5"] and "|shift| px median 3.0" in steps["3/5"]
    assert "{'registered': 1}" in steps["4/5"]
    assert "T1 vs T2" in steps["5/5"] and "|shift| px median 0.0" in steps["5/5"]
    assert (tmp_path / "registered" / SPLIT / "2022000001_T100.h5").is_file()


def test_a_complete_averaged_tree_is_reused_by_the_next_pair(tmp_path: Path) -> None:
    _acquired(tmp_path)
    assert _run(tmp_path, "T1", "T2").returncode == 0
    (tmp_path / "registered").rename(tmp_path / "t1_to_t2")
    result = _run(tmp_path, "FLAIR", "T2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reusing" in _steps(result.stdout)["2/5"]


def test_an_averaged_tree_without_a_complete_sidecar_stops_the_job(tmp_path: Path) -> None:
    _acquired(tmp_path)
    (tmp_path / "averaged").mkdir()
    result = _run(tmp_path, "T1", "T2")
    assert result.returncode == 1
    assert "exists without a complete nex_averaging.json" in result.stdout
    assert not (tmp_path / "registered").exists()


def test_reference_and_target_are_required(tmp_path: Path) -> None:
    result = _run(tmp_path, "T1")
    assert result.returncode == 2 and "usage:" in result.stdout


@pytest.mark.parametrize("flag", ["--out-root=x", "--src-root", "--splits"])
def test_the_roots_and_splits_cannot_be_passed_through(tmp_path: Path, flag: str) -> None:
    result = _run(tmp_path, "T1", "T2", flag)
    assert result.returncode == 2
    assert "STOP: set SRC_ROOT/AVG_ROOT/OUT_ROOT/SPLITS" in result.stdout


def test_an_existing_output_tree_stops_the_job(tmp_path: Path) -> None:
    (tmp_path / "registered").mkdir()
    result = _run(tmp_path, "T1", "T2")
    assert result.returncode == 1 and "exists; remove it" in result.stdout


def test_the_header_targets_the_cpu_lane() -> None:
    text = SBATCH.read_text()
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --account=johnsson" in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text


def test_an_averaged_tree_missing_a_requested_split_stops_the_job(tmp_path: Path) -> None:
    _acquired(tmp_path)
    assert _run(tmp_path, "T1", "T2").returncode == 0  # averages the train split only
    (tmp_path / "registered").rename(tmp_path / "t1_to_t2")
    result = subprocess.run(
        ["bash", str(SBATCH), "T1", "T2"],
        env={**_env(tmp_path), "SPLITS": f"{SPLIT} multicoil_val/multicoil_val"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "without a complete nex_averaging.json covering" in result.stdout
    assert not (tmp_path / "registered").exists()


def test_the_check_split_defaults_to_the_first_requested_split(tmp_path: Path) -> None:
    val = "multicoil_val/multicoil_val"
    _acquired(tmp_path, val)
    result = subprocess.run(
        ["bash", str(SBATCH), "T1", "T2"],
        env={**_env(tmp_path), "SPLITS": val},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    steps = _steps(result.stdout)
    for step in ("1/5", "3/5", "5/5"):  # every check reads val; train was never requested
        header = steps[step].splitlines()[0]
        assert val in header and SPLIT not in header
    assert "|shift| px median 0.0" in steps["5/5"]


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"SPLITS": " "}, "STOP: SPLITS is empty"),
        ({"SPLITS": f"{SPLIT} {SPLIT}"}, f"STOP: SPLITS names {SPLIT} more than once"),
        ({"CHECK_SPLIT": "multicoil_val/multicoil_val"}, "is not one of SPLITS"),
    ],
)
def test_bad_splits_stop_the_job_before_any_python_runs(
    tmp_path: Path, env: dict[str, str], message: str
) -> None:
    result = subprocess.run(
        ["bash", str(SBATCH), "T1", "T2"],
        env={**_env(tmp_path), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and message in result.stdout
    assert "===" not in result.stdout
