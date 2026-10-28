"""Unit tests for scripts/data/extract_fastmri_brain.sbatch.

Each guard is exercised on planted archives. Both layouts the release may use
land in one flat directory per split; the DICOM tree keeps its shape. An unknown
layout, an unknown name, a missing archive and a disk below the floor each stop
the job before ``tar -x`` runs. A truncated download is salvaged: its complete
members are kept, the member cut short is removed, and the job exits 2 after the
remaining archives, including when the cut falls before the first member.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

SBATCH = Path(__file__).resolve().parents[3] / "scripts/data/extract_fastmri_brain.sbatch"
SRC = "databases/brain/fastmri"

pytestmark = pytest.mark.skipif(shutil.which("xz") is None, reason="tar -J needs xz")


def _archive(root: Path, name: str, members: dict[str, bytes]) -> Path:
    src = root / SRC
    src.mkdir(parents=True, exist_ok=True)
    stage = root / f"stage_{name}"
    for member, data in members.items():
        (stage / member).parent.mkdir(parents=True, exist_ok=True)
        (stage / member).write_bytes(data)
    mode = "w:gz" if name.endswith(".gz") else "w:xz"
    with tarfile.open(src / name, mode) as tar:
        for member in members:
            tar.add(stage / member, arcname=member)
    return src / name


def _train(root: Path, batch: int, members: dict[str, bytes]) -> Path:
    return _archive(root, f"brain_multicoil_train_batch_{batch}.tar.xz", members)


def _truncate(path: Path, keep: float) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep)])


def _run(root: Path, *archives: str, floor_gb: int = 0) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(root), "FLOOR_GB": str(floor_gb)}
    return subprocess.run(
        ["bash", str(SBATCH), *archives], env=env, capture_output=True, text=True, check=False
    )


def _flat(root: Path, split: str = "multicoil_train") -> list[str]:
    return sorted(p.name for p in (root / SRC / split).iterdir())


def test_both_layouts_land_in_the_flat_directory(tmp_path: Path) -> None:
    _train(tmp_path, 0, {"multicoil_train/file_brain_AXT2_1.h5": b"a"})
    _train(tmp_path, 1, {"file_brain_AXT1_2.h5": b"b"})
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _flat(tmp_path) == ["file_brain_AXT1_2.h5", "file_brain_AXT2_1.h5"]


def test_each_split_and_the_dicom_tree_land_in_their_own_directory(tmp_path: Path) -> None:
    val = "brain_multicoil_val_batch_0.tar.xz"
    full = "brain_multicoil_test_full_batch_2.tar.xz"
    dicom = "brain_fastMRI_DICOM.tar.gz"
    _archive(tmp_path, val, {"multicoil_val/file_brain_AXT2_9.h5": b"v"})
    _archive(tmp_path, full, {"multicoil_test_full/file_brain_AXT1_8.h5": b"t"})
    _archive(tmp_path, dicom, {"fastMRI_brain_DICOM/446005329771/219.dcm": b"d"})
    result = _run(tmp_path, val, f"{SRC}/{full}", dicom)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _flat(tmp_path, "multicoil_val") == ["file_brain_AXT2_9.h5"]
    assert _flat(tmp_path, "multicoil_test_full") == ["file_brain_AXT1_8.h5"]
    assert (tmp_path / SRC / "fastMRI_brain_DICOM/446005329771/219.dcm").read_bytes() == b"d"
    members = (tmp_path / SRC / "brain_multicoil_val_batch_0.members.txt").read_text().split()
    assert members == ["multicoil_val/file_brain_AXT2_9.h5"]


def test_a_done_record_vouches_for_the_archive(tmp_path: Path) -> None:
    path = _train(tmp_path, 0, {"file_brain_AXT2_1.h5": b"a"})
    Path(f"{path}.done").write_text(f"bytes\t{path.stat().st_size}\n")
    result = _run(tmp_path, "0")
    assert result.returncode == 0
    assert "brain_multicoil_train_batch_0.tar.xz: archive complete" in result.stdout


def test_a_truncated_file_from_an_earlier_run_is_overwritten(tmp_path: Path) -> None:
    _train(tmp_path, 0, {"file_brain_AXT2_1.h5": b"complete"})
    dest = tmp_path / SRC / "multicoil_train"
    dest.mkdir(parents=True)
    (dest / "file_brain_AXT2_1.h5").write_bytes(b"trunc")
    assert _run(tmp_path, "0").returncode == 0
    assert (dest / "file_brain_AXT2_1.h5").read_bytes() == b"complete"


def test_an_unknown_layout_stops_before_extracting(tmp_path: Path) -> None:
    _train(tmp_path, 2, {"weird/file_brain_AXT2_1.h5": b"a"})
    result = _run(tmp_path, "2")
    assert result.returncode == 1
    assert "STOP: unknown directory" in result.stdout
    assert _flat(tmp_path) == []


def test_a_top_level_non_h5_entry_stops(tmp_path: Path) -> None:
    _train(tmp_path, 3, {"README.txt": b"a"})
    result = _run(tmp_path, "3")
    assert result.returncode == 1
    assert "STOP: unknown layout" in result.stdout


def test_a_missing_or_misnamed_archive_stops_before_any_extraction(tmp_path: Path) -> None:
    _train(tmp_path, 0, {"file_brain_AXT2_1.h5": b"a"})
    missing = _run(tmp_path, "0", "5")
    assert missing.returncode == 1
    assert "brain_multicoil_train_batch_5.tar.xz not found" in missing.stdout
    misnamed = _run(tmp_path, "0", "brain_multicoil_trian_batch_0.tar.xz")
    assert misnamed.returncode == 1
    assert "is not a fastMRI brain archive name" in misnamed.stdout
    assert not (tmp_path / SRC / "multicoil_train").exists()


def test_the_free_space_floor_stops_before_extracting(tmp_path: Path) -> None:
    _train(tmp_path, 0, {"file_brain_AXT2_1.h5": b"a"})
    result = _run(tmp_path, "0", floor_gb=10**9)
    assert result.returncode == 1
    assert "STOP before brain_multicoil_train_batch_0.tar.xz" in result.stdout
    assert _flat(tmp_path) == []


def test_a_truncated_archive_is_salvaged_and_the_next_one_still_runs(tmp_path: Path) -> None:
    # Random payloads defeat compression, so the cut lands inside the third member.
    payload = {f"multicoil_train/file_brain_AXT2_{i}.h5": os.urandom(400_000) for i in range(4)}
    _truncate(_train(tmp_path, 0, payload), keep=0.55)
    _train(tmp_path, 1, {"file_brain_AXT1_9.h5": b"b"})
    result = _run(tmp_path, "0", "1")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "brain_multicoil_train_batch_0.tar.xz: TRUNCATED" in result.stdout
    assert "brain_multicoil_train_batch_1.tar.xz: archive complete" in result.stdout
    assert "2 files kept, removed partial multicoil_train/file_brain_AXT2_2.h5" in result.stdout
    assert "PARTIAL: truncated archives brain_multicoil_train_batch_0.tar.xz;" in result.stdout
    assert _flat(tmp_path) == [
        "file_brain_AXT1_9.h5",
        "file_brain_AXT2_0.h5",
        "file_brain_AXT2_1.h5",
    ]
    dest = tmp_path / SRC / "multicoil_train"
    assert (dest / "file_brain_AXT2_1.h5").read_bytes() == payload[
        "multicoil_train/file_brain_AXT2_1.h5"
    ]


def test_an_archive_cut_before_its_first_member_does_not_stop_the_job(tmp_path: Path) -> None:
    empty = _train(tmp_path, 4, {"file_brain_AXT2_1.h5": os.urandom(1000)})
    empty.write_bytes(empty.read_bytes()[:64])
    _train(tmp_path, 1, {"file_brain_AXT1_9.h5": b"b"})
    result = _run(tmp_path, "4", "1")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "0 files, cut before its first member" in result.stdout
    assert _flat(tmp_path) == ["file_brain_AXT1_9.h5"]


def test_the_default_floor_is_400_gb_and_the_header_names_no_partition() -> None:
    text = SBATCH.read_text()
    assert "FLOOR_GB=${FLOOR_GB:-400}" in text
    assert "#SBATCH --partition" not in text
    assert "#SBATCH --account=johnsson" in text
    assert "--skip-old-files" not in text.split("tar -xv", 1)[1]
