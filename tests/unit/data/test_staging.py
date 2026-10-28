"""``data.staging``: the node-local copy is complete, current, shared safely and never silent."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from spectramr.config.schemas.data import DataStagingConfigSchema
from spectramr.data.staging import (
    StagingNoSpaceError,
    stage_files,
    stage_index_records,
    stage_path_lists,
    stage_subtree,
)


def _cfg(root: Path, **kw) -> DataStagingConfigSchema:
    return DataStagingConfigSchema(mode="node_local", root=str(root), min_free_gb=0.0, **kw)


def _tree(base: Path, n: int = 4) -> list[Path]:
    files = []
    for i in range(n):
        path = base / "src" / f"vol_{i}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=np.full((2, 4, 8, 8), i + 1, dtype=np.complex64))
        files.append(path)
    return files


def test_off_is_the_default_and_changes_nothing(tmp_path: Path) -> None:
    files = _tree(tmp_path)
    assert DataStagingConfigSchema().mode == "off"
    (out,), report = stage_path_lists(DataStagingConfigSchema(), files)
    assert out == files and report is None


def test_unknown_key_is_refused() -> None:
    with pytest.raises(ValueError):
        DataStagingConfigSchema(mod="node_local")


def test_copy_is_byte_identical_and_second_call_reuses_it(tmp_path: Path) -> None:
    files = _tree(tmp_path)
    cfg = _cfg(tmp_path / "local")
    mapping, first = stage_files(files, cfg)
    assert first.copied == len(files) and first.reused == 0
    for src in files:
        assert mapping[src].read_bytes() == src.read_bytes()
        assert not str(mapping[src]).startswith(str(src.parent))
    _, again = stage_files(files, cfg)
    assert again.copied == 0 and again.reused == len(files)


def test_a_changed_source_is_copied_again(tmp_path: Path) -> None:
    (src,) = _tree(tmp_path, 1)
    cfg = _cfg(tmp_path / "local")
    mapping, _ = stage_files([src], cfg)
    with h5py.File(src, "w") as f:
        f.create_dataset("kspace", data=np.zeros((3, 4, 8, 8), dtype=np.complex64))
    mapping, report = stage_files([src], cfg)
    assert report.copied == 1
    assert mapping[src].read_bytes() == src.read_bytes()


def test_a_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="do not exist"):
        stage_files([tmp_path / "nope.h5"], _cfg(tmp_path / "local"))


def test_no_space_raises_by_default_and_never_reads_in_place_silently(tmp_path: Path) -> None:
    files = _tree(tmp_path)
    cfg = DataStagingConfigSchema(mode="node_local", root=str(tmp_path / "l"), min_free_gb=1e9)
    with pytest.raises(StagingNoSpaceError):
        stage_files(files, cfg)


def test_read_in_place_is_an_explicit_loud_choice(tmp_path: Path, caplog) -> None:
    files = _tree(tmp_path)
    cfg = DataStagingConfigSchema(
        mode="node_local", root=str(tmp_path / "l"), min_free_gb=1e9, when_full="read_in_place"
    )
    with caplog.at_level("WARNING"):
        mapping, report = stage_files(files, cfg)
    assert report.in_place and all(mapping[f] == f for f in files)
    assert "in place" in caplog.text


_WORKER = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from spectramr.config.schemas.data import DataStagingConfigSchema
    from spectramr.data.staging import stage_files
    root, files = sys.argv[1], [Path(p) for p in json.loads(sys.argv[2])]
    cfg = DataStagingConfigSchema(mode="node_local", root=root, min_free_gb=0.0)
    _, report = stage_files(files, cfg)
    print(json.dumps({"copied": report.copied, "reused": report.reused}))
    """
)


def test_concurrent_stagers_copy_each_file_exactly_once(tmp_path: Path) -> None:
    files = _tree(tmp_path, 6)
    args = [
        sys.executable,
        "-c",
        _WORKER,
        str(tmp_path / "local"),
        json.dumps([str(f) for f in files]),
    ]
    procs = [subprocess.Popen(args, stdout=subprocess.PIPE, text=True) for _ in range(4)]
    reports = [json.loads(p.communicate()[0]) for p in procs]
    assert all(p.returncode == 0 for p in procs)
    assert sum(r["copied"] for r in reports) == len(files)
    assert sum(r["reused"] for r in reports) == 3 * len(files)
    mapping, _ = stage_files(files, _cfg(tmp_path / "local"))
    assert all(mapping[f].read_bytes() == f.read_bytes() for f in files)


_SLOW = textwrap.dedent(
    """
    import json, shutil, sys, time
    from pathlib import Path
    import spectramr.data.staging as staging
    real = shutil.copyfile
    def slow(src, dst):
        real(src, dst)
        Path(sys.argv[3]).write_text("copying")
        time.sleep(60)
    staging.shutil.copyfile = slow
    from spectramr.config.schemas.data import DataStagingConfigSchema
    cfg = DataStagingConfigSchema(mode="node_local", root=sys.argv[1], min_free_gb=0.0, threads=1)
    staging.stage_files([Path(p) for p in json.loads(sys.argv[2])], cfg)
    """
)


def test_a_stager_killed_mid_copy_leaves_nothing_the_next_one_trusts(tmp_path: Path) -> None:
    files = _tree(tmp_path, 3)
    flag = tmp_path / "flag"
    root = tmp_path / "local"
    proc = subprocess.Popen(
        [sys.executable, "-c", _SLOW, str(root), json.dumps([str(f) for f in files]), str(flag)]
    )
    deadline = time.time() + 30
    while not flag.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert flag.exists(), "the slow stager never started copying"
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    partials = list(root.rglob("*.partial.*"))
    assert partials, "the kill left no partial, so the test did not exercise recovery"
    mapping, report = stage_files(files, _cfg(root))
    assert report.copied == len(files)
    assert not list(root.rglob("*.partial.*"))
    assert all(mapping[f].read_bytes() == f.read_bytes() for f in files)


def test_stage_subtree_copies_only_the_named_files_and_keeps_absent_ones_absent(
    tmp_path: Path,
) -> None:
    maps = tmp_path / "maps"
    maps.mkdir()
    for name in ("a.pt", "b.pt", "unused.pt"):
        (maps / name).write_bytes(name.encode())
    staged = Path(stage_subtree(_cfg(tmp_path / "local"), maps, ["a.pt", "b.pt", "gone.pt"]))
    assert staged != maps
    assert sorted(p.name for p in staged.iterdir()) == ["a.pt", "b.pt"]
    assert (staged / "a.pt").read_bytes() == b"a.pt"


def test_stage_subtree_off_returns_the_root(tmp_path: Path) -> None:
    assert stage_subtree(DataStagingConfigSchema(), tmp_path, ["a.pt"]) == str(tmp_path)


def test_index_records_are_copied_not_edited_and_only_hdf5_is_staged(tmp_path: Path) -> None:
    files = _tree(tmp_path, 2)
    nii = tmp_path / "scan.nii.gz"
    nii.write_bytes(b"n")
    train = [{"primary_path": str(files[0]), "slice_index": 3}, {"primary_path": str(nii)}]
    val = [{"primary_path": str(files[1]), "slice_index": 0}]
    new_train, new_val = stage_index_records(_cfg(tmp_path / "local"), train, val)
    assert train[0]["primary_path"] == str(files[0])
    assert new_train[0]["slice_index"] == 3
    assert new_train[0]["primary_path"].startswith(str(tmp_path / "local"))
    assert new_train[1]["primary_path"] == str(nii)
    assert Path(new_val[0]["primary_path"]).read_bytes() == files[1].read_bytes()
    assert stage_index_records(DataStagingConfigSchema(), train)[0] == train
