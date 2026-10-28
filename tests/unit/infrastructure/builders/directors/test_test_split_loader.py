"""The test loader serves the test manifest's files, not the training or validation ones."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from spectramr.infrastructure.builders.directors.test_split import build_test_loader

H, W = 32, 40
GROUPS = {"train": "2022_T1A", "val": "2022_T1B", "test": "2022_T1C"}


def _write_group(root: Path, stem: str) -> list[str]:
    from tests.fixtures.self_supervised_batches import phantom_kspace

    names = []
    for rep in ("01", "02", "03"):
        name = f"{stem}{rep}.h5"
        with h5py.File(root / name, "w") as f:
            f.create_dataset("kspace", data=phantom_kspace(H, W, 4, seed=int(rep))[None])
        names.append(name)
    return names


def _manifest(path: Path, root: Path, names: list[str]) -> str:
    path.write_text(
        json.dumps(
            {
                "data_root": str(root),
                "records": [{"relative_path": n, "file_id": n[:-3]} for n in names],
            }
        )
    )
    return str(path)


@pytest.fixture()
def arm(tmp_path):
    from tests.fixtures.self_supervised_batches import load_arm, with_shape, without_shift_store

    config = without_shift_store(with_shape(load_arm("ei_multicoil_m4raw_r4"), (H, W)))
    files = {split: _write_group(tmp_path, stem) for split, stem in GROUPS.items()}
    source = config.data.source.model_copy(
        update={
            "root": str(tmp_path),
            "index_path": _manifest(tmp_path / "train.json", tmp_path, files["train"]),
            "validation_index_path": _manifest(tmp_path / "val.json", tmp_path, files["val"]),
            "test_index_path": _manifest(tmp_path / "test.json", tmp_path, files["test"]),
        }
    )
    data = config.data.model_copy(
        update={
            "source": source,
            "coil_maps_root": None,
            "coil_maps_reference_root": None,
            "coil_maps_missing": "estimate",
        }
    )
    return config.model_copy(update={"data": data})


def test_the_test_loader_reads_the_test_files_only(arm):
    loader = build_test_loader(arm, num_workers=0, val_batch_size=1, persistent_workers=False)
    names = {Path(p).name for rec in loader.dataset.index for p in rec["paths"]}
    assert names and all(n.startswith(GROUPS["test"]) for n in names), names
