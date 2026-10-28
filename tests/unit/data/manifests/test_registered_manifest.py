"""Unit tests for spectramr.data.manifests.registered_manifest.

A manifest of the mirror is judged by the loader's own resolver: every record must
exist and every registered reference must land on its copy inside the mirror.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectramr.data.manifests.registered_manifest import verify_registered_manifest

REF, OTHER = "2022000001_T101.h5", "2022000001_T201.h5"


def _trees(tmp_path: Path, *, register_reference: bool = True) -> tuple[Path, Path]:
    """The layout register_split leaves: the reference written, everything else linked."""
    src, mirror = tmp_path / "src", tmp_path / "mirror"
    src.mkdir()
    mirror.mkdir()
    for name in (REF, OTHER):
        (src / name).touch()
    if register_reference:
        (mirror / REF).touch()
    else:
        (mirror / REF).symlink_to(src / REF)
    (mirror / OTHER).symlink_to(src / OTHER)
    return src, mirror


def _manifest(tmp_path: Path, data_root: Path) -> Path:
    records = [{"file_id": Path(n).stem, "relative_path": n} for n in (REF, OTHER)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"data_root": str(data_root), "records": records}))
    return path


def test_a_manifest_of_the_mirror_passes(tmp_path: Path) -> None:
    _, mirror = _trees(tmp_path)
    assert verify_registered_manifest(_manifest(tmp_path, mirror), mirror, {REF}) == 2


def test_a_reference_that_still_reaches_its_original_is_refused(tmp_path: Path) -> None:
    _, mirror = _trees(tmp_path, register_reference=False)
    with pytest.raises(ValueError, match="1 of 2 records"):
        verify_registered_manifest(_manifest(tmp_path, mirror), mirror, {REF})


def test_a_manifest_of_the_source_tree_is_refused(tmp_path: Path) -> None:
    src, mirror = _trees(tmp_path)
    with pytest.raises(ValueError, match="1 of 2 records"):
        verify_registered_manifest(_manifest(tmp_path, src), mirror, {REF})


def test_a_missing_record_is_refused(tmp_path: Path) -> None:
    _, mirror = _trees(tmp_path)
    (mirror / OTHER).unlink()
    with pytest.raises(ValueError, match="1 of 2 records"):
        verify_registered_manifest(_manifest(tmp_path, mirror), mirror, {REF})


def test_a_registered_reference_both_missing_and_outside_is_counted_once(tmp_path: Path) -> None:
    _, mirror = _trees(tmp_path)
    # A data_root that does not exist, as when the manifest resolves from the wrong cwd.
    with pytest.raises(ValueError, match="2 of 2 records"):
        verify_registered_manifest(_manifest(tmp_path, tmp_path / "absent"), mirror, {REF})


def test_a_manifest_without_a_registered_reference_is_refused(tmp_path: Path) -> None:
    _, mirror = _trees(tmp_path)
    with pytest.raises(ValueError, match="no registered reference"):
        verify_registered_manifest(_manifest(tmp_path, mirror), mirror, {"2099000001_T101.h5"})
