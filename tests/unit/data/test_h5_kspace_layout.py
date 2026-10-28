"""The slice-addressable HDF5 layout for k-space that training reads one slice at a time."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.contrast_registration import write_kspace_copy
from spectramr.data.h5_kspace_layout import rechunk_file, slice_chunked_kwargs

SHAPE = (6, 4, 16, 16)


def _kspace(shape=SHAPE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _write_m4raw_like(path: Path, data: np.ndarray, *, chunks, compression="gzip") -> None:
    """The released M4Raw layout: one chunk is a single k-space column across all slices."""
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "kspace",
            data=data,
            chunks=chunks,
            compression=compression,
            compression_opts=4 if compression else None,
        )
        ds.attrs["kspace_note"] = "dataset attribute"
        f.create_dataset("ismrmrd_header", data=np.bytes_("<header/>"))
        f.create_dataset("reconstruction_rss", data=np.ones((6, 16, 16), np.float32))
        f.attrs["acquisition"] = "T101"


def test_one_chunk_is_one_slice() -> None:
    kwargs = slice_chunked_kwargs(SHAPE)
    assert kwargs["chunks"] == (1, 4, 16, 16)
    assert kwargs["compression"] == "gzip" and kwargs["compression_opts"] == 1
    assert kwargs["shuffle"] is True


def test_real_valued_storage_with_a_trailing_pair_is_chunked_per_slice() -> None:
    assert slice_chunked_kwargs((6, 4, 16, 16, 2))["chunks"] == (1, 4, 16, 16, 2)


def test_a_shape_without_a_slice_axis_raises() -> None:
    with pytest.raises(ValueError, match="slice axis"):
        slice_chunked_kwargs((16, 16))


def test_rechunk_is_lossless_and_keeps_every_other_dataset_and_attribute(tmp_path: Path) -> None:
    data = _kspace()
    src, dst = tmp_path / "src.h5", tmp_path / "dst.h5"
    _write_m4raw_like(src, data, chunks=(6, 4, 16, 1))
    rechunk_file(src, dst)
    with h5py.File(dst, "r") as f:
        assert f["kspace"].chunks == (1, 4, 16, 16)
        np.testing.assert_array_equal(f["kspace"][()], data)
        assert f["kspace"].attrs["kspace_note"] == "dataset attribute"
        assert bytes(f["ismrmrd_header"][()]) == b"<header/>"
        assert f["reconstruction_rss"].shape == (6, 16, 16), "originals keep their images"
        assert f.attrs["acquisition"] == "T101"


def test_rechunk_reads_a_slice_without_touching_the_others(tmp_path: Path) -> None:
    src, dst = tmp_path / "src.h5", tmp_path / "dst.h5"
    _write_m4raw_like(src, _kspace(), chunks=(6, 4, 16, 1))
    rechunk_file(src, dst)
    with h5py.File(dst, "r") as f:
        ds = f["kspace"]
        # One chunk per slice: the dataset's storage is exactly n_slices chunks.
        assert ds.id.get_num_chunks() == SHAPE[0]
        assert ds[3].shape == SHAPE[1:]


def test_rechunk_handles_uncompressed_contiguous_sources(tmp_path: Path) -> None:
    data = _kspace()
    src, dst = tmp_path / "src.h5", tmp_path / "dst.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("kspace", data=data)
    rechunk_file(src, dst)
    with h5py.File(dst, "r") as f:
        assert f["kspace"].chunks == (1, 4, 16, 16)
        np.testing.assert_array_equal(f["kspace"][()], data)


def test_rechunk_refuses_a_file_without_kspace(tmp_path: Path) -> None:
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("other", data=np.zeros(3))
    with pytest.raises(KeyError, match="kspace"):
        rechunk_file(src, tmp_path / "dst.h5")


def test_derived_trees_are_written_slice_chunked(tmp_path: Path) -> None:
    """NEX averages and registered copies come out in the layout training reads."""
    data = _kspace()
    src, dst = tmp_path / "src.h5", tmp_path / "dst.h5"
    _write_m4raw_like(src, data, chunks=(6, 4, 16, 1))
    write_kspace_copy(src, dst, torch.from_numpy(data), {"nex_average_count": 2})
    with h5py.File(dst, "r") as f:
        assert f["kspace"].chunks == (1, 4, 16, 16)
        assert "reconstruction_rss" not in f, "derived images are still dropped"
        np.testing.assert_array_equal(f["kspace"][()], data)


def test_rechunk_split_mirrors_a_directory_and_links_what_it_does_not_convert(
    tmp_path: Path,
) -> None:
    from spectramr.data.h5_kspace_layout import rechunk_split

    src, dst = tmp_path / "split", tmp_path / "out" / "split"
    src.mkdir()
    for name in ("s1_T101.h5", "s1_T102.h5"):
        _write_m4raw_like(src / name, _kspace(seed=len(name)), chunks=(6, 4, 16, 1))
    (src / "notes.txt").write_text("keep me")
    records = rechunk_split(src, dst)
    assert [r["status"] for r in records] == ["rechunked", "rechunked"]
    assert (dst / "notes.txt").is_symlink() and (dst / "notes.txt").read_text() == "keep me"
    with h5py.File(dst / "s1_T101.h5", "r") as f:
        assert f["kspace"].chunks == (1, 4, 16, 16)


def test_rechunk_split_flags_a_file_whose_content_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted corruption must turn the verification red, not pass as a conversion."""
    from spectramr.data import h5_kspace_layout as layout

    src, dst = tmp_path / "split", tmp_path / "out"
    src.mkdir()
    _write_m4raw_like(src / "s1_T101.h5", _kspace(), chunks=(6, 4, 16, 1))
    real = layout.rechunk_file

    def corrupting(source: Path, dest: Path) -> None:
        real(source, dest)
        with h5py.File(dest, "r+") as f:
            f["kspace"][0, 0, 0, 0] = 99 + 0j

    monkeypatch.setattr(layout, "rechunk_file", corrupting)
    records = layout.rechunk_split(src, dst)
    assert records[0]["status"].startswith("mismatch: kspace")


def test_rechunk_split_flags_a_dropped_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spectramr.data import h5_kspace_layout as layout

    src, dst = tmp_path / "split", tmp_path / "out"
    src.mkdir()
    _write_m4raw_like(src / "s1_T101.h5", _kspace(), chunks=(6, 4, 16, 1))
    real = layout.rechunk_file

    def dropping(source: Path, dest: Path) -> None:
        real(source, dest)
        with h5py.File(dest, "r+") as f:
            del f["reconstruction_rss"]

    monkeypatch.setattr(layout, "rechunk_file", dropping)
    assert "reconstruction_rss" in layout.rechunk_split(src, dst)[0]["status"]
