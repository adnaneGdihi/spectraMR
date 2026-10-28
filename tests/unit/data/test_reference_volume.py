"""Suffix dispatch and NIfTI geometry for whole reference volumes."""

from __future__ import annotations

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
h5py = pytest.importorskip("h5py")

from spectramr.data.reference_volume import (  # noqa: E402
    is_nifti_path,
    load_reference_volume,
    read_nifti_sidecar,
    sidecar_path_for,
)


def _write_nifti(path, arr_hwd, zooms_hwd):
    affine = np.diag([*zooms_hwd, 1.0])
    nib.save(nib.Nifti1Image(arr_hwd.astype(np.float32), affine), str(path))
    return str(path)


def test_nifti_is_read_slice_last_and_returned_slice_first(tmp_path):
    arr = np.arange(8 * 10 * 4, dtype=np.float32).reshape(8, 10, 4)
    path = _write_nifti(tmp_path / "v.nii.gz", arr, (0.5, 0.7, 3.0))
    ref = load_reference_volume(path)
    assert tuple(ref["data"].shape) == (4, 8, 10)
    assert np.array_equal(ref["data"][2].numpy(), arr[..., 2])
    assert ref["header"] is None


def test_nifti_spacing_is_slice_row_col_from_the_affine(tmp_path):
    arr = np.zeros((8, 10, 4), dtype=np.float32)
    path = _write_nifti(tmp_path / "v.nii.gz", arr, (0.5, 0.7, 3.0))
    # Voxel axes are (row, col, slice) on disk; the tuple is (slice, row, col).
    assert load_reference_volume(path)["spacing_mm"] == pytest.approx((3.0, 0.5, 0.7))


def test_h5_dispatch_returns_header_and_no_affine_spacing(tmp_path):
    path = tmp_path / "v.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("reconstruction_rss", data=np.ones((2, 6, 6), dtype=np.float32))
    ref = load_reference_volume(str(path))
    assert tuple(ref["data"].shape) == (2, 6, 6)
    assert ref["spacing_mm"] is None
    assert ref["header"] is None  # the file carries no ISMRMRD header


def test_suffix_detection_is_case_insensitive_and_excludes_h5():
    assert is_nifti_path("A.NII.GZ")
    assert is_nifti_path("a.nii")
    assert not is_nifti_path("a.h5")


def test_sidecar_path_sits_beside_the_nifti():
    assert sidecar_path_for("/d/x.nii.gz").name == "x.json"
    assert sidecar_path_for("/d/x.nii").name == "x.json"


def test_absent_sidecar_raises_rather_than_returning_empty(tmp_path):
    path = _write_nifti(tmp_path / "v.nii.gz", np.zeros((4, 4, 2)), (1, 1, 1))
    with pytest.raises(FileNotFoundError, match="no JSON sidecar"):
        read_nifti_sidecar(path)


def test_present_sidecar_is_read(tmp_path):
    path = _write_nifti(tmp_path / "v.nii.gz", np.zeros((4, 4, 2)), (1, 1, 1))
    (tmp_path / "v.json").write_text(json.dumps({"MagneticFieldStrength": 0.064}))
    assert read_nifti_sidecar(path) == {"MagneticFieldStrength": 0.064}
