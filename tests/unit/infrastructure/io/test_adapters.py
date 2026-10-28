"""Tests for ``spectramr.infrastructure.io.adapters``.

Focus: the intensity-normalisation contract. ``load_dicom`` / ``load_nifti``
previously assumed 8-bit ``[0, 255]`` pixel data and applied a fixed
``image / 127.5 - 1.0`` rescale, which blew real 12-/16-bit DICOM/NIfTI
intensities far outside ``[-1, 1]`` (a 4095-valued voxel mapped to ~31.1).
They now min-max rescale the true dynamic range onto ``[-1, 1]`` for any bit
depth — these tests pin that behaviour.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from spectramr.infrastructure.io.adapters import (
    _normalize_to_unit_range,
    load_dicom,
    load_h5_kspace,
    load_nifti,
    load_numpy,
)

# ---------------------------------------------------------------------------
# _normalize_to_unit_range — the shared dynamic-range helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hi", [255.0, 4095.0, 65535.0])
def test_normalize_maps_full_range_to_unit_interval(hi: float) -> None:
    """min → -1 and max → +1, regardless of bit depth (8/12/16-bit)."""
    image = np.array([[0.0, hi / 2.0], [hi, hi / 4.0]], dtype=np.float32)
    out = _normalize_to_unit_range(image)
    assert out.dtype == np.float32
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)
    # Within [-1, 1] for every element.
    assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_normalize_16bit_does_not_exceed_unit_range() -> None:
    """Regression: a 16-bit voxel must NOT map to ~31 (the old 8-bit bug)."""
    image = np.full((4, 4), 4095.0, dtype=np.float32)
    image[0, 0] = 0.0  # give it a real span
    out = _normalize_to_unit_range(image)
    assert out.max() <= 1.0 + 1e-6
    # The fixed /127.5 - 1.0 rescale would have produced ~31.1 here.
    assert out.max() < 2.0


def test_normalize_constant_image_is_zeros_not_nan() -> None:
    """A flat image (max == min) maps to all-zeros, never NaN."""
    image = np.full((3, 3), 1234.0, dtype=np.float32)
    out = _normalize_to_unit_range(image)
    assert np.all(out == 0.0)
    assert not np.isnan(out).any()


def test_normalize_preserves_relative_ordering() -> None:
    """Monotonic in the input: brighter voxels stay brighter."""
    image = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    out = _normalize_to_unit_range(image)
    assert np.all(np.diff(out) > 0)


# ---------------------------------------------------------------------------
# load_nifti — real temp NIfTI round-trip via nibabel
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, array: np.ndarray) -> None:
    import nibabel as nib

    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), str(path))


def test_load_nifti_16bit_volume_in_unit_range(tmp_path: Path) -> None:
    """A 16-bit-range NIfTI volume loads into [-1, 1] (not ~500x out)."""
    # 3D volume with values up to 32000 (well beyond 8-bit).
    vol = (np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) / 31.0) * 32000.0
    path = tmp_path / "vol16.nii.gz"
    _write_nifti(path, vol)

    out = load_nifti(path)  # no slice_index → all slices, per-volume normalised
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_load_nifti_emits_no_numpy2_copy_deprecation(tmp_path: Path) -> None:
    """np.array on nibabel's proxy passes ``copy=``, which its __array__ rejects."""
    import warnings

    path = tmp_path / "int16.nii.gz"
    _write_nifti(path, np.arange(2 * 4 * 4, dtype=np.int16).reshape(2, 4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = load_nifti(path)
    assert out.dtype == np.float32 and out.shape == (2, 4, 4)


def test_load_nifti_no_slice_index_returns_all_slices(tmp_path: Path) -> None:
    """No slice_index must return the WHOLE volume, never a silent middle slice."""
    vol = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    path = tmp_path / "vol_all.nii.gz"
    _write_nifti(path, vol)

    out = load_nifti(path)
    # All 3 slices are served (axis 0), not collapsed to the central one.
    assert out.shape == (3, 4, 4)
    # Distinct slices survive the round-trip (monotonic voxel ramp per slice).
    assert not np.allclose(out[0], out[1])


def test_load_nifti_slice_index_selects_slice(tmp_path: Path) -> None:
    """An explicit slice_index selects that slice of a 3D volume."""
    vol = np.zeros((3, 4, 4), dtype=np.float32)
    vol[0] = 1000.0
    vol[1] = 2000.0
    vol[2] = 4000.0
    path = tmp_path / "vol.nii.gz"
    _write_nifti(path, vol)

    # Slice 0 is constant → normalised to all-zeros (the constant-image branch).
    out0 = load_nifti(path, slice_index=0)
    assert out0.shape == (4, 4)
    assert np.all(out0 == 0.0)


def test_normalization_scope_differs_between_sliced_and_whole_volume(
    tmp_path: Path,
) -> None:
    """Normalization runs AFTER slice selection, so its scope is call-dependent.

    With ``slice_index`` the min/max are taken over that slice alone; without it
    they span the whole volume. The SAME slice therefore carries different values
    depending on which path served it. That is a real divergence, not a rounding
    difference, so pin it explicitly — a future refactor that hoists the
    normalization above the slicing would silently change every legacy
    ``slice_index`` caller's intensity scale, and no other test would notice.
    """
    vol = np.zeros((3, 4, 4), dtype=np.float32)
    vol[0] = 1000.0
    vol[1] = 2000.0
    vol[2] = 4000.0
    path = tmp_path / "scope.nii.gz"
    _write_nifti(path, vol)

    per_slice = load_nifti(path, slice_index=1)
    per_volume = load_nifti(path)[1]

    # Per-slice: slice 1 is constant → the constant-image branch yields zeros.
    assert np.all(per_slice == 0.0)
    # Per-volume: slice 1 sits between the 1000 and 4000 extremes → interior value.
    assert np.allclose(per_volume, (2000.0 - 1000.0) / 3000.0 * 2.0 - 1.0)
    assert not np.allclose(per_slice, per_volume)


# ---------------------------------------------------------------------------
# load_dicom — pixel_array via a stubbed pydicom (no real .dcm needed)
# ---------------------------------------------------------------------------


def test_load_dicom_16bit_in_unit_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """16-bit DICOM pixel data normalises into [-1, 1]."""
    pixels = np.array([[0, 2048], [4095, 1024]], dtype=np.uint16)

    fake_ds = types.SimpleNamespace(pixel_array=pixels)
    fake_pydicom = types.SimpleNamespace(dcmread=lambda _p: fake_ds)
    monkeypatch.setattr(
        "spectramr.infrastructure.io.adapters.get_pydicom",
        lambda: fake_pydicom,
    )

    out = load_dicom("ignored.dcm")
    assert out.dtype == np.float32
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)


def test_load_dicom_3d_returns_all_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 3D DICOM stack with no slice_index returns ALL slices, not the middle."""
    stack = np.zeros((3, 2, 2), dtype=np.uint16)
    stack[0] = 1000
    stack[1] = 4000
    stack[2] = 2000
    fake_ds = types.SimpleNamespace(pixel_array=stack)
    fake_pydicom = types.SimpleNamespace(dcmread=lambda _p: fake_ds)
    monkeypatch.setattr(
        "spectramr.infrastructure.io.adapters.get_pydicom",
        lambda: fake_pydicom,
    )

    out = load_dicom("ignored.dcm")
    # Every slice is served, not silently collapsed to shape (2, 2).
    assert out.shape == (3, 2, 2)


def test_load_dicom_slice_index_selects_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit slice_index still selects that single slice."""
    stack = np.zeros((3, 2, 2), dtype=np.uint16)
    stack[0] = 1000
    stack[1] = 4000
    stack[2] = 2000
    fake_ds = types.SimpleNamespace(pixel_array=stack)
    fake_pydicom = types.SimpleNamespace(dcmread=lambda _p: fake_ds)
    monkeypatch.setattr(
        "spectramr.infrastructure.io.adapters.get_pydicom",
        lambda: fake_pydicom,
    )

    out = load_dicom("ignored.dcm", slice_index=1)
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# load_h5_kspace — multi-slice k-space must serve every slice
# ---------------------------------------------------------------------------


def test_load_h5_kspace_no_slice_index_returns_all_slices(tmp_path: Path) -> None:
    """A 3D (S, H, W) k-space volume returns all S slices as [S, H, W, 2]."""
    h5py = pytest.importorskip("h5py")
    ksp = (
        np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        + 1j * np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    ).astype(np.complex64)
    path = tmp_path / "ksp.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=ksp)

    out = load_h5_kspace(path)
    # All 3 slices survive; real/imag on the trailing axis.
    assert out.shape == (3, 4, 4, 2)


def test_load_h5_kspace_slice_index_selects_slice(tmp_path: Path) -> None:
    """An explicit slice_index still returns a single [H, W, 2] slice."""
    h5py = pytest.importorskip("h5py")
    ksp = (
        np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        + 1j * np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    ).astype(np.complex64)
    path = tmp_path / "ksp1.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=ksp)

    out = load_h5_kspace(path, slice_index=1)
    assert out.shape == (4, 4, 2)


# ---------------------------------------------------------------------------
# load_numpy — .npy/.npz, range-aware normalisation (8-bit only for PNG/JPG)
# ---------------------------------------------------------------------------


def test_load_numpy_16bit_npy_in_unit_range(tmp_path: Path) -> None:
    """A raw 16-bit .npy is min-max rescaled to [-1, 1], not /127.5 - 1.0."""
    arr = np.array([[0.0, 2048.0], [4095.0, 1024.0]], dtype=np.float32)
    path = tmp_path / "raw16.npy"
    np.save(path, arr)

    out = load_numpy(path)
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)
    # The old 8-bit /127.5 - 1.0 rescale would have produced ~31.1 here.
    assert out.max() < 2.0


def test_load_numpy_16bit_npz_in_unit_range(tmp_path: Path) -> None:
    """A raw 16-bit .npz (first array) is min-max rescaled to [-1, 1]."""
    arr = np.linspace(0.0, 32000.0, 16, dtype=np.float32).reshape(4, 4)
    path = tmp_path / "raw16.npz"
    np.savez(path, arr)

    out = load_numpy(path)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_load_numpy_unit_interval_array_affine_mapped(tmp_path: Path) -> None:
    """Data already in [0, 1] is affine-mapped to [-1, 1] (unchanged behaviour)."""
    arr = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    path = tmp_path / "unit.npy"
    np.save(path, arr)

    out = load_numpy(path)
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)


def test_load_numpy_already_normalized_left_untouched(tmp_path: Path) -> None:
    """Signed data already within [-1, 1] is not re-scaled (no double-norm)."""
    arr = np.array([[-0.5, 0.0], [0.8, -0.2]], dtype=np.float32)
    path = tmp_path / "norm.npy"
    np.save(path, arr)

    out = load_numpy(path)
    assert np.allclose(out, arr)


def test_load_numpy_empty_npz_raises(tmp_path: Path) -> None:
    """An .npz with no arrays raises rather than loading garbage."""
    path = tmp_path / "empty.npz"
    np.savez(path)  # no arrays
    with pytest.raises(ValueError, match="no arrays"):
        load_numpy(path)
