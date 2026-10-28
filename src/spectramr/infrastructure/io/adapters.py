"""IO Adapters for Medical Imaging
=============================

Low-level adapters for loading various medical imaging formats.
Encapsulates library dependencies (nibabel, pydicom, h5py) and
provides standardized output formatting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from spectramr.domain.entities.optional_imports import get_h5py, get_nibabel, get_pydicom


def _normalize_to_unit_range(image: np.ndarray) -> np.ndarray:
    """Rescale an arbitrary-bit-depth intensity image to ``[-1, 1]``.

    Uses per-image min-max scaling rather than the fixed ``image / 127.5 - 1.0``
    (which assumes 8-bit ``[0, 255]`` data). Real DICOM/NIfTI pixel data is
    commonly 12- or 16-bit, so the fixed rescale blew those intensities far
    outside ``[-1, 1]`` (e.g. a 4095-valued voxel mapped to ~31.1). Min-max
    maps the true dynamic range onto ``[-1, 1]`` for any bit depth, and reduces
    to the old behaviour when the data already spans ``[0, 255]``.

    A constant image (``max == min``) maps to all-zeros (the midpoint of the
    target range) rather than producing ``NaN`` from a divide-by-zero.

    Args:
        image: Real-valued intensity array of any shape.

    Returns:
        ``np.float32`` array with the same shape, values in ``[-1, 1]``.
    """
    data_min = float(image.min())
    data_max = float(image.max())
    span = data_max - data_min
    if span <= 0.0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - data_min) / span * 2.0 - 1.0).astype(np.float32)


def load_dicom(file_path: str | Path, slice_index: int | None = None) -> np.ndarray:
    """Load DICOM file.

    Args:
        file_path: Path to DICOM file
        slice_index: Optional slice index for 3D volumes

    Returns:
        np.ndarray: ``[H, W]`` when ``slice_index`` is given, otherwise the whole
        ``[S, H, W]`` volume. Normalized to [-1, 1] (handles 8/12/16-bit pixel
        data — see :func:`_normalize_to_unit_range`).

    Note:
        Normalization runs AFTER slice selection, so the scope differs by call:
        per-slice min/max with ``slice_index``, per-volume min/max without it.
        The same slice therefore scales differently depending on which path
        served it — pin ``slice_index`` if you need the per-slice convention.
    """
    pydicom = get_pydicom()
    ds = pydicom.dcmread(str(file_path))
    image = ds.pixel_array.astype(np.float32)

    # Select a single slice only when explicitly requested; otherwise serve the
    # whole volume (all slices) rather than silently collapsing to the middle one
    # (that silent drop trained/inferred on one slice per volume — CLAUDE.md #9).
    if image.ndim == 3 and slice_index is not None:
        image = image[slice_index]

    # Normalize to [-1, 1] using the true intensity range (not an 8-bit assumption)
    image = _normalize_to_unit_range(image)
    return image


def load_nifti(file_path: str | Path, slice_index: int | None = None) -> np.ndarray:
    """Load NIfTI file.

    Args:
        file_path: Path to NIfTI file
        slice_index: Optional slice index for 3D volumes

    Returns:
        np.ndarray: ``[H, W]`` when ``slice_index`` is given, otherwise the whole
        ``[S, H, W]`` volume. Normalized to [-1, 1] (handles 8/12/16-bit voxel
        data — see :func:`_normalize_to_unit_range`).

    Note:
        Normalization runs AFTER slice selection, so the scope differs by call:
        per-slice min/max with ``slice_index``, per-volume min/max without it.
    """
    nibabel = get_nibabel()
    img = nibabel.load(str(file_path))
    # np.array() passes copy= to the proxy's __array__, which nibabel does not accept under numpy 2.
    image = np.asarray(img.dataobj, dtype=np.float32)

    # Select a single slice only when explicitly requested; otherwise serve the
    # whole volume (all slices) rather than silently collapsing to the middle one
    # (that silent drop trained/inferred on one slice per volume — CLAUDE.md #9).
    if image.ndim == 3 and slice_index is not None:
        image = image[slice_index]

    # Normalize to [-1, 1] using the true intensity range (not an 8-bit assumption)
    image = _normalize_to_unit_range(image)
    return image


def load_h5_kspace(file_path: str | Path, slice_index: int | None = None) -> np.ndarray:
    """Load H5 file containing k-space data.

    Args:
        file_path: Path to H5 file
        slice_index: Optional slice index for 3D volumes

    Returns:
        np.ndarray: Complex k-space as ``[H, W, 2]`` (real, imaginary) when
        ``slice_index`` is given, otherwise the whole ``[S, H, W, 2]`` volume.
    """
    h5py = get_h5py()
    with h5py.File(str(file_path), "r") as f:
        # Try common k-space dataset names
        kspace_keys = ["kspace", "data", "k-space", "ksp"]
        kspace_data = None
        for key in kspace_keys:
            if key in f:
                kspace_data = f[key][()]
                break

        if kspace_data is None:
            # If no standard key, take the first dataset
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    kspace_data = f[key][()]
                    break

        if kspace_data is None:
            raise ValueError(f"No k-space dataset found in H5 file: {file_path}")

        # Convert to complex if needed
        if kspace_data.dtype == np.complex64 or kspace_data.dtype == np.complex128:
            kspace_data = np.array(kspace_data, dtype=np.complex64)
        elif kspace_data.shape[-1] == 2:
            # Already real/imag
            kspace_data = kspace_data[..., 0] + 1j * kspace_data[..., 1]
        else:
            raise ValueError(f"Unsupported k-space data format in {file_path}")

        # Select a single slice only when explicitly requested; otherwise serve
        # every slice rather than silently collapsing to the middle one
        # (that silent drop inferred on one slice per volume — CLAUDE.md #9).
        if kspace_data.ndim == 3 and slice_index is not None:
            kspace_data = kspace_data[slice_index]

        # Convert to [H, W, 2] format
        kspace_real_imag = np.stack([kspace_data.real, kspace_data.imag], axis=-1)

        return kspace_real_imag


def load_image_standard(file_path: str | Path) -> np.ndarray:
    """Load standard image (PNG, JPG, etc.).

    Args:
        file_path: Path to image file

    Returns:
        np.ndarray: Image data normalized to [-1, 1]
    """
    pil_image = Image.open(str(file_path)).convert("L")
    image = np.array(pil_image, dtype=np.float32)
    image = image / 127.5 - 1.0
    return image


def load_numpy(file_path: str | Path) -> np.ndarray:
    """Load NumPy file (.npy or .npz).

    Args:
        file_path: Path to NumPy file

    Returns:
        np.ndarray: Image data. A ``.npy``/``.npz`` array may already be
        normalized, so the range is inspected before scaling: data already in
        ``[0, 1]`` is affine-mapped to ``[-1, 1]``, data already in ``[-1, 1]``
        is left untouched, and raw out-of-range intensities (any bit depth) are
        min-max rescaled to ``[-1, 1]``. The 8-bit ``/127.5 - 1.0`` assumption
        only holds for PNG/JPG, which :func:`load_image_standard` handles.
    """
    file_path_str = str(file_path)
    if file_path_str.endswith(".npz"):
        with np.load(file_path_str) as npz_file:
            if not npz_file.files:
                raise ValueError(f"NPZ file {file_path} contains no arrays")
            image = npz_file[npz_file.files[0]].astype(np.float32)
    else:
        image = np.load(file_path_str).astype(np.float32)

    # Auto-normalize. Raw out-of-range intensities are min-max rescaled to the
    # true range (any bit depth — the 8-bit /127.5 assumption only holds for
    # PNG/JPG, handled by load_image_standard). Data already in [0, 1] is
    # affine-mapped; data already in [-1, 1] is left untouched.
    data_min, data_max = float(np.min(image)), float(np.max(image))
    if data_max > 1.5 and data_min >= 0.0:
        image = _normalize_to_unit_range(image)
    elif 0.0 <= data_min <= data_max <= 1.0:
        image = image * 2.0 - 1.0

    return image
