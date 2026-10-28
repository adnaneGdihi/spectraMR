"""IO Strategies for Universal MRI Data Loading.

This module abstracts file format details (H5 vs. NIfTI vs. DICOM)
from the dataset class. Normalizes everything into a standard format:
{'data': Tensor, 'kspace': Optional[Tensor], 'affine': Optional[Matrix]}

{'data': Tensor, 'kspace': Optional[Tensor], 'affine': Optional[Matrix]}

.. mermaid::

    classDiagram
        class IOStrategy {
            <<interface>>
            +load(path) -> dict
        }
        class FastMRIH5Strategy {
            +load(h5_path)
        }
        class NiftiStrategy {
            +load(nii_path)
        }
        class DicomStrategy {
            +load(dcm_path)
        }
        class IOStrategyFactory {
            +create(name)
            +register(name, wrapper)
        }

        IOStrategy <|-- FastMRIH5Strategy
        IOStrategy <|-- NiftiStrategy
        IOStrategy <|-- DicomStrategy
        IOStrategyFactory ..> IOStrategy : creates
"""

import logging
import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:  # h5py stays a lazy import on every load path below
    import h5py

logger = logging.getLogger(__name__)


def read_h5_kspace(ds: "h5py.Dataset", slice_index: int | None = None) -> np.ndarray:
    """Read one slice, or the whole array, of an open HDF5 ``kspace`` dataset.

    The single spelling of the HDF5 k-space read for both routes that have one:
    :meth:`FastMRIH5Strategy.load` (slice-aware through ``metadata["slice_index"]``)
    and ``M4RawRepetitionDataset._load_kspace`` (whole volume by default, one
    slice under ``data.slice_level_records``, #1757). Factoring the read here is
    what keeps the second route from becoming a third HDF5 reader; dtype
    conversion stays with each caller because the two disagree on it (M4Raw
    casts to complex64 / float32, fastMRI only downcasts float64).

    Args:
        ds: The open ``h5py.Dataset`` (``f["kspace"]``); the caller keeps the
            file handle and its own presence check.
        slice_index: ``None`` reads the whole array (``ds[()]``). An int reads
            ``ds[slice_index]`` along the leading axis, which h5py drops, so a
            ``(S, C, H, W)`` store yields ``(C, H, W)``.

    Returns:
        The numpy array h5py hands back, untouched.

    Raises:
        IndexError: ``slice_index`` is outside ``[0, ds.shape[0])``. h5py's own
            error for that case does not name the slice count, and a negative
            index would wrap silently, so the bound is checked here.
        TypeError: ``slice_index`` is not an integer.
    """
    if slice_index is None:
        return ds[()]
    index = operator.index(slice_index)
    # Every real ``h5py.Dataset`` carries ``shape``; the bound is checked
    # against it so a negative index cannot wrap and an overrun names the
    # count. A stand-in without ``shape`` (the e2e tests hand in a list) is
    # indexed natively and still raises IndexError on overrun.
    shape = getattr(ds, "shape", None)
    if shape is not None:
        n_slices = int(shape[0]) if len(shape) else 0
        if not 0 <= index < n_slices:
            raise IndexError(
                f"slice_index={index} is out of range for the kspace dataset "
                f"{getattr(ds, 'name', '?')!s} in "
                f"{getattr(getattr(ds, 'file', None), 'filename', '?')!s}: "
                f"it has {n_slices} slice(s) (shape {tuple(shape)})."
            )
    return ds[index]


def _downcast_real_float64(t: torch.Tensor) -> torch.Tensor:
    """Downcast a REAL float64 tensor to float32; leave everything else alone.

    NumPy defaults to float64, so ``torch.from_numpy(np.load(...))`` on a
    double-stored ``.npy`` yields a float64 tensor that silently upcasts the
    (float32) model downstream. This guards the real-valued ingestion paths.
    Complex tensors are preserved untouched — coil-sensitivity maps and k-space
    are legitimately complex and must not be coerced to real.
    """
    if t.dtype == torch.float64:
        return t.float()
    return t


class BaseIOStrategy(ABC):
    """Abstract base for file I/O strategies."""

    @abstractmethod
    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """
        Reads a file and returns a standardized dictionary.

        Must return: {'data': torch.Tensor, ...}
        Optional keys: 'kspace', 'affine', 'header', 'metadata'
        """
        pass

    def load_sensitivity(self, path: str) -> torch.Tensor:
        """Default handler for sensitivity maps (usually .pt or .npy)."""
        p = Path(path)
        if p.suffix == ".pt":
            return _downcast_real_float64(torch.load(p, map_location="cpu", weights_only=True))
        elif p.suffix == ".npy":
            return _downcast_real_float64(torch.from_numpy(np.load(p)))
        else:
            raise ValueError(f"Unsupported sensitivity map format: {p.suffix}")


class FastMRIH5Strategy(BaseIOStrategy):
    """
    Handles FastMRI .h5 files (Multi-coil K-space).

    Returns 'kspace' (complex) and 'data' (RSS reconstruction target).

    FastMRI layouts:
        - Single-coil: kspace=(Slices, H, W), rss=(Slices, H, W)
        - Multi-coil: kspace=(Slices, Coils, H, W), rss=(Slices, H, W)
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        # SAFETY CHECK: If path is clearly NIfTI, this is a config mismatch and
        # must be surfaced — do NOT silently delegate (NN#3 / pitfall #9).
        """load.

        Args:
            path (str): Description.
            metadata (Optional[dict]): Description.
        Returns:
            dict[str, Any]: Description.
        """
        path_str = str(path).lower()
        if path_str.endswith((".nii", ".nii.gz")):
            raise ValueError(
                f"FastMRIH5Strategy received a NIfTI path: '{path}'. "
                "Use io_strategy='nifti' or io_strategy='auto' for NIfTI files. "
                "FastMRIH5Strategy only handles .h5 files."
            )

        import h5py

        result = {"metadata": {}}

        with h5py.File(path, "r", swmr=True) as f:
            # [PERFORMANCE] Slice-Specific Loading
            slice_idx = None
            if metadata and "slice_index" in metadata:
                slice_idx = metadata["slice_index"]

            # 1. Load K-Space
            kspace_t = None
            if "kspace" in f:
                # Whole volume, or one slice along the leading axis when the
                # caller named one: (Slices, [Coils,] H, W) -> ([Coils,] H, W).
                # The read itself is shared with the M4Raw loader (#1757).
                kspace_np = read_h5_kspace(f["kspace"], slice_idx)

                # Downcast a real float64 store to float32 BEFORE the complex
                # view (double real/imag pairs → complex64, not complex128);
                # a natively-complex array is left untouched.
                kspace_t = _downcast_real_float64(torch.from_numpy(kspace_np))

                # Handle complex conversion if stored as (... ,2) float
                if not torch.is_complex(kspace_t):
                    if kspace_t.shape[-1] == 2:
                        kspace_t = torch.view_as_complex(kspace_t)

            # 2. Load Target Image (RSS)
            target_t = None
            key = None
            if "reconstruction_rss" in f:
                key = "reconstruction_rss"
            elif "reconstruction_esc" in f:
                key = "reconstruction_esc"

            if key:
                if slice_idx is not None:
                    target_np = f[key][slice_idx]
                else:
                    target_np = f[key][()]
                target_t = _downcast_real_float64(torch.from_numpy(target_np))

            # 3. Metadata & Header Check
            # Both FastMRI and M4Raw use 'ismrmrd_header' (XML string)
            header = None
            if "ismrmrd_header" in f:
                header = f["ismrmrd_header"][()]  # Dataset
            elif "ismrmrd_header" in f.attrs:
                header = f.attrs["ismrmrd_header"]  # Attribute

            if header is None:
                logger.warning(
                    f"[H5-LOAD] Missing 'ismrmrd_header' in {path}. "
                    "This is required for accurate physics simulation (M4Raw/FastMRI)."
                )
            else:
                result["header"] = header  # Store explicit header

            result["metadata"] = dict(f.attrs)
            if slice_idx is not None:
                result["metadata"]["slice_index"] = slice_idx

        result["kspace"] = kspace_t
        result["data"] = target_t
        result["affine"] = np.eye(4)  # Identity for H5 (no spatial info)

        return result

    @staticmethod
    def load_reference_volume(path: str | Path) -> dict[str, Any]:
        """The clean reference volume + its ISMRMRD header, WITHOUT reading k-space.

        ``load()`` reads every dataset in the file. A fastMRI brain multicoil volume is
        ~500 MB of k-space wrapping a ~13 MB ``reconstruction_rss``, so using ``load()``
        to export the reference is a ~40x read amplification -- half a terabyte of IO
        across the annotated cohort, for data that is thrown away.

        Returns ``data`` as ``[S, H, W]`` (the on-disk order: axis 0 is slices) plus the
        raw ``header``, which is the only place voxel spacing exists -- an h5 has no
        affine. See ``spectramr.data.nifti_export``.
        """
        import h5py

        with h5py.File(path, "r", swmr=True) as f:
            key = next(
                (k for k in ("reconstruction_rss", "reconstruction_esc") if k in f),
                None,
            )
            if key is None:
                raise KeyError(
                    f"{path} has no reconstruction_rss/_esc: there is no clean reference "
                    "to segment or to grade against."
                )
            data = _downcast_real_float64(torch.from_numpy(f[key][()]))
            header = f["ismrmrd_header"][()] if "ismrmrd_header" in f else None
            if header is None:
                header = f.attrs.get("ismrmrd_header")
            attrs = dict(f.attrs)

        return {"data": data, "header": header, "metadata": attrs}

    @staticmethod
    def load_torchio_tensor(
        path: str | Path,
        key_precedence: tuple[str, ...] = ("kspace", "reconstruction_rss", "data"),
    ) -> torch.Tensor:
        """Read the first available dataset from an H5 file and shape it
        for ``tio.ScalarImage``.

        Used by the manifest-driven subject builder in
        ``spectramr.data.metadata.index_builder`` to keep the h5py read inside
        the canonical data layer (CLAUDE.md pitfall #11 — data-loading
        SSOT). Preserves the manifest reader's key-precedence behaviour:

        1. Walk ``key_precedence`` and use the first key that exists.
        2. Fall back to ``list(f.keys())[0]`` if none matched.
        3. Stack complex arrays into ``[2, H, W]`` (real/imag).
        4. Promote ``[H, W]`` slices to ``[1, H, W]``.

        Returns a CPU tensor suitable for ``tio.ScalarImage(tensor=...)``.
        """
        import h5py

        with h5py.File(path, "r") as f:
            for k in key_precedence:
                if k in f:
                    data = f[k][()]
                    break
            else:
                first_key = list(f.keys())[0]
                data = f[first_key][()]

        if np.iscomplexobj(data):
            data = np.stack([data.real, data.imag], axis=0)
        elif data.ndim == 2:
            data = data[np.newaxis, ...]

        return torch.from_numpy(data)


def _nifti_spacing_mm(affine: Any) -> tuple[float, float, float]:
    """``(slice, row, col)`` from a NIfTI affine: voxel axes are (row, col, slice) on disk."""
    from spectramr.data.nifti_export import zooms_from_affine

    row, col, slc = zooms_from_affine(affine)
    return (slc, row, col)


class NiftiStrategy(BaseIOStrategy):
    """
    Handles .nii.gz files (Brain, Prostate volumes).

    Used for Experiment 32 (Image Translation).
    NIfTI is usually (W, H, D). TorchIO expects (C, W, H, D).
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """load.

        Args:
            path (str): Description.
            metadata (Optional[dict]): Description.
        Returns:
            dict[str, Any]: Description.
        """
        import nibabel as nib

        img = nib.load(path)

        # SLICING LOGIC
        slice_idx = metadata.get("slice_index") if metadata else None

        if slice_idx is not None:
            # Efficiently load single slice (if backend supports it, nibabel lazy loading)
            # NIfTI is (W, H, D)
            data = img.dataobj[..., slice_idx]
        else:
            # dtype=float32: get_fdata() defaults to float64, which doubles both the
            # decode cost and the transient host RAM of a full-volume read. The result
            # is cast to float32 on the next line regardless, so the wide intermediate
            # buys nothing. Callers that cannot take the lazy slice branch above (every
            # `slice_2d` arm, whose volume-level normalization forces a whole-volume
            # read) pay this on every cache miss.
            data = img.get_fdata(dtype=np.float32)

        # Normalize to float32 Tensor
        tensor = torch.from_numpy(np.asanyarray(data)).float()

        # Ensure Channel Dimension:
        # 2D Slice: (W, H) -> (1, W, H, 1) or (1, W, H) - TorchIO expects (C, W, H, D) usually?
        # TorchIO 2D images are usually (C, W, H, 1)

        if tensor.ndim == 2:
            # (W, H) -> (1, W, H, 1)
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        elif tensor.ndim == 3:
            # Case 1: Was 3D volume (W, H, D) -> (1, W, H, D)
            if slice_idx is None:
                tensor = tensor.unsqueeze(0)
            # Case 2: Was 2D slice with channels (W, H, C) -> (C, W, H, 1)
            else:
                # If we sliced a 4D vol (W, H, D, C) at D=idx -> (W, H, C)
                tensor = tensor.permute(2, 0, 1).unsqueeze(-1)
        elif tensor.ndim == 4:
            # (W, H, D, C) -> (C, W, H, D)
            tensor = tensor.permute(3, 0, 1, 2)

        final_shape = tensor.shape
        return {
            "data": tensor,
            "affine": np.array(img.affine),
            "metadata": {
                "shape": final_shape,
                "original_shape": img.shape,
                "slice_index": slice_idx,
            },
        }

    @staticmethod
    def load_reference_volume(path: str | Path) -> dict[str, Any]:
        """The whole volume as ``[S, H, W]`` plus its voxel spacing ``(slice, row, col)`` mm.

        The slice axis is the LAST on disk, as ``load`` and MRIxFields index it
        (``dataobj[..., i]``). Spacing comes from the affine; the acquisition sidecar
        is read by the caller.
        """
        import nibabel as nib

        img = nib.load(str(path))
        data = torch.from_numpy(np.asanyarray(img.dataobj, dtype=np.float32).copy())
        return {"data": data.movedim(-1, 0), "spacing_mm": _nifti_spacing_mm(img.affine)}

    @staticmethod
    def read_spacing_mm(path: str | Path) -> tuple[float, float, float]:
        """``(slice, row, col)`` mm from the affine alone; no voxel is decoded."""
        import nibabel as nib

        return _nifti_spacing_mm(nib.load(str(path)).affine)


class DicomStrategy(BaseIOStrategy):
    """Handles DICOM folders or individual DICOM files.

    Uses pydicom for DICOM parsing and properly stacks slices into volumes.
    Supports both single files and directory-based series.
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """Load DICOM data from file or directory.

        Args:
            path: Path to DICOM file or directory containing series
            metadata: Optional metadata hints

        Returns:
            Dictionary with 'data' tensor and metadata
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required for DICOM loading. Install with: pip install pydicom"
            )

        p = Path(path)

        if p.is_file():
            # Single DICOM file
            return self._load_single_file(p)
        elif p.is_dir():
            # DICOM directory (series)
            return self._load_series(p)
        else:
            raise FileNotFoundError(f"DICOM path not found: {path}")

    def _load_single_file(self, path: Path) -> dict[str, Any]:
        """Load a single DICOM file."""
        import pydicom

        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * slope + intercept

        tensor = torch.from_numpy(pixel_array).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dim

        return {
            "data": tensor,
            "affine": self._build_affine(ds),
            "metadata": self._extract_metadata(ds),
        }

    def _load_series(self, directory: Path, metadata: dict | None = None) -> dict[str, Any]:
        """Load DICOM series from directory with lazy optimization."""
        import pydicom

        # Find all DICOM files
        dicom_files = []
        for f in directory.iterdir():
            if f.is_file():
                try:
                    # Quick check if it's a DICOM
                    with open(f, "rb") as test_f:
                        test_f.seek(128)
                        if test_f.read(4) == b"DICM":
                            dicom_files.append(f)
                except Exception:
                    continue

        if not dicom_files:
            raise ValueError(f"No valid DICOM files found in {directory}")

        # 1. Lazy Header Read & Sort
        # We need headers to sort, but don't need pixels yet
        headers = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                headers.append((f, ds))
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        # Sort by instance number or slice location
        try:
            headers.sort(key=lambda x: int(x[1].InstanceNumber))
        except (AttributeError, ValueError):
            try:
                headers.sort(key=lambda x: float(x[1].SliceLocation))
            except (AttributeError, ValueError) as _exc:
                logger.debug("Suppressed exception: %s", _exc)  # Keep original order

        # 2. Check for Slice Index
        slice_idx = None
        if metadata and "slice_index" in metadata:
            slice_idx = metadata["slice_index"]

        # 3. Load Data
        if slice_idx is not None:
            # Load SINGLE slice
            if slice_idx < 0 or slice_idx >= len(headers):
                raise IndexError(
                    f"Slice index {slice_idx} out of range for series of length {len(headers)}"
                )

            target_file, target_header = headers[slice_idx]
            ds = pydicom.dcmread(str(target_file))  # Full load

            pixel_array = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)
            pixel_array = pixel_array * slope + intercept

            # (H, W) -> (1, 1, H, W)
            tensor = torch.from_numpy(pixel_array).float().unsqueeze(0).unsqueeze(0)

            # Use this header for metadata
            ref_ds = ds
        else:
            # Load ALL slices
            volume = []
            # We already have headers, but need full load for pixels
            # We iterate over SORTED headers
            for f, _ in headers:
                ds = pydicom.dcmread(str(f))  # Full load
                pixel_array = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, "RescaleSlope", 1.0)
                intercept = getattr(ds, "RescaleIntercept", 0.0)
                pixel_array = pixel_array * slope + intercept
                volume.append(pixel_array)

            volume = np.stack(volume, axis=0)  # [D, H, W]
            tensor = torch.from_numpy(volume).float().unsqueeze(0)  # [1, D, H, W]

            ref_ds = headers[0][1] if headers else None

        return {
            "data": tensor,
            "affine": self._build_affine(ref_ds) if ref_ds else np.eye(4),
            "metadata": self._extract_metadata(ref_ds) if ref_ds else {},
        }

    def _build_affine(self, ds) -> np.ndarray:
        """Build affine matrix from DICOM metadata."""
        affine = np.eye(4)
        try:
            # Extract spatial information
            pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
            slice_thickness = getattr(ds, "SliceThickness", 1.0)
            affine[0, 0] = float(pixel_spacing[0])
            affine[1, 1] = float(pixel_spacing[1])
            affine[2, 2] = float(slice_thickness)
        except Exception as _exc:
            logger.debug("Suppressed exception: %s", _exc)
        return affine

    def _extract_metadata(self, ds) -> dict[str, Any]:
        """Extract relevant metadata from DICOM dataset."""
        metadata = {}
        fields = [
            "PatientID",
            "StudyDescription",
            "SeriesDescription",
            "Modality",
            "Manufacturer",
            "MagneticFieldStrength",
        ]
        for field in fields:
            if hasattr(ds, field):
                metadata[field] = str(getattr(ds, field))
        return metadata


class BartCflStrategy(BaseIOStrategy):
    """Reads BART ``.cfl``/``.hdr`` pairs (non-Cartesian / multi-coil k-space).

    BART stores a complex array as two sibling files: a ``.hdr`` text header
    carrying the dimension vector and a ``.cfl`` raw little-endian ``complex64``
    payload in **Fortran (column-major)** order. This reader is a *pure format
    decoder* — it returns the raw complex array plus the BART dim vector and
    makes **no** assumption about which dims are readout / coil / echo / frame.
    That mapping is dataset-specific and is configured explicitly downstream via
    ``data.bart.bart_dim_map`` (see spec E2); the reader never silently reshapes,
    squeezes, or invents a recon target (CLAUDE.md pitfall #9: no silent
    fallback).

    Accepts either the ``.cfl`` path or BART's bare basename (no suffix). Returns
    ``{'data': complex tensor, 'header': {'bart_dims': [...]},
    'metadata': {'format': 'bart_cfl', ...}}``.
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """Decode a BART ``.cfl``/``.hdr`` pair into the standard IO dict."""
        p = Path(path)
        # BART addresses arrays by basename; tolerate an explicit ``.cfl`` too.
        base = p.with_suffix("") if p.suffix == ".cfl" else p
        hdr_path = base.with_suffix(".hdr")
        cfl_path = base.with_suffix(".cfl")

        if not hdr_path.is_file():
            raise FileNotFoundError(
                f"BART header not found for {path!r}: expected {hdr_path}. "
                "A .cfl without its .hdr cannot be reshaped — refusing to guess "
                "(CLAUDE.md pitfall #9: no silent fallback)."
            )
        if not cfl_path.is_file():
            raise FileNotFoundError(f"BART payload not found: {cfl_path}")

        dims = self._parse_dims(hdr_path.read_text())
        flat = np.fromfile(cfl_path, dtype=np.complex64)
        expected = int(np.prod(dims)) if dims else 0
        if flat.size != expected:
            raise ValueError(
                f"BART dim/payload mismatch for {cfl_path.name}: header dims "
                f"{dims} imply {expected} complex samples but the .cfl holds "
                f"{flat.size}. Header and payload are out of sync."
            )
        # BART payloads are column-major; recover the logical array, then copy to
        # C-order so the returned tensor has standard contiguous strides.
        arr = np.ascontiguousarray(flat.reshape(dims, order="F"))
        data = torch.from_numpy(arr)

        out_meta = dict(metadata or {})
        out_meta["format"] = "bart_cfl"
        out_meta["source_path"] = str(cfl_path)
        return {
            "data": data,
            "header": {"bart_dims": dims},
            "metadata": out_meta,
        }

    @staticmethod
    def _parse_dims(text: str) -> list[int]:
        """Extract the BART dimension vector from a ``.hdr`` text body.

        The first non-comment, non-blank line is the space-separated dim vector.
        """
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                return [int(tok) for tok in stripped.split()]
            except ValueError as exc:
                raise ValueError(f"Malformed BART header dimension line: {stripped!r}") from exc
        raise ValueError("BART header contains no dimension line.")


class IsmrmrdStrategy(BaseIOStrategy):
    """Reads ISMRMRD ``.mrd`` raw-acquisition containers (spec E1).

    ISMRMRD stores raw MR acquisitions in an HDF5 group (default ``dataset``)
    holding a structured ``data`` array — one row per acquisition with a
    ``head`` (``AcquisitionHeader``), a variable-length ``traj`` (trajectory),
    and a variable-length ``data`` (interleaved real/imag k-space) — plus an
    ``xml`` ISMRMRD header. This reader decodes the container with ``h5py``
    directly (no ISMRMRD C++/Python dependency) by addressing header fields *by
    name*, so it works on both real files and minimal fixtures.

    It is a *pure decoder*: it assembles the acquisitions into a complex k-space
    tensor ``[n_acq, n_coil, n_samp]`` plus an optional trajectory
    ``[n_acq, n_samp, ndim]`` and parses the XML encoding header. It **raises**
    rather than guessing when the container layout is unrecognised or the
    acquisitions are ragged (varying samples/channels) — pitfall #9.
    """

    _NS = "{http://www.ismrm.org/ISMRMRD}"

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """Decode an ISMRMRD ``.mrd`` container into the standard IO dict."""
        import h5py

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"ISMRMRD file not found: {p}")

        with h5py.File(p, "r") as f:
            group = self._find_ismrmrd_group(f)
            prefix = f"{group}/" if group else ""
            xml_raw = f[f"{prefix}xml"][0]
            header = self._parse_header(xml_raw)
            kspace, trajectory = self._read_acquisitions(f[f"{prefix}data"][:])

        header["n_acquisitions"] = int(kspace.shape[0])
        out_meta = dict(metadata or {})
        out_meta["format"] = "ismrmrd"
        out_meta["source_path"] = str(p)
        result: dict[str, Any] = {
            "data": kspace,
            "kspace": kspace,
            "header": header,
            "metadata": out_meta,
        }
        if trajectory is not None:
            result["trajectory"] = trajectory
        return result

    @staticmethod
    def _find_ismrmrd_group(f: Any) -> str:
        """Return the name of the group holding both ``data`` and ``xml``.

        ISMRMRD wraps acquisitions in a named group (default ``dataset``);
        dispatch is by *structure*, not by the group's name. Returns ``""`` if
        the datasets sit at the file root. Raises if no such group exists.
        """
        for key in f:
            item = f[key]
            if hasattr(item, "keys") and "data" in item and "xml" in item:
                return str(key)
        if "data" in f and "xml" in f:
            return ""
        raise ValueError(
            f"No ISMRMRD group (a group containing both 'data' and 'xml') found "
            f"in {getattr(f, 'filename', '<in-memory>')}; top-level keys: "
            f"{list(f.keys())}."
        )

    @classmethod
    def is_ismrmrd_h5(cls, path: str | Path) -> bool:
        """True if ``path`` is an HDF5 holding an ISMRMRD container (a group with
        both ``data`` and ``xml``).

        ISMRMRD is often distributed as ``.h5`` (e.g. kasper's monitored spiral
        raw data), which would otherwise route to ``FastMRIH5Strategy``. This
        lets ``AutoDetectStrategy`` route it here instead. Reads only the group
        structure, not the payload.
        """
        try:
            import h5py

            with h5py.File(path, "r") as f:
                cls._find_ismrmrd_group(f)  # raises if not an ISMRMRD container
            return True
        except Exception:
            return False

    @classmethod
    def _read_acquisitions(cls, acqs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Assemble the structured acquisition array into k-space + trajectory."""
        heads = acqs["head"]
        n_samp = heads["number_of_samples"].astype(int)
        n_chan = heads["active_channels"].astype(int)
        traj_dim = heads["trajectory_dimensions"].astype(int)

        if len(np.unique(n_samp)) != 1 or len(np.unique(n_chan)) != 1:
            raise ValueError(
                "Ragged ISMRMRD acquisitions (varying number_of_samples / "
                "active_channels) are not yet supported — refusing to pad or "
                "truncate silently (pitfall #9). "
                f"samples={np.unique(n_samp).tolist()}, "
                f"channels={np.unique(n_chan).tolist()}."
            )
        ns, nc, td = int(n_samp[0]), int(n_chan[0]), int(traj_dim[0])
        n_acq = len(acqs)

        kspace = np.empty((n_acq, nc, ns), dtype=np.complex64)
        for i in range(n_acq):
            raw = np.asarray(acqs["data"][i], dtype=np.float32).reshape(nc, ns, 2)
            kspace[i] = raw[..., 0] + 1j * raw[..., 1]
        kspace_t = torch.from_numpy(kspace)

        trajectory_t: torch.Tensor | None = None
        if td > 0:
            trajectory = np.empty((n_acq, ns, td), dtype=np.float32)
            for i in range(n_acq):
                trajectory[i] = np.asarray(acqs["traj"][i], dtype=np.float32).reshape(ns, td)
            trajectory_t = torch.from_numpy(trajectory)
        return kspace_t, trajectory_t

    @classmethod
    def _parse_header(cls, xml_raw: Any) -> dict[str, Any]:
        """Extract trajectory + encoded/recon matrix sizes from the ISMRMRD XML.

        Header parsing is provenance, not the correctness path: missing fields
        degrade to ``None`` and a malformed body records ``parsed=False`` rather
        than raising (the *acquisitions* are what must decode correctly).
        """
        import xml.etree.ElementTree as ET

        text = xml_raw.decode() if isinstance(xml_raw, bytes | bytearray) else str(xml_raw)
        header: dict[str, Any] = {"raw_xml_present": bool(text)}
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            header["parsed"] = False
            return header
        ns = cls._NS
        enc = root.find(f"{ns}encoding")
        if enc is not None:
            header["trajectory"] = enc.findtext(f"{ns}trajectory")
            matrix = enc.find(f"{ns}encodedSpace/{ns}matrixSize")
            if matrix is not None:
                header["encoded_matrix"] = [
                    int(matrix.findtext(f"{ns}{axis}", default="0")) for axis in ("x", "y", "z")
                ]
        header["parsed"] = True
        return header


class MatStrategy(BaseIOStrategy):
    """Reads MATLAB ``.mat`` containers — v7.3 (HDF5) and legacy v5 (spec — 4D-flow / OCMR).

    A ``.mat`` is a *container of named variables*, not a single array, so this
    reader returns the full decoded structure under ``variables`` (nested dicts
    mirroring MATLAB structs), decoding the MATLAB complex-compound dtype
    ``[('real', f4), ('imag', f4)]`` into native complex tensors. The MATLAB
    internal ``#refs#`` reference table is skipped.

    Version dispatch is by *try-open*, not a silent fallback: v7.3 files are
    HDF5 (read with ``h5py``, which transparently handles the MATLAB userblock);
    if the file is not HDF5 it is a legacy v5 file (read with ``scipy.io``). A
    file that is neither raises.

    ``metadata['mat_key']`` selects a single variable via a dotted path (e.g.
    ``'D.kb'``) into ``data``; without it, ``data`` is the whole ``variables``
    dict (the caller / dataset layer picks). Axis order is as the backend reads
    it (h5py reverses MATLAB's column-major order) — the dataset layer owns that
    convention, the reader does not silently transpose.
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """Decode a MATLAB ``.mat`` (v7.3 or v5) into ``{data, variables, metadata}``."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"MAT file not found: {p}")

        try:
            import h5py

            with h5py.File(p, "r") as f:
                variables = {k: self._decode_h5(f[k]) for k in f if k != "#refs#"}
            fmt = "mat_v73"
        except OSError:
            import scipy.io

            raw = scipy.io.loadmat(str(p))
            variables = {k: self._decode_scipy(v) for k, v in raw.items() if not k.startswith("__")}
            fmt = "mat_v5"

        mat_key = (metadata or {}).get("mat_key")
        data = self._navigate(variables, mat_key) if mat_key else variables

        out_meta = dict(metadata or {})
        out_meta["format"] = fmt
        out_meta["source_path"] = str(p)
        out_meta["keys"] = sorted(variables.keys())
        return {"data": data, "variables": variables, "metadata": out_meta}

    @classmethod
    def _decode_h5(cls, item: Any) -> Any:
        """Recursively decode an HDF5 group/dataset into dicts / tensors."""
        if hasattr(item, "keys"):  # group → MATLAB struct
            return {k: cls._decode_h5(item[k]) for k in item if k != "#refs#"}
        arr = item[()]
        return cls._array_to_tensor(arr)

    @staticmethod
    def _array_to_tensor(arr: Any) -> Any:
        names = getattr(arr.dtype, "names", None)
        if names and {"real", "imag"} <= set(names):
            cplx = arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)
            return torch.from_numpy(np.ascontiguousarray(cplx))
        if arr.dtype.kind in "fc":
            return torch.from_numpy(np.ascontiguousarray(arr))
        if arr.dtype.kind in "iu":
            try:
                return torch.from_numpy(np.ascontiguousarray(arr))
            except TypeError:  # torch has no uint16/uint32 → widen
                return torch.from_numpy(np.ascontiguousarray(arr.astype(np.int64)))
        return arr  # char / object arrays: leave as numpy

    @classmethod
    def _decode_scipy(cls, value: Any) -> Any:
        """Convert a top-level scipy ``loadmat`` value to a tensor when numeric."""
        if hasattr(value, "dtype") and value.dtype.kind in "fc":
            return torch.from_numpy(np.ascontiguousarray(value))
        if hasattr(value, "dtype") and value.dtype.kind in "iu":
            try:
                return torch.from_numpy(np.ascontiguousarray(value))
            except TypeError:
                return torch.from_numpy(np.ascontiguousarray(value.astype(np.int64)))
        return value

    @staticmethod
    def _navigate(variables: dict[str, Any], key: str) -> Any:
        """Resolve a dotted ``mat_key`` path into the nested ``variables`` dict."""
        node: Any = variables
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                available = list(node.keys()) if isinstance(node, dict) else type(node)
                raise KeyError(
                    f"mat_key {key!r} not found at segment {part!r}; available: {available}."
                )
            node = node[part]
        return node


class TwixStrategy(BaseIOStrategy):
    """Reads Siemens TWIX ``.dat`` raw-acquisition files (phase-cycled bSSFP / oracle_bssfp).

    TWIX is Siemens' proprietary raw-data container. This strategy delegates the binary
    decoding to the dependency-free :mod:`spectramr.data.twix` decoder (the SSOT, also used
    by the header-only profiler) and adapts its output to the standard IO dict, returning
    the imaging measurement's *flat* complex k-space ``[n_acq, n_coil, n_samp]`` as a
    torch tensor. Like :class:`IsmrmrdStrategy` it does **not** grid the acquisitions into
    a ``[Lin, Par, …]`` volume — that loop-counter mapping is sequence-specific and owned
    by the dataset layer (pitfall #9: no silent reshape). VB-format k-space is unimplemented
    and the decoder raises rather than guess.
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """Decode a TWIX ``.dat`` into ``{data, kspace, line_indices, header, metadata}``."""
        from dataclasses import asdict

        from spectramr.data.twix import read_twix

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"TWIX file not found: {p}")

        result = read_twix(p)
        kspace = torch.from_numpy(np.ascontiguousarray(result["kspace"]))
        out_meta = dict(metadata or {})
        out_meta.update(result["metadata"])
        return {
            "data": kspace,  # raw-k-space reader: 'data' mirrors 'kspace'
            "kspace": kspace,
            "line_indices": torch.from_numpy(result["line_indices"]),
            "header": asdict(result["header"]),
            "metadata": out_meta,
        }


class AutoDetectStrategy(BaseIOStrategy):
    """Automatic file format detection based on extension.

    This strategy routes to the correct handler based on file path,
    preventing issues like h5py trying to open NIfTI files.
    """

    def load(self, path: str, metadata: dict | None = None) -> dict[str, Any]:
        """load.

        Args:
            path (str): Description.
            metadata (Optional[dict]): Description.
        Returns:
            dict[str, Any]: Description.
        """
        p = Path(path)
        suffix = p.suffix.lower()

        # Handle compound extensions like .nii.gz
        if suffix == ".gz" and p.stem.endswith(".nii"):
            suffix = ".nii.gz"

        # Route based on extension
        if suffix == ".h5":
            # ISMRMRD is commonly distributed as .h5 (e.g. kasper monitored
            # spiral); sniff the container structure so it routes to the right
            # reader instead of always FastMRIH5.
            strategy = (
                IsmrmrdStrategy() if IsmrmrdStrategy.is_ismrmrd_h5(path) else FastMRIH5Strategy()
            )
        elif suffix in (".nii", ".nii.gz"):
            strategy = NiftiStrategy()
        elif suffix in (".cfl", ".hdr"):
            strategy = BartCflStrategy()
        elif suffix == ".mrd":
            strategy = IsmrmrdStrategy()
        elif suffix == ".mat":
            strategy = MatStrategy()
        elif suffix == ".dat":
            strategy = TwixStrategy()
        elif suffix in (".dcm", ".dicom", ".ima") or p.is_dir():
            # ``.ima`` is Siemens' uppercase DICOM export extension (lower-cased above).
            strategy = DicomStrategy()
        else:
            # No silent fallback (CLAUDE.md pitfall #9/#10): an unrecognised
            # extension must fail loud rather than be handed to nibabel, which
            # would either raise a confusing internal error or mis-decode a
            # magic-bearing file. (The old warn-and-default-to-NIfTI also
            # mis-routed extensionless DICOM exports.)
            raise ValueError(
                f"Unrecognised file extension {suffix!r} for {path!r}: no IO "
                f"reader is registered for it. Known: .h5/.mrd/.nii(.gz)/.cfl/"
                f".hdr/.mat/.dat/.dcm/.ima or a DICOM directory."
            )

        return strategy.load(path, metadata)


class IOStrategyFactory:
    """Factory to dispatch the correct strategy."""

    _registry = {
        "fastmri_h5": FastMRIH5Strategy,
        "fastmri": FastMRIH5Strategy,
        "fastmri_kspace": FastMRIH5Strategy,
        "fastmri_knee": FastMRIH5Strategy,
        "fastmri_brain": FastMRIH5Strategy,
        "nifti": NiftiStrategy,
        "paired_nifti": NiftiStrategy,
        "m4raw": AutoDetectStrategy,  # Use auto-detect for m4raw (may have mixed formats)
        "dicom": DicomStrategy,
        "bart": BartCflStrategy,  # BART .cfl/.hdr non-Cartesian / multi-coil k-space (E2)
        "bart_cfl": BartCflStrategy,
        "ismrmrd": IsmrmrdStrategy,  # ISMRMRD .mrd raw-acquisition container (E1)
        "mrd": IsmrmrdStrategy,
        "mat": MatStrategy,  # MATLAB v7.3/v5 .mat container (4D-flow / OCMR)
        "matlab": MatStrategy,
        "twix": TwixStrategy,  # Siemens TWIX .dat raw acquisition (oracle_bssfp / phase-cycled bSSFP)
        "dat": TwixStrategy,
        "auto": AutoDetectStrategy,  # Explicit auto-detect option
    }

    @classmethod
    def get(cls, strategy_name: str, **kwargs) -> BaseIOStrategy:
        """Get an IO strategy by name."""
        if strategy_name not in cls._registry:
            raise ValueError(
                f"Unknown IO Strategy: {strategy_name}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[strategy_name]()

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a new IO strategy."""
        cls._registry[name] = strategy_class


def load_tensor_from_file(path: str | Path, h5_keys: Sequence[str] | None = None) -> torch.Tensor:
    """Load a single tensor from a file, dispatching on suffix.

    Centralizes the multi-format reader that the data-layer builders
    used to inline. Reusing this from
    ``src/data/builders/torchio_subject_builder.py`` and from the inference
    pipeline keeps the "any-format → torch.Tensor" surface in one place
    (CLAUDE.md pitfall #11: data-loading SSOT under ``src/data/``).

    Supported suffixes: ``.pt``, ``.npy``, ``.nii`` / ``.nii.gz``, ``.h5`` /
    ``.hdf5``.

    Args:
        path: Local filesystem path.
        h5_keys: Optional ordered preference of dataset keys to read from an
            HDF5 file (e.g. ``("kspace", "reconstruction_rss", "data")``). The
            first key present in the file wins; if none match (or ``h5_keys`` is
            ``None``) the first key in the file is used. Ignored for non-HDF5.

    Returns:
        ``torch.Tensor`` with the file contents.

    Raises:
        ValueError: If the suffix isn't one of the supported formats.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    # Normalise compound extension: Path('foo.nii.gz').suffix == '.gz'
    if suffix == ".gz" and p.stem.lower().endswith(".nii"):
        suffix = ".nii.gz"

    if suffix == ".pt":
        return _downcast_real_float64(torch.load(p, map_location="cpu", weights_only=True))
    if suffix == ".npy":
        return _downcast_real_float64(torch.from_numpy(np.load(p)))
    if suffix in (".nii", ".nii.gz"):
        import nibabel as nib

        img = nib.load(str(p))
        # float32: get_fdata() would decode the whole volume as float64 only for it to
        # be cast down on the same line (see NiftiStrategy.load).
        return torch.from_numpy(img.get_fdata(dtype=np.float32)).float()
    if suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(p, "r") as f:
            key = None
            if h5_keys:
                key = next((k for k in h5_keys if k in f), None)
            if key is None:
                key = list(f.keys())[0]
            return _downcast_real_float64(torch.from_numpy(f[key][:]))
    raise ValueError(f"Unsupported file format: {suffix} ({p})")


def load_h5_dataset_if_present(path: str | Path, key: str) -> torch.Tensor | None:
    """Read ONE named dataset from an HDF5 file, or ``None``.

    Unlike :func:`load_tensor_from_file`, whose ``h5_keys`` is a *preference*
    that falls back to the first key in the file when none match, this never
    returns a different dataset than the one asked for. That is the property a
    sidecar read needs: asking for ``"mask"`` on a file that has none must say
    "absent", never hand back the k-space under the mask's name.

    Args:
        path: Local filesystem path. A non-HDF5 suffix yields ``None``.
        key: Exact dataset name.

    Returns:
        The dataset as a tensor (float64 downcast like every other reader
        here), or ``None`` when the file is not HDF5 or lacks ``key``.
    """
    p = Path(path)
    if p.suffix.lower() not in (".h5", ".hdf5"):
        return None
    import h5py

    with h5py.File(p, "r") as f:
        if key not in f:
            return None
        return _downcast_real_float64(torch.from_numpy(f[key][:]))


def lazy_load_nifti(path):
    # type signature: path: str | pathlib.Path -> nibabel image handle
    # (Plain annotation kept off the def line because this module is
    # imported under Python without ``from __future__ import annotations``
    # and the parent has runtime-evaluated unions in this position.)
    """Return the ``nib.Nifti1Image`` (or compatible) handle WITHOUT eagerly loading the data.

    Use this when a caller needs the lazy ``dataobj`` proxy for memory-
    mapped / chunked reads — e.g. streaming inference over a volume too
    large to fit in RAM. The eager ``NiftiStrategy.load()`` returns the
    full array, which defeats streaming.

    Centralising the ``nib.load`` call here keeps it inside the data
    SSOT subtree (CLAUDE.md pitfall #11) so consumers in
    ``src/infrastructure/inference/`` don't import ``nibabel`` directly.
    Phase 4 of ``TODO/backlog_ssot_and_layering_cleanup.md``.

    Args:
        path: Path to a ``.nii`` / ``.nii.gz`` file.

    Returns:
        A ``nibabel`` image handle. Caller accesses ``.dataobj`` for the
        lazy proxy, ``.shape`` for dimensions, and ``.affine`` for the
        transformation matrix.
    """
    import nibabel as nib

    return nib.load(str(path))


def save_nifti(path, array, affine):
    """Write *array* as a NIfTI volume at *path* with *affine*.

    The write-side twin of :func:`lazy_load_nifti`: keeps ``nib.save`` inside the
    data SSOT so ``infrastructure/inference/`` never imports ``nibabel``.
    """
    import nibabel as nib

    nib.save(nib.Nifti1Image(array, affine), str(path))
