"""Export a volume to NIfTI for SynthSeg: real voxel spacing, slice axis pinned.

SynthSeg reads NIfTI, not HDF5, and until now nothing in this repo could produce one --
``build_synthseg_region_cache.py --emit-synthseg-cmd`` printed a command pointing SynthSeg
at a directory of ``.h5`` files it cannot open. This module is the missing half.

Two facts make it load-bearing rather than plumbing.

**The affine is not decoration -- SynthSeg resamples with it.** ``FastMRIH5Strategy``
returns ``affine = np.eye(4)`` for every h5, because an h5 carries no spatial information.
Hand SynthSeg an identity affine for a 16x320x320 stack and it reads that volume as 16 mm
through-plane by 320 mm in-plane, up-samples the slice axis ~5x, and segments a brain that
was never acquired. It does not fail: it emits a confident, plausible label map, and every
tissue metric downstream is a number about anatomy that does not exist (pitfall #16). So
the spacing comes from the ISMRMRD header or the export **raises**. There is no 1 mm
default, because a wrong default here is undetectable in the output.

**The stored array is a CROP of what the header describes.** ``reconstruction_rss`` is
fastMRI's centre crop of the scanner reconstruction -- 320x320 for brain -- while
``reconSpace`` reports the *uncropped* recon matrix and field of view. A crop changes the
extent, not the voxel size, so spacing is ``fieldOfView_mm / matrixSize`` from the header's
own pair and never ``fieldOfView_mm / array_size``. Getting that backwards inflates every
in-plane spacing by ``recon_matrix / 320`` and, because SynthSeg resamples with the affine,
returns a plausible segmentation of a brain at the wrong physical scale. The spacing is
recorded per volume and re-checked after segmentation by :func:`check_label_spacing` --
shape checks alone cannot see this class of error.

**The array order is a contract with the cache, not a preference.** ``RegionMaskCache``
indexes ``vol[slice_index]`` (``mask_cache.py``) -- axis 0 is slices -- and the image slice
the metric grades is ``f["reconstruction_rss"][i]``, also axis 0. So an exported volume is
``[S, H, W]``, slice **first**, and SynthSeg hands the order back untouched (it saves the
segmentation with the input's own affine and header, realigned to the input's array order).
Note this is the *opposite* of the ``(W, H, D)`` order ``NiftiStrategy``'s docstring
ascribes to the pre-existing brain/prostate NIfTIs. That docstring describes those files;
it is not a law, and ``NiftiStrategy`` transposes nothing. Get the order backwards and every
"tissue mask" is a sagittal cut through the volume, mis-registered against the axial slice
it is scored on -- while the run stays green.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "LATERALITY_VERIFIED",
    "PROVENANCE_FILENAME",
    "SLICE_AXIS",
    "ExportVerificationError",
    "GeometryUnavailableError",
    "GridMismatchError",
    "VoxelGeometry",
    "check_label_grid",
    "check_label_spacing",
    "load_export_grids",
    "load_export_spacings",
    "parse_ismrmrd_geometry",
    "verify_slice_first_nifti",
    "verify_slice_last_nifti",
    "write_slice_first_nifti",
    "write_slice_last_nifti",
    "zooms_from_affine",
]

PROVENANCE_FILENAME = "export_provenance.json"
"""Written beside the exported NIfTIs. It is what makes the export *checkable*: it pins,
per volume, the grid the labels must come back on."""

SLICE_AXIS = 0
"""Exported volumes are ``[S, H, W]``. See the module docstring -- this is the axis
``RegionMaskCache`` indexes and the axis ``reconstruction_rss`` is stored on."""

LATERALITY_VERIFIED = False
"""fastMRI ships no patient-orientation metadata, so left/right handedness is an
*assumption* of the affine below, not a measurement. Consequences are bounded and must
stay that way: a bilateral region (``structure:thalamus``) is unaffected, but a lateralised
one (``structure:left_thalamus``) would carry a name we cannot defend. Regions fold L/R by
default for exactly this reason; a lateral split must gate on this flag."""

# In-plane spacings outside this band mean the header is in metres, or is not describing
# the volume we hold. Either way the affine would be wrong by ~1000x and SynthSeg would
# still return a segmentation.
_MIN_INPLANE_MM, _MAX_INPLANE_MM = 0.05, 10.0
_MIN_SLICE_MM, _MAX_SLICE_MM = 0.1, 20.0


class GeometryUnavailableError(ValueError):
    """The header does not pin the voxel spacing. Never downgrade this to a default."""


class ExportVerificationError(RuntimeError):
    """A written NIfTI did not read back as the volume that was written."""


class GridMismatchError(ValueError):
    """A segmentation is not on the grid of the image it is supposed to label."""


@dataclass(frozen=True)
class VoxelGeometry:
    """Millimetre spacing of an ``[S, H, W]`` volume, plus the affine that declares it."""

    slice_mm: float
    row_mm: float
    col_mm: float
    n_slices: int
    rows: int
    cols: int
    slice_gap_assumed_zero: bool
    source: str
    #: The header's in-plane recon matrix, mapped onto (row, col). ``reconstruction_rss``
    #: is a CENTER CROP of this, so it is >= (rows, cols); equal when nothing was cropped.
    recon_rows: int = 0
    recon_cols: int = 0

    @property
    def crop_rows(self) -> int:
        """Rows discarded by fastMRI's center crop (0 when the array is uncropped)."""
        return max(0, self.recon_rows - self.rows)

    @property
    def crop_cols(self) -> int:
        """Columns discarded by fastMRI's center crop."""
        return max(0, self.recon_cols - self.cols)

    def affine(self) -> np.ndarray:
        """The 4x4 mapping voxel index ``(s, row, col)`` -> mm.

        Axis 0 (slice) -> +z, axis 1 (row) -> -y (row 0 is anterior in an axial stack),
        axis 2 (col) -> +x. ``det > 0``, so SynthSeg's RAS alignment does not read the
        volume as flipped. The origin centres the volume, which keeps SynthSeg's internal
        crop centred on the brain rather than on a corner.
        """
        m = np.zeros((3, 3), dtype=np.float64)
        m[2, 0] = self.slice_mm
        m[1, 1] = -self.row_mm
        m[0, 2] = self.col_mm
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = m
        shape = np.array([self.n_slices, self.rows, self.cols], dtype=np.float64)
        affine[:3, 3] = -m @ ((shape - 1.0) / 2.0)
        return affine

    def zooms(self) -> tuple[float, float, float]:
        return (self.slice_mm, self.row_mm, self.col_mm)

    def as_provenance(self) -> dict[str, Any]:
        return {
            "slice_mm": self.slice_mm,
            "row_mm": self.row_mm,
            "col_mm": self.col_mm,
            "shape_shw": [self.n_slices, self.rows, self.cols],
            # The receipt that lets a later reader tell a correct crop-aware spacing from
            # the old FOV/array-size one: with these, row_mm * recon_rows must reproduce
            # the header's field of view.
            "recon_matrix_rc": [self.recon_rows, self.recon_cols],
            "center_crop_rc": [self.crop_rows, self.crop_cols],
            "slice_axis": SLICE_AXIS,
            "slice_gap_assumed_zero": self.slice_gap_assumed_zero,
            "laterality_verified": LATERALITY_VERIFIED,
            "source": self.source,
        }


def _as_text(header: Any) -> str:
    if isinstance(header, np.ndarray):
        header = header.item() if header.ndim == 0 else header[0]
    if isinstance(header, bytes | bytearray | np.bytes_):
        return bytes(header).decode("utf-8", errors="replace")
    return str(header)


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for el in root.iter():
        el.tag = el.tag.rpartition("}")[2]
    return root


def _triplet(parent: ET.Element | None, tag: str, cast: type) -> tuple[Any, Any, Any]:
    node = None if parent is None else parent.find(tag)
    if node is None:
        raise GeometryUnavailableError(f"ISMRMRD header has no reconSpace/{tag}")
    out = []
    for axis in ("x", "y", "z"):
        child = node.find(axis)
        if child is None or child.text is None:
            raise GeometryUnavailableError(f"reconSpace/{tag} has no <{axis}>")
        try:
            out.append(cast(child.text))
        except ValueError as exc:
            raise GeometryUnavailableError(
                f"reconSpace/{tag}/{axis} is not a number: {child.text!r}"
            ) from exc
    return out[0], out[1], out[2]


def _checked(value: float, lo: float, hi: float, what: str) -> float:
    if not math.isfinite(value) or not lo <= value <= hi:
        raise GeometryUnavailableError(
            f"{what} resolves to {value!r} mm, outside the plausible band "
            f"[{lo}, {hi}]. The header does not describe this volume (metres? a "
            "different acquisition?) -- exporting it would hand SynthSeg a "
            "confidently wrong grid."
        )
    return float(value)


def parse_ismrmrd_geometry(header: Any, *, n_slices: int, rows: int, cols: int) -> VoxelGeometry:
    """Voxel spacing for an ``[S, H, W]`` volume, from the file's ISMRMRD header.

    Raises ``GeometryUnavailableError`` rather than assuming anything: an unparseable
    header must stop the export, not silently produce a 1 mm isotropic lie.
    """
    if header is None:
        raise GeometryUnavailableError(
            "no ismrmrd_header in the file -- voxel spacing is unknowable and a "
            "default would be a fabrication"
        )
    try:
        root = _strip_namespaces(ET.fromstring(_as_text(header)))
    except ET.ParseError as exc:
        raise GeometryUnavailableError(f"ismrmrd_header is not valid XML: {exc}") from exc

    recon = root.find("encoding/reconSpace")
    if recon is None:
        raise GeometryUnavailableError("ISMRMRD header has no encoding/reconSpace")

    mx, my, mz = _triplet(recon, "matrixSize", int)
    fx, fy, fz = _triplet(recon, "fieldOfView_mm", float)

    # ISMRMRD's x is the readout (columns), y the phase-encode (rows). Two things have
    # to be right here, and only one of them used to be.
    #
    # WHICH header axis is which image axis: still confirmed against the array rather
    # than trusted, because a swapped in-plane spacing is invisible on a square matrix.
    #
    # WHERE the spacing comes from: the header's own (fieldOfView_mm, matrixSize) pair,
    # NEVER fieldOfView / array-size. ``reconstruction_rss`` is fastMRI's CENTER CROP of
    # the scanner reconstruction (320x320 for brain), and a crop changes the field of
    # view, not the voxel size. Dividing the full FOV by the cropped 320 inflates every
    # in-plane spacing by recon_matrix/320 -- and SynthSeg *resamples with the affine*,
    # so it would segment the brain at the wrong physical scale and return a perfectly
    # plausible label map. Requiring matrixSize == array size (the old rule) turned that
    # into a hard drop whenever the crop was real, and into a silent wrong spacing
    # whenever the matrix happened to equal 320 while the FOV described the full recon.
    direct = mx >= cols and my >= rows
    transposed = mx >= rows and my >= cols
    if direct and transposed and rows != cols and (mx, my) != (cols, rows):
        # Both mappings admit the array, and it is not square, so "which header axis is
        # the readout" is genuinely undetermined -- exactly the guess the exact-match
        # rule existed to prevent. Only an exact hit disambiguates it.
        raise GeometryUnavailableError(
            f"reconSpace matrixSize ({mx}x{my}) admits the array's {rows}x{cols} in BOTH "
            "orientations without matching either exactly. The readout/phase-encode "
            "mapping would be a guess, and a swapped in-plane spacing is undetectable "
            "downstream."
        )
    if direct:
        col_mm, row_mm = fx / mx, fy / my
        recon_cols, recon_rows = mx, my
        axes = "reconSpace(x=cols, y=rows)"
    elif transposed:
        col_mm, row_mm = fy / my, fx / mx
        recon_cols, recon_rows = my, mx
        axes = "reconSpace(x=rows, y=cols) [transposed]"
    else:
        raise GeometryUnavailableError(
            f"reconSpace matrixSize ({mx}x{my}) is smaller than the array's "
            f"{rows}x{cols} in both orientations. The stored volume cannot be a crop of "
            "the reconstruction the header describes, so this header does not belong to "
            "this array and the in-plane spacing would be a guess."
        )
    # A crop is expected (fastMRI stores a 320x320 centre crop) but an ODD-sized one is
    # not: the affine centres the volume, which is only still the brain's centre if equal
    # amounts came off both sides.
    if (recon_rows, recon_cols) != (rows, cols) and (
        (recon_rows - rows) % 2 or (recon_cols - cols) % 2
    ):
        raise GeometryUnavailableError(
            f"reconSpace matrixSize ({recon_rows}x{recon_cols}) minus the array's "
            f"{rows}x{cols} is odd in at least one axis, so the stored volume is not "
            "a symmetric centre crop. The centred affine would put the origin half a "
            "voxel off the acquired centre and SynthSeg would resample around the "
            "wrong point."
        )

    if mz == n_slices and fz > 0:
        slice_mm, gap_assumed = fz / mz, False
    elif mz == 1 and fz > 0:
        # 2-D multi-slice: reconSpace.z describes ONE slice, so its FOV is the slice
        # thickness. True slice SPACING is thickness + inter-slice gap, and the gap is
        # nowhere in a fastMRI header. Assume contiguous, and say so out loud -- a
        # recorded assumption is falsifiable, a hidden one is not.
        slice_mm, gap_assumed = fz, True
    else:
        raise GeometryUnavailableError(
            f"reconSpace matrixSize.z={mz} with {n_slices} slices and "
            f"fieldOfView_mm.z={fz}: cannot resolve slice spacing"
        )

    return VoxelGeometry(
        slice_mm=_checked(slice_mm, _MIN_SLICE_MM, _MAX_SLICE_MM, "slice spacing"),
        row_mm=_checked(row_mm, _MIN_INPLANE_MM, _MAX_INPLANE_MM, "row spacing"),
        col_mm=_checked(col_mm, _MIN_INPLANE_MM, _MAX_INPLANE_MM, "column spacing"),
        n_slices=n_slices,
        rows=rows,
        cols=cols,
        slice_gap_assumed_zero=gap_assumed,
        source=axes,
        recon_rows=recon_rows,
        recon_cols=recon_cols,
    )


def write_slice_first_nifti(
    path: str | Path, volume: torch.Tensor, geometry: VoxelGeometry
) -> Path:
    """Write ``[S, H, W]`` float32 to ``path`` (which MUST end ``.nii.gz``).

    The suffix is not cosmetic. SynthSeg keys its QC csv on
    ``basename.replace('.nii.gz', '')`` and names its output ``p.replace('.nii',
    '_synthseg.nii')``. Export an uncompressed ``.nii`` and the QC subject key (``x.nii``)
    stops matching the volume id (``x_synthseg``); every volume then drops as
    ``no_qc_row`` and ``build_synthseg_region_cache`` declares the tissue axis unsound --
    after the GPU segmentation run has been paid for.
    """
    import nibabel as nib

    out = Path(path)
    if out.name.endswith(".nii"):
        raise ValueError(
            f"{out.name}: SynthSeg's QC-csv subject key and its output filename only "
            "line up for '.nii.gz'. See this function's docstring."
        )
    if not out.name.endswith(".nii.gz"):
        raise ValueError(f"{out.name}: expected a '.nii.gz' path")
    if volume.ndim != 3:
        raise ValueError(f"expected [S, H, W], got {tuple(volume.shape)}")
    if tuple(volume.shape) != (geometry.n_slices, geometry.rows, geometry.cols):
        raise ValueError(
            f"volume {tuple(volume.shape)} does not match the geometry it is being "
            f"written with ({geometry.n_slices}, {geometry.rows}, {geometry.cols})"
        )
    array = volume.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(
            "volume contains non-finite values; SynthSeg's percentile rescale would "
            "propagate them into every label"
        )

    img = nib.Nifti1Image(array, geometry.affine())
    img.header.set_zooms(geometry.zooms())
    img.header.set_xyzt_units("mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out))
    return out


def write_slice_last_nifti(path: str | Path, volume: torch.Tensor, geometry: VoxelGeometry) -> Path:
    """Write ``[S, H, W]`` float32 on disk as ``(H, W, S)``, the training loaders' layout.

    ``NiftiStrategy.load`` and the paired-NIfTI path take the LAST on-disk axis as depth,
    so a dataset meant for training is written slice-last; the slice-first writer above is
    the SynthSeg export's layout and would make a loader cut through the wrong plane. The
    affine permutes the same columns, so every voxel keeps its millimetre position.
    """
    import nibabel as nib

    out = Path(path)
    if not out.name.endswith((".nii.gz", ".nii")):
        raise ValueError(f"{out.name}: expected a '.nii.gz' or '.nii' path")
    if tuple(volume.shape) != (geometry.n_slices, geometry.rows, geometry.cols):
        raise ValueError(
            f"volume {tuple(volume.shape)} does not match the geometry it is being "
            f"written with ({geometry.n_slices}, {geometry.rows}, {geometry.cols})"
        )
    array = volume.detach().cpu().to(torch.float32).permute(1, 2, 0).contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values")
    affine = geometry.affine()
    affine[:, :3] = affine[:, [1, 2, 0]]
    img = nib.Nifti1Image(array, affine)
    img.header.set_zooms((geometry.row_mm, geometry.col_mm, geometry.slice_mm))
    img.header.set_xyzt_units("mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out))
    return out


def verify_slice_last_nifti(
    path: str | Path, volume: torch.Tensor, spacing_mm: tuple[float, float, float]
) -> None:
    """Read back through ``NiftiStrategy.load_reference_volume`` and assert values and
    ``(slice, row, col)`` spacing match what was written."""
    from spectramr.data.io_strategies import NiftiStrategy

    ref = NiftiStrategy.load_reference_volume(path)
    want = volume.detach().cpu().to(torch.float32)
    if tuple(ref["data"].shape) != tuple(want.shape) or not torch.equal(ref["data"], want):
        raise ExportVerificationError(
            f"{path} read back as {tuple(ref['data'].shape)}, wrote {tuple(want.shape)}; "
            "the slice axis moved or the values changed."
        )
    if not np.allclose(ref["spacing_mm"], spacing_mm, rtol=1e-5):
        raise ExportVerificationError(
            f"{path} reads back spacing {ref['spacing_mm']}, wrote {tuple(spacing_mm)}"
        )


def verify_slice_first_nifti(path: str | Path, volume: torch.Tensor) -> None:
    """Read the file back through ``NiftiStrategy`` -- the reader the cache builder
    itself uses -- and assert it is the volume that was written.

    A shape check alone is enough to catch a transposition here only because S != H on
    every real brain volume; the value check catches the rest (rescaling, dtype, a
    silently reordered write).
    """
    from spectramr.data.io_strategies import NiftiStrategy

    got = NiftiStrategy().load(str(path))["data"].squeeze()
    want = volume.detach().cpu().to(torch.float32)
    if tuple(got.shape) != tuple(want.shape):
        raise ExportVerificationError(
            f"{path} read back as {tuple(got.shape)}, wrote {tuple(want.shape)}. The "
            "slice axis moved: RegionMaskCache indexes axis 0, so every tissue mask "
            "would be a cut through the wrong plane."
        )
    if not torch.equal(got, want):
        raise ExportVerificationError(
            f"{path} read back with different values than were written "
            f"(max |diff| = {(got - want).abs().max().item():.3e})"
        )


def load_export_grids(provenance: str | Path) -> dict[str, tuple[int, int, int]]:
    """``{volume id: (S, H, W)}`` for every volume this export actually wrote.

    The segmenter is a black box between the export and the cache. This is the receipt
    that lets the cache builder check what came back against what went in.
    """
    path = Path(provenance)
    if path.is_dir():
        path = path / PROVENANCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {PROVENANCE_FILENAME} at {path}. It is written by "
            "scripts/data/export_h5_to_nifti.py; without it there is no record of which "
            "grid the labels are supposed to be on, and a transposed or resampled "
            "segmentation cannot be told from a correct one."
        )
    doc = json.loads(path.read_text())
    grids: dict[str, tuple[int, int, int]] = {}
    for vol in doc.get("volumes", []):
        shape = vol.get("geometry", {}).get("shape_shw")
        if shape is None:
            continue
        grids[Path(str(vol["file_id"])).stem] = (
            int(shape[0]),
            int(shape[1]),
            int(shape[2]),
        )
    if not grids:
        raise GridMismatchError(f"{path} records no exported volumes")
    return grids


def load_export_spacings(
    provenance: str | Path,
) -> dict[str, tuple[float, float, float]]:
    """``{volume id: (slice_mm, row_mm, col_mm)}`` for every volume this export wrote.

    The companion to :func:`load_export_grids`. Shape is what the old guard chain
    checked; spacing is what SynthSeg actually *consumes*, and it was unchecked end to
    end -- see :func:`check_label_spacing`.
    """
    path = Path(provenance)
    if path.is_dir():
        path = path / PROVENANCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {PROVENANCE_FILENAME} at {path}. Without it the mm spacing the labels "
            "were produced under is unrecorded, and a wrong affine is invisible."
        )
    doc = json.loads(path.read_text())
    out: dict[str, tuple[float, float, float]] = {}
    for vol in doc.get("volumes", []):
        geom = vol.get("geometry", {})
        try:
            spacing = (
                float(geom["slice_mm"]),
                float(geom["row_mm"]),
                float(geom["col_mm"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        out[Path(str(vol["file_id"])).stem] = spacing
    if not out:
        raise GridMismatchError(f"{path} records no volume spacings")
    return out


def zooms_from_affine(affine: Any) -> tuple[float, float, float]:
    """``(axis0, axis1, axis2)`` mm spacing from a 4x4 voxel-to-mm affine.

    The spacing along voxel axis ``i`` is the length of the affine's ``i``-th column,
    which is orientation-agnostic: it does not care which anatomical direction the axis
    points in, only how far one voxel step moves in millimetres. That is what makes it
    comparable against :meth:`VoxelGeometry.zooms` regardless of sign conventions.
    """
    mat = np.asarray(affine, dtype=np.float64)
    if mat.shape[:2] != (4, 4):
        raise GridMismatchError(f"expected a 4x4 affine, got shape {mat.shape}")
    lengths = np.linalg.norm(mat[:3, :3], axis=0)
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]))


def check_label_spacing(
    volume_id: str,
    label_zooms: tuple[float, ...],
    spacings: dict[str, tuple[float, float, float]],
    *,
    rtol: float = 1e-3,
) -> None:
    """Assert the labels came back at the mm spacing the image was exported with.

    This is the gap :func:`check_label_grid` cannot close. Every other check in the
    export -> segment -> cache chain compares **shapes**: ``write_slice_first_nifti``
    checks the array against the geometry's shape, ``verify_slice_first_nifti`` reads
    the file back and compares *values*, and ``check_label_grid`` compares the label
    shape. The affine is compared by none of them -- yet SynthSeg resamples to 1 mm
    using exactly that affine, so a wrong spacing yields a confident segmentation of a
    brain at the wrong physical scale, on the right grid, with a clean QC row.

    ``rtol`` is deliberately tight. This is not a numerical-tolerance question: the
    failure it guards is spacing wrong by the crop ratio (320/recon_matrix, tens of
    percent), never by an ulp.
    """
    want = spacings.get(volume_id)
    if want is None:
        raise GridMismatchError(
            f"{volume_id} has no recorded spacing in the export provenance, so the "
            "physical scale its labels were produced under is unknown. Re-export."
        )
    got = tuple(float(z) for z in label_zooms[:3])
    if len(got) != 3:
        raise GridMismatchError(f"{volume_id}: label header carries {len(got)} zooms, expected 3.")
    for axis, (g, w) in zip(("slice", "row", "col"), zip(got, want, strict=True), strict=True):
        if w <= 0 or abs(g - w) > rtol * abs(w):
            raise GridMismatchError(
                f"{volume_id}: segmentation {axis} spacing is {g:.6g} mm but the image "
                f"was exported at {w:.6g} mm (ratio {g / w if w else float('nan'):.4g}). "
                "SynthSeg resampled against a different physical scale than the image is "
                "graded on, so the labels describe anatomy of the wrong size. A ratio "
                "near recon_matrix/320 means the export divided the field of view by the "
                "cropped array size instead of the recon matrix."
            )


def check_label_grid(
    volume_id: str, label_shape: tuple[int, ...], grids: dict[str, tuple[int, int, int]]
) -> None:
    """Assert a segmentation came back on the grid its image is graded on.

    SynthSeg resamples to 1 mm internally and resamples *back* to the input grid before
    saving, so the label volume's shape must equal the exported image's. When it does
    not, the segmenter silently handed us a different sampling of space -- and the cache
    would go on indexing it with the h5's slice numbers, so ``labels[7]`` would mask a
    slice the image never had. That is not a degraded mask, it is a mask of somewhere
    else, and no downstream check can see it: a resampled label map is still a valid
    label map.

    A pure in-plane transpose is invisible to this check on a square matrix (fastMRI
    brain is 320x320) -- that one is caught at write time by
    :func:`verify_slice_first_nifti`, before the segmenter ever runs.
    """
    want = grids.get(volume_id)
    if want is None:
        raise GridMismatchError(
            f"{volume_id} is not in the export provenance. This segmentation was not "
            "produced from this export, so the grid its labels live on is unknown. "
            "Re-export and re-segment rather than caching a volume of unknown origin."
        )
    if tuple(label_shape) != want:
        raise GridMismatchError(
            f"{volume_id}: segmentation is {tuple(label_shape)} but the image exported "
            f"as {want}. The labels are not on the image's grid, so the cache would mask "
            "slice i of the labels against slice i of a different volume. Check that "
            "SynthSeg ran without --resample/--crop and that the segmentation directory "
            "matches this export."
        )
