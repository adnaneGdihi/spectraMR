"""Multi-coil ``.h5`` k-space reads and the per-slice NEX pseudo-GT volume stack.

This is the one place under ``data/`` that opens an M4Raw / fastMRI ``.h5`` for the
pseudo-GT references (rule 7): the header probe, the slice or full ``kspace`` read and the
``ismrmrd_header`` read. The physics (NEX merge, ESPIRiT, SENSE, p99) is the pure tensor
code of :mod:`spectramr.infrastructure.physics.m4raw_pseudo_gt`.

:func:`load_kspace_volume` turns one :class:`ContentKey` into a :class:`KspaceVolume`:
every slice NEX-merged across the content's repetitions, ESPIRiT maps per slice, one p99
over the whole stack, and the slice spacing and voxel size from the header.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch

from spectramr.data.datasets.contrast_grid import ContentKey
from spectramr.infrastructure.physics.m4raw_pseudo_gt import (
    nex_merge,
    parse_acquisition_header,
    pseudo_gt_from_kspace,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SPACING_SOURCES",
    "HeaderVoxel",
    "KspaceVolume",
    "content_header_voxel",
    "header_voxel_mm",
    "kspace_shape_from_h5",
    "load_kspace_from_h5",
    "load_kspace_volume",
    "read_acquisition_params",
]


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")


def kspace_shape_from_h5(path: Path) -> tuple[int, ...]:
    """The ``kspace`` dataset's shape from the HDF5 header only (no voxel IO, #616)."""
    _require(path)
    with h5py.File(str(path), "r") as f:
        if "kspace" not in f:
            raise KeyError(f"No 'kspace' dataset in {path}. Keys: {list(f.keys())}")
        return tuple(int(d) for d in f["kspace"].shape)


def load_kspace_from_h5(path: Path, slice_index: int | None = None) -> torch.Tensor:
    """Complex64 k-space ``(Slices, Coils, H, W)``, or one slice when ``slice_index`` is given.

    A trailing real/imag axis of size 2 is viewed as complex; the slice read is a partial
    h5py read, so one slice never materialises the volume (#616).
    """
    _require(path)
    with h5py.File(str(path), "r") as f:
        if "kspace" not in f:
            raise KeyError(f"No 'kspace' dataset in {path}. Keys: {list(f.keys())}")
        raw = f["kspace"][()] if slice_index is None else f["kspace"][slice_index]
    t = torch.from_numpy(raw)
    if torch.is_complex(t):
        return t.to(torch.complex64)
    if t.shape[-1] == 2:
        return torch.view_as_complex(t.contiguous().float())
    return t.float()


def read_acquisition_params(path: Path) -> dict[str, object]:
    """:func:`parse_acquisition_header` of the file's ``ismrmrd_header``; ``{}`` without one."""
    _require(path)
    with h5py.File(str(path), "r") as f:
        if "ismrmrd_header" not in f:
            return {}
        raw = f["ismrmrd_header"][()]
    xml = raw.decode() if isinstance(raw, bytes | bytearray) else str(raw)
    return parse_acquisition_header(xml)


#: Where a volume's slice spacing came from, stamped as ``spacing_source``.
SPACING_SOURCES = ("spacingBetweenSlices", "encoded_fov_z")
_warned_fov_z = False


@dataclass(frozen=True)
class HeaderVoxel:
    """Voxel size from an ISMRMRD header, in tensor dim order (D, H, W)."""

    voxel_mm: tuple[float, float, float]
    spacing_source: str
    pe_dim: int = -1
    """ISMRMRD: matrix x (readout) is H (dim -2), y (phase encode) is W (dim -1)."""

    def provenance(self) -> dict[str, object]:
        return {
            "voxel_mm": list(self.voxel_mm),
            "slice_spacing_mm": self.voxel_mm[0],
            "spacing_source": self.spacing_source,
            "pe_dim": self.pe_dim,
        }


def header_voxel_mm(params: dict[str, object], height: int, width: int, where: str) -> HeaderVoxel:
    """``(slice spacing, H, W)`` voxel in mm under the ISMRMRD convention (x -> H, y -> W).

    The slice spacing is ``spacingBetweenSlices``, else the encoded FOV z (warned once and
    stamped as ``spacing_source``). A missing matrix or FOV, a matrix that is not
    ``(H, W)`` in x/y order, or a non-positive size raises.
    """
    global _warned_fov_z
    missing = [k for k in ("encoded_matrix", "fov_mm") if k not in params]
    if missing:
        raise ValueError(f"{where}: the ismrmrd_header lacks encodedSpace {missing}")
    mx, my = params["encoded_matrix"]  # type: ignore[misc]
    fx, fy, fz = params["fov_mm"]  # type: ignore[misc]
    if (mx, my) != (height, width):
        raise ValueError(
            f"{where}: encoded matrix x*y = {mx}x{my} is not kspace H*W = {height}x{width} "
            "(ISMRMRD: x is the readout, dim -2)"
        )
    if "slice_spacing_mm" in params:
        spacing, source = float(params["slice_spacing_mm"]), SPACING_SOURCES[0]  # type: ignore[arg-type]
    else:
        spacing, source = float(fz), SPACING_SOURCES[1]
        if not _warned_fov_z:
            logger.warning(
                "%s: no spacingBetweenSlices in the ismrmrd_header; using the encoded FOV z "
                "(%.3g mm) as the slice spacing (stamped spacing_source=encoded_fov_z).",
                where,
                spacing,
            )
            _warned_fov_z = True
    voxel = (spacing, float(fx) / mx, float(fy) / my)
    if not all(math.isfinite(v) and v > 0 for v in voxel):
        raise ValueError(f"{where}: the ismrmrd_header gives no usable voxel size {voxel}")
    return HeaderVoxel(voxel, source)


def _kspace_hw(path: Path) -> tuple[int, int]:
    """``(H, W)`` of the ``kspace`` dataset from its metadata (a real/imag axis is skipped)."""
    _require(path)
    with h5py.File(str(path), "r") as f:
        ds = f["kspace"]
        shape = ds.shape[:-1] if ds.dtype.kind != "c" and ds.shape[-1] == 2 else ds.shape
    return int(shape[-2]), int(shape[-1])


def content_header_voxel(key: ContentKey) -> HeaderVoxel:
    """Every repetition's header resolved, without reading k-space; they must agree."""
    geoms = {header_voxel_mm(read_acquisition_params(p), *_kspace_hw(p), p.name) for p in key.paths}
    if len(geoms) != 1:
        raise ValueError(f"{key.content_id}: repetitions disagree on voxel geometry {geoms}")
    return geoms.pop()


@dataclass(frozen=True)
class KspaceVolume:
    """One content's NEX pseudo-GT volume; ``S`` is the slice axis, ``C`` the coils."""

    key: ContentKey
    coil_images: torch.Tensor
    """Complex image-domain coil images ``(1, C, S, H, W)``."""
    smaps: torch.Tensor
    """ESPIRiT maps ``(1, C, S, H, W)`` (ones for single-coil data)."""
    magnitude: torch.Tensor
    """SENSE magnitude ``(1, 1, S, H, W)`` over one p99 of the whole stack."""
    p99: float
    header: HeaderVoxel
    n_reps: int

    @property
    def voxel_mm(self) -> tuple[float, float, float]:
        return self.header.voxel_mm

    @property
    def contrast(self) -> str:
        return self.key.contrast

    @property
    def content_id(self) -> str:
        return self.key.content_id

    def provenance(self) -> dict[str, object]:
        return {
            "content_id": self.content_id,
            "contrast": self.contrast,
            "file_ids": [Path(f).name for f in self.key.file_ids],
            "n_reps": self.n_reps,
            **self.header.provenance(),
            "raw_p99": self.p99,
            "normalisation": "p99_over_volume",
        }


def _as_volume(ks: torch.Tensor, path: Path) -> torch.Tensor:
    if ks.dim() == 3:  # single-coil (S, H, W)
        return ks.unsqueeze(1)
    if ks.dim() != 4:
        raise ValueError(
            f"{path.name}: kspace must be (S, C, H, W) or (S, H, W); got {tuple(ks.shape)}"
        )
    return ks


def _to_5d(t: torch.Tensor) -> torch.Tensor:
    """``(S, C, H, W)`` -> ``(1, C, S, H, W)``, the volume kernels' canonical layout."""
    return t.permute(1, 0, 2, 3).unsqueeze(0)


def load_kspace_volume(key: ContentKey, device: torch.device | None = None) -> KspaceVolume:
    """Every slice of ``key``'s repetitions, NEX-merged into one pseudo-GT volume stack.

    ``key.slice_index`` must be ``None``: a volume content is the whole stack. Every
    repetition's header is resolved (and must agree) before any k-space is read.
    """
    if key.slice_index is not None:
        raise ValueError(
            f"{key.content_id}: a volume content takes every slice; got slice_index="
            f"{key.slice_index}"
        )
    paths = key.paths
    if not paths:
        raise ValueError(f"{key.content_id}: no files")
    header = content_header_voxel(key)
    reps = [_as_volume(load_kspace_from_h5(p), p) for p in paths]
    y, kept = nex_merge(reps, paths)  # (S, C, H, W)
    if y.shape[0] < 2:
        raise ValueError(f"{key.content_id}: a volume needs >= 2 slices; got {y.shape[0]}")
    if device is not None:
        y = y.to(device)
    coil_images, smaps, mag, p99 = pseudo_gt_from_kspace(y)
    return KspaceVolume(
        key=key,
        coil_images=_to_5d(coil_images),
        smaps=_to_5d(smaps),
        magnitude=_to_5d(mag),
        p99=float(p99),
        header=header,
        n_reps=len(kept),
    )
