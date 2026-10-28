"""Rigidly align one M4Raw contrast onto another and write the result back as k-space.

The cross-contrast arms condition a target contrast on a fully sampled reference
contrast of the same subject, acquired as a separate series, so the head can move
between the two. This module estimates that motion per reference file and removes it.
It takes a split of NEX averages (``nex_averaging``), one ``<subject>_<contrast>00.h5``
per contrast, and refuses acquired repetitions: the motion to remove is between two
contrasts, and one excitation is fitted with the noise its average would not carry.

1. **Estimate** a rigid (Euler 3-D) fit with Mattes mutual information between the
   coil-combined images of the target and the reference, both
   restricted to the centred k-space disc that every undersampling mask of these arms
   keeps (``acs_fraction`` of k-space by area, as in ``sampling.py``). The fit never
   sees target data a deployed model would not have.
2. **Apply** the fit's translation at the field-of-view centre: in-plane exactly, as a
   Fourier shift (``fourier_shift``); through-plane as a whole-slice re-index, since
   interpolating 5 mm slices would blur the reference. Vacated edge slices are zero
   and flagged in ``registration_valid_slices``. Rotation is measured, not applied: a
   fit rotating more than ``max_rotation_deg`` is refused.
3. **Write** the reference with its ``kspace`` replaced and every other dataset and
   attribute copied, bar the images reconstructed from the old k-space, into a mirror of the split directory whose other files are
   symlinks to the originals, so manifests and the loader see an unchanged layout.

The target is never modified: its k-space is the measurement data consistency uses.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from spectramr.data.datasets.m4raw_identity import NEX_AVERAGE_REPETITION, parse_m4raw_file_id
from spectramr.data.h5_kspace_layout import slice_chunked_kwargs
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.repeat_session_registration import (
    require_sitk,
    rigid_transform,
)
from spectramr.infrastructure.physics.subpixel_registration import fourier_shift

try:
    import SimpleITK as sitk  # noqa: N813
except ImportError:  # require_sitk() raises the named error before any use
    sitk = None


@dataclass(frozen=True)
class RigidFit:
    """Displacement to apply to the reference, in index units ``(slice, row, col)``."""

    shift_px: tuple[float, float, float]
    slice_offset: int
    rotation_deg: float


def acs_disc(height: int, width: int, fraction: float) -> torch.Tensor:
    """The centred disc of ``fraction`` of k-space by area, as ``sampling.py`` builds it."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"acs_fraction must be in (0, 1], got {fraction}")
    radius = math.sqrt(max(height * width * fraction, 1.0) / math.pi)
    rows = torch.arange(height, dtype=torch.float32) - height // 2
    cols = torch.arange(width, dtype=torch.float32) - width // 2
    return (rows[:, None] ** 2 + cols[None, :] ** 2).sqrt() <= radius


def read_kspace(dataset: h5py.Dataset) -> torch.Tensor:
    """``[S, C, H, W]`` complex64 from complex storage or a trailing real/imag axis of 2."""
    array = dataset[()]
    if np.iscomplexobj(array):
        return torch.from_numpy(array.astype(np.complex64))
    if array.shape[-1] != 2:
        raise ValueError(f"kspace is real with shape {array.shape}; expected a trailing axis of 2")
    return torch.view_as_complex(torch.from_numpy(np.ascontiguousarray(array, np.float32)))


def lowpass_rss(kspace: torch.Tensor, disc: torch.Tensor) -> np.ndarray:
    """Coil-combined magnitude ``[S, H, W]`` of the k-space inside ``disc``."""
    image = ifft2c(kspace * disc)
    return image.abs().pow(2).sum(1).sqrt().numpy().astype(np.float32)


def voxel_spacing(header: Any, height: int, width: int) -> tuple[float, float, float]:
    """``(slice, row, col)`` spacing in mm from an ISMRMRD header, any namespace form."""
    text = header.decode() if isinstance(header, bytes) else str(header)
    root = ET.fromstring(text)
    space = root.find("{*}encoding/{*}encodedSpace")
    if space is None:
        raise ValueError("ismrmrd_header has no encoding/encodedSpace")

    def axis(tag: str, a: str) -> float:
        return float(space.findtext(f"{{*}}{tag}/{{*}}{a}", "nan"))

    if (axis("matrixSize", "y"), axis("matrixSize", "x")) != (height, width):
        raise ValueError(
            f"header matrix {axis('matrixSize', 'y'):.0f}x{axis('matrixSize', 'x'):.0f} "
            f"does not match kspace {height}x{width}; the pixel spacing would be a guess"
        )
    gap = next((el.text for el in root.iter() if el.tag.endswith("spacingBetweenSlices")), None)
    spacing = (
        float(gap) if gap else axis("fieldOfView_mm", "z"),
        axis("fieldOfView_mm", "y") / height,
        axis("fieldOfView_mm", "x") / width,
    )
    if not all(math.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(f"ismrmrd_header gives no usable voxel spacing: {spacing}")
    return spacing


def _volume(array: np.ndarray, spacing: tuple[float, float, float]) -> Any:
    image = sitk.GetImageFromArray(array)  # numpy (z, y, x) -> ITK index (x, y, z)
    image.SetSpacing((spacing[2], spacing[1], spacing[0]))
    return image


def fit_rigid(
    fixed: np.ndarray, moving: np.ndarray, spacing: tuple[float, float, float]
) -> RigidFit:
    """Rigid fit of ``moving`` onto ``fixed`` (both ``[S, H, W]``), read at the FOV centre."""
    require_sitk()
    fixed_img, moving_img = _volume(fixed, spacing), _volume(moving, spacing)
    transform = rigid_transform(fixed_img, moving_img, metric="mattes")
    centre = fixed_img.TransformContinuousIndexToPhysicalPoint(
        [(n - 1) / 2 for n in fixed_img.GetSize()]
    )
    # moving(T(p)) ~ fixed(p): the reference's content sits at p + t and must move by -t.
    t = np.subtract(transform.TransformPoint(centre), centre)
    shift = (-t[2] / spacing[0], -t[1] / spacing[1], -t[0] / spacing[2])
    cos = (np.trace(np.reshape(transform.GetMatrix(), (3, 3))) - 1) / 2
    return RigidFit(
        shift_px=tuple(float(s) for s in shift),
        slice_offset=int(math.copysign(math.floor(abs(shift[0]) + 0.5), shift[0])),
        rotation_deg=float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
    )


def apply_fit(kspace: torch.Tensor, fit: RigidFit) -> tuple[torch.Tensor, np.ndarray]:
    """Shift ``[S, C, H, W]`` k-space by ``fit``; returns it with the per-slice valid mask."""
    n_slices, n_coils = kspace.shape[:2]
    shifts = torch.tensor(fit.shift_px[1:], dtype=torch.float32).expand(n_slices, n_coils, 2)
    shifted = fft2c(fourier_shift(ifft2c(kspace), shifts))
    out = torch.zeros_like(shifted)
    valid = np.zeros(n_slices, dtype=bool)
    s = fit.slice_offset
    dst = slice(max(s, 0), n_slices + min(s, 0))
    src = slice(max(-s, 0), n_slices + min(-s, 0))
    out[dst], valid[dst] = shifted[src], True
    return out, valid


def write_kspace_copy(source: Path, dest: Path, kspace: torch.Tensor, attrs: dict[str, Any]) -> None:
    """Copy ``source`` to ``dest`` with ``kspace`` replaced, stored one chunk per slice.

    ``reconstruction_*`` images (M4Raw ships ``reconstruction_rss``) are dropped, not
    copied: they were reconstructed from the replaced k-space and would contradict it.
    """
    with h5py.File(source, "r") as src, h5py.File(dest, "w") as dst:
        dropped = [key for key in src if key.startswith("reconstruction_")]
        for key in src:
            if key != "kspace" and key not in dropped:
                src.copy(key, dst)
        original = src["kspace"]
        data = kspace.numpy()
        if not np.issubdtype(original.dtype, np.complexfloating):
            data = np.stack([data.real, data.imag], axis=-1)
        out = dst.create_dataset(
            "kspace",
            data=data.astype(original.dtype),
            **slice_chunked_kwargs(data.shape),
        )
        out.attrs.update(original.attrs)
        dst.attrs.update(src.attrs)
        dst.attrs.update(attrs)
        dst.attrs["dropped_kspace_derived_datasets"] = dropped


def group_registrable(
    src_dir: Path, *, reference: str, target: str
) -> dict[str, dict[str, list[Path]]]:
    """Subject -> contrast -> files of ``src_dir``, reading names only; writes nothing.

    A reference or target that is an acquired repetition rather than a NEX average
    raises: the motion to remove is between two contrasts' averages.
    """
    files = sorted(src_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files directly in {src_dir}")
    by_subject: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        ident = parse_m4raw_file_id(path.stem)
        if ident.contrast in (reference, target) and ident.repetition != NEX_AVERAGE_REPETITION:
            raise ValueError(
                f"{path.name} is an acquired repetition; register the NEX averages "
                "(scripts/data/average_m4raw_nex.py) instead"
            )
        by_subject[ident.subject][ident.contrast].append(path)
    return by_subject


def register_split(
    src_dir: Path,
    dst_dir: Path,
    *,
    reference: str,
    target: str,
    acs_fraction: float,
    max_rotation_deg: float,
) -> list[dict[str, Any]]:
    """Mirror ``src_dir`` into ``dst_dir`` with every ``reference`` file aligned to ``target``.

    Returns one record per reference file: its fit, or why it was left unregistered.
    A refused fit is recorded and its file is neither written nor linked; which files
    may be registered at all is ``group_registrable``'s to decide.
    """
    by_subject = group_registrable(src_dir, reference=reference, target=target)
    files = sorted(p for contrasts in by_subject.values() for ps in contrasts.values() for p in ps)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.iterdir()):
        raise FileExistsError(f"{dst_dir} is not empty; refusing to mix two registrations")

    records: list[dict[str, Any]] = []
    for subject, contrasts in sorted(by_subject.items()):
        refs, targets = contrasts.get(reference, []), contrasts.get(target, [])
        fixed = spacing = None
        if refs and targets:
            with h5py.File(targets[0], "r") as f:
                target_k = read_kspace(f["kspace"])
                disc = acs_disc(*target_k.shape[-2:], acs_fraction)
                spacing = voxel_spacing(f["ismrmrd_header"][()], *target_k.shape[-2:])
            fixed = lowpass_rss(target_k, disc)
        for ref in refs:
            record: dict[str, Any] = {"subject": subject, "file": ref.name}
            if fixed is None:
                record["status"] = f"no {target} file; linked unregistered"
                (dst_dir / ref.name).symlink_to(ref.resolve())
                records.append(record)
                continue
            with h5py.File(ref, "r") as f:
                ref_k = read_kspace(f["kspace"])
            if (ref_k.shape[0], *ref_k.shape[-2:]) != fixed.shape:
                # The re-index pairs slices by index, so a different grid has no aligned copy.
                record["status"] = f"refused: grid {tuple(ref_k.shape)} vs target {fixed.shape}"
                records.append(record)
                continue
            fit = fit_rigid(fixed, lowpass_rss(ref_k, disc), spacing)
            record.update(asdict(fit), target_file=targets[0].name)
            if fit.rotation_deg > max_rotation_deg:
                record["status"] = f"refused: rotation {fit.rotation_deg:.2f} deg"
                records.append(record)
                continue
            kspace, valid = apply_fit(ref_k, fit)
            write_kspace_copy(
                ref,
                dst_dir / ref.name,
                kspace,
                {
                    "registration_reference_to_target": f"{reference}->{target}",
                    "registration_source_file": str(ref.resolve()),
                    "registration_target_file": targets[0].name,
                    "registration_shift_px": np.asarray(fit.shift_px),
                    "registration_slice_offset": fit.slice_offset,
                    "registration_rotation_deg": fit.rotation_deg,
                    "registration_rotation_applied": False,
                    "registration_acs_fraction": acs_fraction,
                    "registration_valid_slices": valid,
                },
            )
            record["status"] = "registered"
            records.append(record)
    for path in files:
        if parse_m4raw_file_id(path.stem).contrast != reference:
            (dst_dir / path.name).symlink_to(path.resolve())
    return records
