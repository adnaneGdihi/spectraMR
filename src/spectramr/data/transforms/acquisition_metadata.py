"""Acquisition-metadata ingestion (Phase 4b).

Reads per-sample scan parameters (TE / TR / TI / FA / B0 / b_value / bvec)
from BIDS-JSON sidecars, DICOM headers, or HDF5 attributes and attaches them
to a TorchIO :class:`tio.Subject` as a Python dict under
``subject["acquisition_metadata"]``.

This is the single source of truth for the metadata block emitted in the
batch. Downstream consumers (``AcquisitionEmbedding``,
``BlochConsistencyLoss``) read from the batch dict by key — they should
never re-parse sidecars themselves.

Sources
=======

- ``bids_json``: looks for a sibling ``<basename>.json`` next to each image
  file (BIDS spec). Keys are mapped from the BIDS canonical names
  (``EchoTime`` → ``TE`` in ms; ``RepetitionTime`` → ``TR``; etc).
- ``dicom``: reads DICOM tags (TE = (0018, 0081), TR = (0018, 0080), etc.).
  Requires ``pydicom``; raises an ImportError-style message at first use
  if missing.
- ``h5_attrs``: reads HDF5 root attributes. Used by the FastMRI-style
  pipeline.
- ``ismrmrd_header``: parses the ``ismrmrd_header`` XML of an ISMRMRD/M4Raw
  ``.h5`` (``sequenceParameters`` TE/TR/TI/flipAngle_deg and
  ``systemFieldStrength_T``). The dataset names the file under the
  ``source_path`` subject key, since its images are in-memory tensors.
- ``yaml_inline``: copies ``defaults`` straight through. Use for smoke
  tests and synthetic data where there is no on-disk source.

Strict-vs-lenient
=================

When ``strict=True`` (recommended for production training), a missing
field that is not provided in ``defaults`` raises
:class:`MissingAcquisitionMetadata`. When ``strict=False``, the transform
logs a warning and inserts the default (or NaN if no default is given).

This matches CLAUDE.md #9 — silent fallbacks corrupt Bloch-loss
residuals; surface the failure loudly during training.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from pathlib import Path
from typing import Any, get_args

import torchio as tio

from spectramr.config.schemas.data import (
    AcquisitionMetadataConfigSchema,
    AcquisitionMetadataSource,
)
from spectramr.data.datasets.kspace_volume_stack import read_acquisition_params

logger = logging.getLogger(__name__)


__all__ = [
    "LoadAcquisitionMetadata",
    "MissingAcquisitionMetadata",
]


class MissingAcquisitionMetadata(KeyError):
    """Raised in strict mode when a required metadata field is missing."""


_BIDS_KEYMAP: dict[str, str] = {
    # BIDS canonical → our internal name (units converted in _from_bids_json).
    "EchoTime": "TE",
    "RepetitionTime": "TR",
    "InversionTime": "TI",
    "FlipAngle": "FA",
    "MagneticFieldStrength": "B0",
    "DiffusionBValue": "b_value",
}

_DICOM_KEYMAP: dict[str, tuple[int, int]] = {
    "TE": (0x0018, 0x0081),
    "TR": (0x0018, 0x0080),
    "TI": (0x0018, 0x0082),
    "FA": (0x0018, 0x1314),
    "B0": (0x0018, 0x0087),
}


def _from_bids_json(path: Path) -> dict[str, float]:
    """Parse a BIDS sidecar and emit our internal key names.

    BIDS stores TE / TR / TI in **seconds**; we convert to milliseconds
    (matches AcquisitionParamsSchema and the rest of the codebase).
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, float] = {}
    for bids_key, our_key in _BIDS_KEYMAP.items():
        if bids_key not in raw:
            continue
        val = raw[bids_key]
        # Convert seconds → milliseconds for times.
        if our_key in ("TE", "TR", "TI"):
            val = float(val) * 1000.0
        out[our_key] = float(val)
    return out


def _from_dicom_path(path: Path) -> dict[str, float]:
    """Read DICOM headers and emit our internal key names.

    Reads the first DICOM in the series; assumes acquisition parameters
    are constant across the series.
    """
    try:
        import pydicom  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "AcquisitionMetadataConfigSchema.source='dicom' requires pydicom. "
            "Install with `pip install pydicom`."
        ) from e

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    out: dict[str, float] = {}
    for our_key, tag in _DICOM_KEYMAP.items():
        if tag in ds:
            try:
                out[our_key] = float(ds[tag].value)
            except (TypeError, ValueError):
                continue
    return out


def _from_h5_attrs(path: Path) -> dict[str, float]:
    """Read HDF5 root attributes — FastMRI-style.

    FastMRI bakes `acquisition` / `patient_id` etc. as attrs on the
    root group; common scan parameters live under attrs like
    ``acquisition`` (string identifier) and per-vendor sub-attrs.
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "AcquisitionMetadataConfigSchema.source='h5_attrs' requires h5py. "
            "Install with `pip install h5py`."
        ) from e

    out: dict[str, float] = {}
    with h5py.File(str(path), "r") as f:
        for key in ("TE", "TR", "TI", "FA", "B0"):
            if key in f.attrs:
                try:
                    out[key] = float(f.attrs[key])
                except (TypeError, ValueError):
                    continue
    return out


#: ``read_acquisition_params`` key -> internal name. TE/TR/TI are already ms.
_ISMRMRD_KEYMAP: dict[str, str] = {
    "TE": "TE",
    "TR": "TR",
    "TI": "TI",
    "flip_angle": "FA",
    "field_strength_T": "B0",
}


@functools.lru_cache(maxsize=4096)
def _from_ismrmrd_header(path_str: str) -> tuple[tuple[str, float], ...]:
    """Header values of one file, cached so each sample does not reopen the h5."""
    parsed = read_acquisition_params(Path(path_str))
    return tuple(
        (ours, float(parsed[theirs]))  # type: ignore[arg-type]
        for theirs, ours in _ISMRMRD_KEYMAP.items()
        if isinstance(parsed.get(theirs), float)
    )


def _read_metadata(
    source: AcquisitionMetadataSource,
    image_path: Path | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch to the right reader for the configured source."""
    if source not in get_args(AcquisitionMetadataSource):
        raise ValueError(f"Unknown acquisition metadata source: {source}")
    if source == "yaml_inline" or image_path is None:
        return dict(defaults)
    if source == "bids_json":
        # Sibling .json — strip .nii / .nii.gz first.
        stem = image_path.name
        for suffix in (".nii.gz", ".nii", ".h5", ".npy"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        sidecar = image_path.with_name(f"{stem}.json")
        if not sidecar.exists():
            return {}
        return _from_bids_json(sidecar)
    if source == "dicom":
        return _from_dicom_path(image_path)
    if source == "h5_attrs":
        return _from_h5_attrs(image_path)
    if source == "ismrmrd_header":
        return dict(_from_ismrmrd_header(str(image_path)))
    raise ValueError(f"Unknown acquisition metadata source: {source}")


def _coalesce(
    parsed: dict[str, Any],
    defaults: dict[str, Any],
    fields: list[str],
    strict: bool,
    subject_id: str | None,
) -> dict[str, Any]:
    """Build the final metadata dict, falling back to defaults / NaN."""
    out: dict[str, Any] = {}
    for field in fields:
        if field in parsed and parsed[field] is not None:
            out[field] = parsed[field]
        elif field in defaults:
            out[field] = defaults[field]
        else:
            if strict:
                raise MissingAcquisitionMetadata(
                    f"Required acquisition-metadata field '{field}' is missing "
                    f"for subject '{subject_id}' and no default was provided. "
                    "Set acquisition_metadata.strict=false to fall back to "
                    "NaN, or add the field under acquisition_metadata.defaults."
                )
            logger.warning(
                "Missing acquisition metadata field '%s' for subject '%s'; "
                "falling back to NaN (acquisition_metadata.strict=false).",
                field,
                subject_id,
            )
            out[field] = math.nan
    return out


def _find_image_path(subject: tio.Subject) -> Path | None:
    """Return the first on-disk path among the subject's images.

    TorchIO stores ``image.path`` (a Path) when the image was constructed
    from a file. Subjects created from in-memory tensors have no path, so a
    dataset that builds them names its file under ``source_path`` instead.
    """
    source_path = subject.get("source_path")
    if isinstance(source_path, str | Path):
        return Path(source_path)
    for key in ("input", "mri", "image", "hr", "target"):
        img = subject.get(key)
        if img is None:
            continue
        path = getattr(img, "path", None)
        if path is not None:
            return Path(path)
    # Fallback — scan all images.
    for img in subject.get_images(intensity_only=False):
        path = getattr(img, "path", None)
        if path is not None:
            return Path(path)
    return None


class LoadAcquisitionMetadata(tio.Transform):
    """Attach an ``acquisition_metadata`` dict to every TorchIO Subject.

    Reads scan parameters from the source configured on the
    :class:`AcquisitionMetadataConfigSchema` and stores them on the subject
    under ``subject["acquisition_metadata"]``. Downstream collation will
    propagate the dict into the batch.

    Args:
        config: A populated :class:`AcquisitionMetadataConfigSchema`.

    Raises:
        MissingAcquisitionMetadata: When ``config.strict=True`` and a
            required field is absent from the source.
    """

    def __init__(self, config: AcquisitionMetadataConfigSchema):
        super().__init__()
        self.config = config

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image_path = _find_image_path(subject)
        parsed = _read_metadata(
            self.config.source,
            image_path,
            dict(self.config.defaults),
        )
        subject_id = subject.get("name") or (image_path.name if image_path is not None else None)
        metadata = _coalesce(
            parsed,
            dict(self.config.defaults),
            list(self.config.fields),
            self.config.strict,
            subject_id,
        )
        subject["acquisition_metadata"] = metadata
        return subject
