"""Which volumes a quality match reads, and how each one's geometry is measured.

Both sides of the match, and every path-to-geometry read, go through this module, so the
strategy holds no file-format branches. Manifest parsing lives here as well: the
strategy and the selectors share one reader.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spectramr.data.reference_volume import (
    is_nifti_path,
    load_reference_volume,
    read_nifti_sidecar,
)
from spectramr.infrastructure.physics.acquisition_params import (
    AcquisitionParams,
    acquisition_from_sidecar,
    read_acquisition_params,
)
from spectramr.infrastructure.physics.quality_descriptors import read_spacing_mm

logger = logging.getLogger(__name__)

__all__ = [
    "acquisition_of_path",
    "contrast_matches",
    "hq_source_paths",
    "load_reference_volume",
    "manifest_records",
    "record_path",
    "ref_spacing_mm",
    "select_paths",
    "split_pairs",
]


def manifest_records(manifest_path: Path | str) -> list[dict[str, Any]]:
    """Manifest records, or RAISE. See :func:`cohort_volume_paths` on tolerance."""
    from spectramr.data.split_leakage import read_manifest_records

    records = read_manifest_records(manifest_path)
    if records is None:
        raise FileNotFoundError(
            f"manifest {manifest_path!r} is missing or unreadable. Manifests are "
            "gitignored and never rsynced -- regenerate it on the cluster with the "
            "committed generator (scripts/data/build_mrixfields2026_manifest.py)."
        )
    return records


#: Keys holding a path usable AS IS (the paired manifests).
ABSOLUTE_PATH_KEYS = ("primary_path", "path", "file", "image")
#: Keys holding a FRAGMENT that must be joined to the manifest's data_root
#: (the cluster inventory manifests).
RELATIVE_PATH_KEYS = ("relative_path", "filename")


def contrast_matches(record: dict[str, Any], contrast: str | None) -> bool:
    """Case-insensitive contrast filter; generators spell ``T1w``/``T1W`` and YAMLs ``t1w``."""
    return contrast is None or str(record.get("contrast", "")).casefold() == contrast.casefold()


def manifest_root(manifest_path: Path | str) -> str | None:
    """The manifest's own declared ``data_root``, if it has one.

    ``regenerate_cluster_manifests.py`` and ``build_mrixfields2026_manifest.py`` store
    paths RELATIVE to a root recorded at the top of the document. Ignoring that root
    leaves you holding a bare filename.
    """
    try:
        payload = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and payload.get("data_root"):
        return str(payload["data_root"])
    return None


def record_path(record: dict[str, Any], root: str | None = None) -> str | None:
    """A loadable path for one record.

    The cluster inventory manifests emit ``relative_path`` / ``filename`` plus a
    top-level ``data_root``; the paired manifests emit a ready ``primary_path``.

    Treating ``filename`` as a path -- which a naive key sweep does, because it looks
    like one -- yields a bare BASENAME that either fails to open or, far worse, opens
    a same-named file in the working directory. Hence the two-tier lookup.
    """
    for key in ABSOLUTE_PATH_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    for key in RELATIVE_PATH_KEYS:
        value = record.get(key)
        if value:
            if root is None:
                raise ValueError(
                    f"record carries only the relative key {key!r} ({value!r}) but the "
                    "manifest declares no data_root, so it cannot be resolved to a "
                    "loadable path. Regenerate with the committed generator, which "
                    "writes data_root."
                )
            return str(Path(root) / str(value))
    return None


def select_paths(
    manifest_path: Path | str,
    *,
    field_t: float | None = None,
    contrast: str | None = None,
    tol: float = 1e-6,
) -> list[str]:
    """Volume paths filtered by field strength and/or contrast, or RAISE when none match.

    An unfiltered call returns every record, as ``cohort_volume_paths`` does. A filter
    that matches nothing raises: an empty selection would fit against zero volumes.
    """
    records = manifest_records(manifest_path)
    root = manifest_root(manifest_path)
    hits = [
        p
        for r in records
        if field_t is None
        or abs(float(r.get("field_strength", float("nan"))) - float(field_t)) <= tol
        if contrast_matches(r, contrast)
        if (p := record_path(r, root)) is not None
    ]
    if not hits:
        raise ValueError(
            f"no volumes in {manifest_path!r} match field_t={field_t}, contrast={contrast!r}. "
            f"Fields present: {sorted({float(r['field_strength']) for r in records if 'field_strength' in r})}."
        )
    return hits


def hq_source_paths(
    *,
    pair_layout: str | None,
    field_pairs: Sequence[tuple[str, str]] | None,
    hq_manifest: str | None,
    source_field_t: float | None,
    contrast: str | None,
) -> list[str] | None:
    """The high-quality volumes the fit and the synthesis read, or ``None`` if undeclared.

    Paired arms take the HQ side of the pairs they already selected: the TARGET of an
    explicit restoration manifest, the source-field side of a field-keyed cohort.
    Unpaired arms select from the manifest by field and contrast. ``None`` means the
    arm declared no manifest, and the caller falls back to the loader's batch.
    """
    if field_pairs is not None:
        if pair_layout == "explicit":
            return [hq for hq, _lq in field_pairs]
        return [src for src, _dst in field_pairs]
    if not hq_manifest:
        return None
    return select_paths(hq_manifest, field_t=source_field_t, contrast=contrast)


def ref_spacing_mm(path: str, ref: dict[str, Any]) -> tuple[float, float, float]:
    """``(slice, row, col)`` mm of one loaded reference volume.

    NIfTI carries its spacing in the affine (``spacing_mm``). An h5 carries an ISMRMRD
    header, from which the spacing is read; an unreadable header raises rather than
    defaulting to 1 mm.
    """
    if ref.get("spacing_mm") is not None:
        return tuple(float(v) for v in ref["spacing_mm"])  # type: ignore[return-value]
    if ref.get("header") is None:
        raise ValueError(f"{path}: no affine and no ISMRMRD header, so no voxel spacing.")
    return read_spacing_mm(ref["header"], ref["data"])


def acquisition_of_path(path: str) -> AcquisitionParams:
    """Acquisition parameters of one volume: its header (h5) or its sidecar (NIfTI).

    A NIfTI reads only the JSON sidecar, not the voxels. Raises when either is absent.
    """
    if is_nifti_path(path):
        return acquisition_from_sidecar(read_nifti_sidecar(path))
    return read_acquisition_params(load_reference_volume(path)["header"])


def split_pairs(
    pairs: Sequence[tuple[str, str]],
    held_out_fraction: float,
    *,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(fit_pairs, held_out_pairs)``: a seeded, subject-level hold-out.

    One pair is one subject, so holding out pairs holds out subjects. Raises unless at
    least one pair is left to fit on and at least one to score.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError(f"held_out_fraction must lie in (0, 1); got {held_out_fraction}")
    ordered = sorted(pairs)
    n_held = round(len(ordered) * held_out_fraction)
    if n_held < 1 or n_held >= len(ordered):
        raise ValueError(
            f"held_out_fraction {held_out_fraction} over {len(ordered)} pair(s) leaves "
            f"{len(ordered) - n_held} to fit on and {n_held} to score; both must be at "
            "least one."
        )
    held_idx = set(random.Random(seed).sample(range(len(ordered)), n_held))
    fit = [p for i, p in enumerate(ordered) if i not in held_idx]
    held = [p for i, p in enumerate(ordered) if i in held_idx]
    return fit, held
