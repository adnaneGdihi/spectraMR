"""One reader for a whole reference volume, dispatched on file suffix.

Quality matching reads cohort and high-quality volumes from fastMRI-style ``.h5`` and
from NIfTI. This module is the only dispatch between them (rule 7); the per-format
readers stay in ``io_strategies``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "is_nifti_path",
    "load_reference_volume",
    "nifti_spacing_mm",
    "read_nifti_sidecar",
    "sidecar_path_for",
]

_NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti_path(path: str | Path) -> bool:
    return str(path).lower().endswith(_NIFTI_SUFFIXES)


def sidecar_path_for(path: str | Path) -> Path:
    """The JSON sidecar beside a NIfTI file: ``x.nii.gz`` -> ``x.json``."""
    text = str(path)
    stem = text[: -len(".nii.gz")] if text.lower().endswith(".nii.gz") else text[: -len(".nii")]
    return Path(stem + ".json")


def read_nifti_sidecar(path: str | Path) -> dict[str, Any]:
    """The JSON sidecar of a NIfTI volume, or RAISE.

    Raises rather than returning ``{}``: an absent sidecar must not look like an
    acquisition with no recorded parameters.
    """
    side = sidecar_path_for(path)
    if not side.is_file():
        raise FileNotFoundError(
            f"{path}: no JSON sidecar at {side}. Acquisition parameters (field strength, "
            "TR, TE) are read from it. Declare acquisition_prior: false on the arm to "
            "run without the prior, rather than letting a default stand in."
        )
    doc = json.loads(side.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{side} is not a JSON object")
    return doc


def nifti_spacing_mm(path: str | Path) -> tuple[float, float, float]:
    """``(slice, row, col)`` mm of a NIfTI volume, read from its header only."""
    from spectramr.data.io_strategies import NiftiStrategy

    return NiftiStrategy.read_spacing_mm(path)


def load_reference_volume(path: str | Path) -> dict[str, Any]:
    """Whole volume as ``data`` ``[S, H, W]`` plus its geometry.

    Returns ``{"data", "header", "spacing_mm"}``. An ``.h5`` carries an ISMRMRD
    ``header`` (spacing is derived from it by the caller); a NIfTI carries
    ``spacing_mm`` from its affine and ``header`` ``None``. Exactly one of the two is
    non-``None``.
    """
    if is_nifti_path(path):
        from spectramr.data.io_strategies import NiftiStrategy

        out = NiftiStrategy.load_reference_volume(path)
        return {"data": out["data"], "header": None, "spacing_mm": out["spacing_mm"]}

    from spectramr.data.io_strategies import FastMRIH5Strategy

    out = FastMRIH5Strategy.load_reference_volume(path)
    return {"data": out["data"], "header": out["header"], "spacing_mm": None}
