r"""Per-scan ESPIRiT coil maps for the M4Raw repetition route: one recipe, one file format.

The null-space split (``target_mode: nullspace_r2r``) and the coil-manifold
metrics need complex ESPIRiT maps, and estimating them costs 0.8-2.3 s per
256x256 slice single-threaded, paid on every group load because the TorchIO
``Queue`` respawns its workers each epoch (#2339). A coil map is a property of
the scan, so it is estimated once and read back.

This module is the only owner of three things, each of which would silently
disagree with itself if written twice:

* **The recipe.** ``all_reps_mean`` (the default) estimates from the
  phase-aligned mean of *every* repetition, which is lower-noise than any one
  of them; the split's independence argument (``PAPER.md`` Theorem 1) needs the
  maps independent of the *noise* it splits only to first order.
  ``input_repetition`` estimates from the repetition the dataset serves as
  input alone, so an arm graded against the leave-one-out mean of the others
  never sees maps built from its reference. :func:`estimate_group_maps` is
  called by the precompute script and by the dataset's
  ``coil_maps_missing: estimate`` fallback, so a run before and after the files
  exist sees the same maps.
* **The key.** One file per repetition group, ``<patient>_<contrast>.pt``
  (:func:`~spectramr.data.datasets.m4raw_identity.repetition_group_key`), not
  per repetition file: the maps average all of them, and a consumer holding
  repetition 02's ``file_id`` must find the same file as one holding 01's.
* **The format and its validation.** :func:`load_coil_maps` raises on a shape,
  coil-count, calibration or recipe mismatch rather than serving maps estimated
  for a different arm, and never recomputes silently.

The recipes themselves are named in
:mod:`~spectramr.data.datasets.m4raw_coil_map_recipe`; the per-load choice
between a file and an estimate is
:class:`~spectramr.data.datasets.m4raw_coil_map_source.CoilMapSource`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from spectramr.data.datasets.m4raw_coil_map_recipe import (
    DEFAULT_RECIPE,
    VALID_RECIPES,
    check_recipe,
    recipe_source,
)
from spectramr.data.datasets.m4raw_identity import repetition_group_key

logger = logging.getLogger(__name__)

__all__ = [
    "MAP_FORMAT_VERSION",
    "CoilMapMismatchError",
    "DEFAULT_RECIPE",
    "VALID_RECIPES",
    "calibration_tag",
    "check_map_meta",
    "estimate_group_maps",
    "group_map_path",
    "group_repetition_files",
    "load_coil_maps",
    "load_group_kspace",
    "map_key",
    "map_path",
    "read_map_meta",
    "save_coil_maps",
]

#: Bumped when the payload layout changes; a reader refuses any other version.
MAP_FORMAT_VERSION: int = 1

#: Calibration fields a file must agree with its consumer on. Each changes the
#: maps, so a mismatch on any of them is a different estimator, not a nuance.
_CALIBRATION_FIELDS: tuple[str, ...] = ("acs_size", "kernel_size", "eigen_threshold")


class CoilMapMismatchError(ValueError):
    """A precomputed map file does not describe what its consumer declared."""


def calibration_tag(calibration: dict[str, Any]) -> str:
    """``acs_size=24;kernel_size=6;eigen_threshold=0.95`` -- comparable across processes."""
    return ";".join(f"{k}={calibration[k]}" for k in _CALIBRATION_FIELDS)


def map_key(file_id: str) -> str:
    """The file stem for any repetition of a group: ``<patient>_<contrast>``."""
    return repetition_group_key(Path(str(file_id)).stem)


def group_map_path(root: str | Path, group_key: str) -> Path:
    """``<root>/<group_key>.pt`` for a key :func:`map_key` already produced."""
    return Path(root) / f"{group_key}.pt"


def map_path(root: str | Path, file_id: str) -> Path:
    """``<root>/<map_key(file_id)>.pt``; takes a repetition's ``file_id``, never a key.

    Applying the grouping rule to a key would strip two more characters, so
    code holding a key calls :func:`group_map_path` instead.
    """
    return group_map_path(root, map_key(file_id))


def group_repetition_files(paths: Iterable[str | Path]) -> dict[str, list[Path]]:
    """Repetition files grouped by :func:`map_key`, each group sorted."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        p = Path(path)
        groups.setdefault(map_key(p.stem), []).append(p)
    return {key: sorted(files) for key, files in sorted(groups.items())}


def load_group_kspace(paths: Sequence[str | Path]) -> list[torch.Tensor]:
    """Raw complex k-space of every readable repetition, through the dataset's reader.

    Raises the dataset's skip error when none load; a partial load is logged
    by the reader and shows up as ``num_repetitions_averaged`` in the metadata.
    """
    from spectramr.data.datasets.m4raw_dataset import _load_reps_or_skip

    files = [Path(p) for p in paths]
    return _load_reps_or_skip(files, f"coil maps for {map_key(files[0].stem)}")


def estimate_group_maps(
    kspace_reps: Sequence[torch.Tensor],
    calibration: dict[str, Any],
    recipe: str = DEFAULT_RECIPE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``(S, C, H, W)`` complex64 maps of one repetition group, by ``recipe``.

    Args:
        kspace_reps: raw ``(S, C, H, W)`` (or ``(C, H, W)``) coil k-space, one
            tensor per repetition, all the same shape, the input repetition first.
        calibration: ``acs_size``, ``kernel_size``, ``eigen_threshold``.
        recipe: one of :data:`VALID_RECIPES`.

    Returns:
        ``(maps, meta)``; ``meta`` carries the recipe, the calibration, the
        repetition count and the phase-encode support the maps were estimated over.
    """
    from spectramr.data.datasets.m4raw_dataset import _pe_support
    from spectramr.infrastructure.physics.scan_coil_calibration import (
        as_slice_batch,
        calibrate_scan,
    )

    if not kspace_reps:
        raise ValueError("estimate_group_maps needs at least one repetition.")
    shapes = {tuple(r.shape) for r in kspace_reps}
    if len(shapes) != 1:
        raise ValueError(f"repetitions of one group disagree on shape: {sorted(shapes)}.")
    source = recipe_source(recipe)(kspace_reps)
    slices = as_slice_batch(source)
    support = _pe_support(slices)
    maps = calibrate_scan(
        slices, support, fit_covariance=False, **{k: calibration[k] for k in _CALIBRATION_FIELDS}
    ).maps.to(torch.complex64)
    meta = {
        "format_version": MAP_FORMAT_VERSION,
        "recipe": recipe,
        **{k: calibration[k] for k in _CALIBRATION_FIELDS},
        "num_repetitions_averaged": len(kspace_reps) if recipe == "all_reps_mean" else 1,
        "shape": list(maps.shape),
        "pe_support_width": int(support.sum()),
        "pe_support_fraction": float(support.float().mean()),
    }
    return maps, meta


def save_coil_maps(path: str | Path, maps: torch.Tensor, meta: dict[str, Any]) -> Path:
    """Write ``{"maps", "meta"}`` atomically, so a killed job never leaves half a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save({"maps": maps.detach().cpu().to(torch.complex64), "meta": dict(meta)}, tmp)
    os.replace(tmp, target)
    return target


def check_map_meta(
    path: str | Path, meta: dict[str, Any], *, calibration: dict[str, Any], recipe: str
) -> None:
    """Raise :class:`CoilMapMismatchError` unless ``meta`` describes this consumer's maps.

    The one check a stored file passes before it is served, and before the
    precompute treats an existing file as done: version, recipe (a file without
    one was written by ``all_reps_mean``) and every calibration field.
    """
    if meta.get("format_version") != MAP_FORMAT_VERSION:
        raise CoilMapMismatchError(
            f"{path}: format_version {meta.get('format_version')!r}, reader expects "
            f"{MAP_FORMAT_VERSION}. Re-run scripts/precompute_m4raw_coil_maps.py."
        )
    stored_recipe = meta.get("recipe", DEFAULT_RECIPE)
    if stored_recipe != check_recipe(recipe):
        raise CoilMapMismatchError(
            f"{path}: estimated with recipe {stored_recipe!r}, but this consumer declares "
            f"{recipe!r}. Maps from other repetitions are a different estimator and, under "
            "a leave-one-out reference, may be built from the reference itself; point "
            "data.coil_maps_root at a store written with the arm's recipe."
        )
    for field in _CALIBRATION_FIELDS:
        if meta.get(field) != calibration[field]:
            raise CoilMapMismatchError(
                f"{path}: estimated with {field}={meta.get(field)!r}, but this consumer "
                f"declares {field}={calibration[field]!r}. Maps from a different "
                "calibration are a different estimator; re-run the precompute with the "
                "arm's value or change the arm."
            )


def read_map_meta(path: str | Path) -> dict[str, Any]:
    """The ``meta`` dict of a stored file, without materialising its maps."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    return dict(payload.get("meta") or {})


def load_coil_maps(
    path: str | Path,
    *,
    expect_shape: Sequence[int],
    calibration: dict[str, Any],
    slice_index: int | None = None,
    recipe: str = DEFAULT_RECIPE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Read and validate one map file.

    Args:
        path: the file.
        expect_shape: ``(S, C, H, W)`` of the k-space the maps must describe;
            on the slice route ``(1, C, H, W)`` with ``slice_index`` set.
        calibration: the consumer's ``acs_size``/``kernel_size``/``eigen_threshold``.
        slice_index: select one slice of a whole-volume file.
        recipe: the consumer's :data:`VALID_RECIPES` entry; a file without one
            was written by ``all_reps_mean``.

    Raises:
        FileNotFoundError: no file (the caller decides what that means).
        CoilMapMismatchError: wrong version, dtype, shape, coil count,
            calibration or recipe -- never answered by recomputing.
    """
    # Memory-mapped, and the selection below is copied out of the mapping: a slice
    # record needs 1/S of the file, and a view would keep the whole volume's
    # storage alive and ship all of it through the loader's shared memory (#2673).
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    maps, meta = payload.get("maps"), payload.get("meta") or {}
    check_map_meta(path, meta, calibration=calibration, recipe=recipe)
    if not isinstance(maps, torch.Tensor) or not torch.is_complex(maps) or maps.dim() != 4:
        raise CoilMapMismatchError(f"{path}: 'maps' is not a (S, C, H, W) complex tensor.")
    expect = tuple(int(d) for d in expect_shape)
    if slice_index is not None:
        if maps.shape[1:] != torch.Size(expect[1:]):
            raise CoilMapMismatchError(
                f"{path}: maps {tuple(maps.shape)} do not match k-space slices {expect[1:]} "
                "(coil count or matrix size)."
            )
        if not 0 <= slice_index < maps.shape[0]:
            raise CoilMapMismatchError(
                f"{path}: slice {slice_index} is outside the {maps.shape[0]} slices on file."
            )
        maps = maps[slice_index : slice_index + 1]
    elif maps.shape != torch.Size(expect):
        raise CoilMapMismatchError(
            f"{path}: maps {tuple(maps.shape)} do not match the k-space {expect} "
            "(slice count, coil count or matrix size)."
        )
    return maps.clone(), meta
