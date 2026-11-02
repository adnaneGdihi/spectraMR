"""The 2-D NEX pseudo-GT reference of one repetition group, read from its ``.h5`` files.

The reads go through :mod:`spectramr.data.datasets.kspace_volume_stack` and the physics
through :func:`~spectramr.infrastructure.physics.m4raw_pseudo_gt.pseudo_gt_from_kspace`;
this module only picks the slice and composes the two.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from spectramr.data.datasets import kspace_volume_stack as io
from spectramr.infrastructure.physics.m4raw_pseudo_gt import nex_merge, pseudo_gt_from_kspace

logger = logging.getLogger(__name__)

__all__ = ["synthesize_pseudo_gt"]


def _failed(rep_paths: list[Path], errors: list[str]) -> RuntimeError:
    return RuntimeError(
        f"All {len(rep_paths)} repetition files failed to load.\n"
        "Errors:\n" + "\n".join(f"  • {e}" for e in errors)
    )


def _probe_shapes(rep_paths: list[Path], errors: list[str]) -> dict[Path, tuple[int, ...]]:
    shapes: dict[Path, tuple[int, ...]] = {}
    for p in rep_paths:
        try:
            shapes[p] = io.kspace_shape_from_h5(p)
        except FileNotFoundError as exc:
            errors.append(f"{p.name}: {exc}")
            logger.warning("File not found: %s", p)
        except KeyError as exc:
            errors.append(f"{p.name}: {exc}")
            logger.warning("No kspace dataset in %s: %s", p, exc)
        except Exception as exc:  # defensive: surfaced via errors[]
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
            logger.warning("Failed to probe %s: %s: %s", p, type(exc).__name__, exc)
    return shapes


def synthesize_pseudo_gt(
    rep_paths: list[Path],
    target_slice: int | None = None,
    device: torch.device | None = None,
    smaps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """NEX pseudo-GT of one slice (the central one by default) across ``rep_paths``.

    Two cheap passes (#616): the slice count comes from the HDF5 headers, then only the
    chosen slice of each rep is read. A rep that fails either pass is named and skipped;
    all failing raises. Reps whose ``(Coils, H, W)`` differ are not one NEX set, so the
    largest consistent subset is kept and the drops are named at WARNING.

    Args:
        rep_paths: The repetitions of one anatomy group (one subject, one contrast).
        target_slice: Slice to read; ``None`` is the central slice.
        device: Device of the returned tensors (CPU when ``None``).
        smaps: Maps ``(1, Coils, H, W)`` to reuse instead of re-estimating ESPIRiT.

    Returns:
        ``(coil_images, smaps, x_gt_mag, p99)`` of
        :func:`~spectramr.infrastructure.physics.m4raw_pseudo_gt.pseudo_gt_from_kspace`
        with ``B = 1``.
    """
    errors: list[str] = []
    shapes = _probe_shapes(rep_paths, errors)
    if not shapes:
        raise _failed(rep_paths, errors)

    # The slice count comes from the first rep that probed.
    n_slices = next(iter(shapes.values()))[0]
    if target_slice is None:
        target_slice = n_slices // 2
    target_slice = min(target_slice, n_slices - 1)

    slices: list[torch.Tensor] = []
    loaded: list[Path] = []
    for p, shape in shapes.items():
        try:
            ks = io.load_kspace_from_h5(p, slice_index=min(target_slice, shape[0] - 1))
        except Exception as exc:  # defensive: surfaced via errors[]
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
            logger.warning("Failed to load %s: %s: %s", p, type(exc).__name__, exc)
            continue
        if ks.dim() == 2:  # single-coil (H, W) still needs a coil axis
            ks = ks.unsqueeze(0)
        elif ks.dim() != 3:
            raise ValueError(f"Unexpected kspace slice shape: {ks.shape}")
        slices.append(ks)
        loaded.append(p)
        logger.debug("Loaded rep %s slice %d: shape=%s", p.name, target_slice, ks.shape)
    if not slices:
        raise _failed(rep_paths, errors)

    y_gt, _ = nex_merge(slices, loaded)
    logger.info("Pseudo-GT slice %d/%d", target_slice, n_slices)
    y_gt = y_gt.unsqueeze(0).to(device or torch.device("cpu"))
    return pseudo_gt_from_kspace(y_gt, smaps=smaps)
