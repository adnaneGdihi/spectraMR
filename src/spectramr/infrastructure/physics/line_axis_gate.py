"""First-batch check that a line mask removes lines along the data's phase-encode axis.

The data says which axis was acquired line by line: a phase-encode line the
scanner never acquired (partial phase FOV, partial Fourier) is exactly zero in
every coil and slice, and the readout axis carries no such lines. A line mask
whose lines run along the other axis undersamples the readout, which is what
every M4Raw and fastMRI-brain ``kspace_filling`` arm did before issue #2160.

The verdict is one of:

* ``agree`` -- the structural-zero lines and the mask's lines share an axis;
* ``inconclusive`` -- the batch has no structural-zero line (fully sampled or
  synthetic data), or zero lines on both axes, so the data cannot say. Recorded,
  never counted as agreement;
* ``not_line_structured`` -- the mask is a point pattern or trajectory, or keeps
  everything, so it has no line axis to compare.

A disagreement raises :class:`LineAxisMismatchError`.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.infrastructure.physics.line_axis import LINE_AXIS_DIM, MEASURE_SCRIPT
from spectramr.infrastructure.physics.pe_support import pe_support_of

__all__ = ["LineAxisMismatchError", "check_line_axis", "structural_zero_lines"]


class LineAxisMismatchError(ValueError):
    """The mask removes lines along an axis the data acquired fully."""


def structural_zero_lines(kspace: torch.Tensor) -> dict[int, int]:
    """Exactly-zero lines of a ``[..., H, W]`` batch, per plane dimension.

    A column (dim -1 line) counts when it is zero across every leading
    dimension and every row, which is :func:`pe_support_of`'s definition of an
    unacquired line; a row is the same read on the transposed plane.
    """
    if kspace.dim() < 2:
        raise ValueError(f"expected a [..., H, W] k-space batch, got {tuple(kspace.shape)}.")
    plane = kspace.detach()
    return {
        -1: int((~pe_support_of(plane)).sum()),
        -2: int((~pe_support_of(plane.transpose(-1, -2))).sum()),
    }


def _mask_line_dim(mask_plane: torch.Tensor) -> int | None:
    """The dimension the mask's kept lines index, or None when it has none."""
    from spectramr.models.diffusion.reveal_attribution import resolve_line_axis

    plane = mask_plane > 0
    if bool(plane.all()) or not bool(plane.any()):
        return None
    try:
        return LINE_AXIS_DIM[resolve_line_axis(plane)]
    except ValueError:
        return None


def check_line_axis(
    kspace: torch.Tensor, mask_plane: torch.Tensor, *, split: str
) -> dict[str, Any]:
    """Compare the data's structural-zero line axis with the mask's line axis.

    Args:
        kspace: the clean batch, plane last (``[..., H, W]``), before masking.
        mask_plane: the mask at its sparsest rung, ``[H, W]`` (leading
            singleton dimensions are reduced).
        split: ``train`` or ``validation``, for the record and the message.

    Returns:
        The provenance record: ``verdict``, ``data_axis``, ``mask_axis`` and the
        per-dimension zero-line counts.

    Raises:
        LineAxisMismatchError: the data's zero lines lie on one axis and the
            mask removes lines along the other.
    """
    zeros = structural_zero_lines(kspace)
    mask_dim = _mask_line_dim(mask_plane)
    zero_dims = [dim for dim, count in zeros.items() if count > 0]
    data_dim = zero_dims[0] if len(zero_dims) == 1 else None
    record: dict[str, Any] = {
        "split": split,
        "data_axis": data_dim,
        "mask_axis": mask_dim,
        "zero_lines_dim_-1": zeros[-1],
        "zero_lines_dim_-2": zeros[-2],
        "shape": list(kspace.shape[-2:]),
    }
    if mask_dim is None:
        record["verdict"] = "not_line_structured"
    elif data_dim is None:
        record["verdict"] = "inconclusive"
    elif data_dim == mask_dim:
        record["verdict"] = "agree"
    else:
        raise LineAxisMismatchError(
            f"[{split}] the first batch {tuple(kspace.shape)} has {zeros[data_dim]} "
            f"never-acquired lines on dim {data_dim} (its phase-encode axis) and none on "
            f"dim {mask_dim}, but the mask removes lines along dim {mask_dim}: it "
            "undersamples the readout axis. Declare data.acquisition.phase_encode_axis: "
            f"{data_dim} (measure it with `python {MEASURE_SCRIPT} --arm <yaml>`)."
        )
    return record
