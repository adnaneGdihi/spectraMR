"""The boundary between TorchIO's ``(C, W, H, D)`` tensors and the canonical simulator layout.

The degradation simulator reads volumes as ``(B, C, D, PE, RO)``: dim -3 the slice or kz axis,
dim -2 phase-encode and dim -1 readout. TorchIO stores the spatial axes as ``(W, H, D)``, and
which of ``W`` and ``H`` is phase-encode is a property of the dataset, so the caller declares
it as ``pe_axis``; it is never inferred.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = [
    "PE_AXES",
    "UNDECLARED_PE_AXIS",
    "PeAxis",
    "canonical_to_tio",
    "check_pe_axis",
    "pe_axis_of_plane_dim",
    "tio_to_canonical",
]

PeAxis = Literal["W", "H"]

#: ``pe_axis`` -> the permutation of a TorchIO ``(C, W, H, D)`` tensor into ``(C, D, PE, RO)``.
_TO_CANONICAL: dict[str, tuple[int, int, int, int]] = {
    "W": (0, 3, 1, 2),
    "H": (0, 3, 2, 1),
}
_TO_TIO: dict[str, tuple[int, ...]] = {
    axis: tuple(perm.index(i) for i in range(4)) for axis, perm in _TO_CANONICAL.items()
}

PE_AXES: tuple[str, ...] = tuple(_TO_CANONICAL)

#: ``data.acquisition.phase_encode_axis`` (a dim of the training batch's ``[..., H, W]`` plane)
#: -> the TorchIO axis holding it: batch dim -2 is TorchIO's first spatial axis ``W``.
_PE_AXIS_OF_PLANE_DIM: dict[int, PeAxis] = {-2: "W", -1: "H"}
#: The axis used when a dataset declares no ``data.acquisition`` block (stamped as such).
UNDECLARED_PE_AXIS: PeAxis = "W"


def pe_axis_of_plane_dim(phase_encode_axis: int | None) -> tuple[PeAxis, str]:
    """The TorchIO PE axis for a declared ``phase_encode_axis``, and where it came from."""
    if phase_encode_axis is None:
        return UNDECLARED_PE_AXIS, "undeclared"
    if phase_encode_axis not in _PE_AXIS_OF_PLANE_DIM:
        raise ValueError(f"phase_encode_axis must be -1 or -2, got {phase_encode_axis!r}")
    return _PE_AXIS_OF_PLANE_DIM[phase_encode_axis], "data.acquisition.phase_encode_axis"


def check_pe_axis(pe_axis: object) -> str:
    """``pe_axis`` if it is a declared TorchIO axis; anything else raises."""
    if pe_axis not in _TO_CANONICAL:
        raise ValueError(f"pe_axis must be one of {list(PE_AXES)}, got {pe_axis!r}")
    return str(pe_axis)


def tio_to_canonical(t: torch.Tensor, pe_axis: PeAxis) -> torch.Tensor:
    """A TorchIO ``(C, W, H, D)`` tensor as one canonical subject ``(1, C, D, PE, RO)``."""
    axis = check_pe_axis(pe_axis)
    if t.dim() != 4:
        raise ValueError(f"expected a TorchIO (C, W, H, D) tensor, got shape {tuple(t.shape)}")
    return t.permute(_TO_CANONICAL[axis]).unsqueeze(0)


def canonical_to_tio(x: torch.Tensor, pe_axis: PeAxis) -> torch.Tensor:
    """The inverse of :func:`tio_to_canonical`: ``(1, C, D, PE, RO)`` back to ``(C, W, H, D)``."""
    axis = check_pe_axis(pe_axis)
    if x.dim() != 5:
        raise ValueError(
            f"expected a canonical (1, C, D, PE, RO) tensor, got shape {tuple(x.shape)}"
        )
    if x.shape[0] != 1:
        raise ValueError(f"a TorchIO tensor holds one subject; got batch {x.shape[0]}")
    return x[0].permute(_TO_TIO[axis])
