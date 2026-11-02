"""The item every phase-contrast dataset emits: one readout slice of one subject.

A 3D Cartesian readout is fully encoded, so an inverse FFT along ``kx`` splits a
4D-flow acquisition into independent readout slices, each a 2D+t problem over the
two phase-encode axes ``(H, W)`` per velocity encoding; a 2D through-plane
acquisition is the same problem over ``(ky, kx)``. Slices, not volumes, are the
item for training and validation alike: a whole subject's coil k-space does not
fit in memory, so a strategy assembles volumes from consecutive slices of one
``subject_id`` (the validation loader is unshuffled) and scores them once
``n_slices`` have arrived.

``input`` and ``target`` are complex ``[C, E, T, H, W]``, 6-D once batched, which
the generic validation preprocess (complex interleave and depth flatten act on
4-D/5-D tensors only) leaves untouched. Everything else rides as extra keys.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class FlowSample(TypedDict, total=False):
    """Keys, dtypes and layouts of one flow item (``E`` encodings, ``n_dir = E - 1``)."""

    input: torch.Tensor  # complex64 [C, E, T, H, W], undersampled k-space
    target: torch.Tensor  # complex64 [C, E, T, H, W], fully sampled (or the acquired data)
    mask: torch.Tensor  # float32 [E, T, H, W], 1 where sampled
    coil_maps: torch.Tensor  # complex64 [C, H, W]
    velocity: torch.Tensor  # float32 [n_dir, T, H, W], m/s; absent without a reference
    image_ref: torch.Tensor  # complex64 [E, T, H, W]; absent without a reference
    vessel_mask: torch.Tensor  # float32 [H, W], frozen per subject
    static_mask: torch.Tensor  # float32 [H, W], tissue with zero true velocity
    inlet_mask: torch.Tensor  # float32 [n_dir, H, W], area-weighted normal in mm^2
    outlet_mask: torch.Tensor  # float32 [n_dir, H, W]
    venc: torch.Tensor  # float32 [n_dir], m/s, owned by the data
    voxel_size_mm: torch.Tensor  # float32 [3], in (readout, H, W) order
    encoding_scheme: str
    subject_id: str
    slice_index: int
    n_slices: int


#: Keys every flow item carries, with or without a velocity reference.
REQUIRED_FLOW_KEYS: frozenset[str] = frozenset(
    {
        "input",
        "target",
        "mask",
        "coil_maps",
        "vessel_mask",
        "static_mask",
        "venc",
        "voxel_size_mm",
        "encoding_scheme",
        "subject_id",
        "slice_index",
        "n_slices",
    }
)


def check_flow_sample(sample: dict) -> None:
    """Raise if ``sample`` breaks the contract a flow strategy relies on."""
    missing = REQUIRED_FLOW_KEYS - set(sample)
    if missing:
        raise KeyError(f"flow sample lacks {sorted(missing)}")
    for key in ("input", "target"):
        value = sample[key]
        if not (torch.is_complex(value) and value.ndim == 5):
            raise ValueError(
                f"flow {key!r} must be complex [C, E, T, H, W]; got {value.dtype} {tuple(value.shape)}"
            )
    n_enc = sample["input"].shape[1]
    if sample["venc"].shape != (n_enc - 1,):
        raise ValueError(f"venc must be [n_dir={n_enc - 1}]; got {tuple(sample['venc'].shape)}")


__all__ = ["REQUIRED_FLOW_KEYS", "FlowSample", "check_flow_sample"]
