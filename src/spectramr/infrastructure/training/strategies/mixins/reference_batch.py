"""Read the source-contrast reference off a diffusion batch.

The M4Raw cross-contrast route serves the source contrast as ``kspace_source``
when ``data.pairing.source_layout: separate`` (#2452). The diffusion strategy
hands it to the generator as ``reference``; it never enters the diffusion state.

The stacked layout is refused here. With the source concatenated into the state,
the degradation, data consistency, the reverse loop, the losses, the metrics and
the coil-map estimate each needed their own notion of which channels were the
source, and they disagreed (#2452, #2453, #2454).
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.data.batch_types import read_batch_field


def refuse_stacked_cross_contrast(batch_data: Any) -> None:
    """Raise when a batch carries the source contrast stacked into its channels."""
    if read_batch_field(batch_data, "federated_target_channel_start") is not None:
        raise ValueError(
            "The batch stacks the source contrast onto the target "
            "(data.pairing.source_layout: stacked). Diffusion degrades, "
            "data-consistency-projects and scores every channel of its state, so "
            "the source would be treated as target (#2452). Set "
            "data.pairing.source_layout: separate and "
            "model.model_kwargs.reference_channels to the source's channel count."
        )


def reference_from_batch(
    batch_data: Any, device: Any, flat_batch: int | None = None
) -> torch.Tensor | None:
    """Return ``kspace_source`` as a 4-D tensor aligned with the generator's rows.

    A 5-D ``(B, C, H, W, D)`` tensor is flattened volume-major, the order the
    strategy uses for the target, so row ``v * D + i`` is slice ``i`` of volume
    ``v`` in both.
    """
    value = read_batch_field(batch_data, "kspace_source")
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("data")
    elif hasattr(value, "tensor") and not isinstance(value, torch.Tensor):
        value = value.tensor
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"kspace_source must be a tensor, got {type(value).__name__}.")
    # Only the loader's transform scales the reference; the strategy's own
    # normalization fallback would scale the target and leave the reference raw.
    marker = read_batch_field(batch_data, "kspace_normalized")
    normalized = bool(torch.as_tensor(marker).all()) if marker is not None else False
    if not normalized:
        raise ValueError(
            "kspace_source arrived without the loader's kspace_normalized marker. "
            "Set data.processing.enable_kspace_normalization: true so the "
            "reference and the target are each scaled by the transform."
        )
    if value.dim() == 5:
        b, c, h, w, d = value.shape
        value = value.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
    if flat_batch is not None and value.shape[0] != flat_batch:
        raise ValueError(
            f"kspace_source has {value.shape[0]} row(s) but the generator sees "
            f"{flat_batch}; the reference and the target must be flattened alike."
        )
    return value.to(device)
