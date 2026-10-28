"""Source-contrast conditioning for the k-space cold-diffusion generator.

A cross-contrast arm (``data.pairing.source_layout: separate``) reconstructs one
target contrast and receives a second, fully sampled contrast as a reference.
The reference is concatenated onto the backbone input and nothing else: it never
enters the diffusion state, so the forward degradation, data consistency, the
reverse loop, the losses and the metrics all see the target alone (#2452).
"""

from __future__ import annotations

from typing import Any

import torch

#: Spellings that described the source channels while the source was stacked
#: into the diffusion state. Each had a different consumer and they disagreed.
_RETIRED_KWARGS: dict[str, str] = {
    "prior_channel_range": (
        "the source contrast no longer shares the diffusion state, so there is "
        "no channel range to keep sampled"
    ),
    "target_channels": ("every output channel is the target; TargetAwareFSDC takes out_channels"),
}


def resolve_reference_channels(kwargs: dict[str, Any]) -> int:
    """Read and validate ``reference_channels`` from the generator kwargs.

    Raises on a retired spelling so an arm written against the stacked layout
    fails at build instead of training on a layout it does not describe.
    """
    for key, why in _RETIRED_KWARGS.items():
        if key in kwargs:
            raise ValueError(
                f"model.model_kwargs.{key} was removed: {why}. Declare "
                "data.pairing.source_layout: separate and "
                "model.model_kwargs.reference_channels instead (#2452)."
            )
    # Read, not popped: a backbone that encodes the reference separately
    # (``reference_guided_unet``) reads the same declaration from its kwargs.
    value = kwargs.get("reference_channels", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"model.model_kwargs.reference_channels must be a non-negative int, got {value!r}."
        )
    return value


def prepend_reference(
    x: torch.Tensor, reference: torch.Tensor | None, channels: int
) -> torch.Tensor:
    """Return ``cat([reference, x])`` on the channel axis, validating the contract.

    ``channels`` is the width the backbone was built for. A model built without
    a reference refuses one, and a model built with one refuses to run without
    it: either mismatch would otherwise change the arm's hypothesis silently.
    """
    if channels == 0:
        if reference is not None:
            raise ValueError(
                "The generator received a source-contrast reference but was built "
                "with reference_channels=0. Set model.model_kwargs.reference_channels "
                f"to {reference.shape[1]}."
            )
        return x
    if reference is None:
        raise ValueError(
            f"The generator was built with reference_channels={channels} and received "
            "no reference. The batch must carry kspace_source "
            "(data.pairing.source_layout: separate)."
        )
    if reference.dim() != x.dim() or reference.shape[1] != channels:
        raise ValueError(
            f"reference has shape {tuple(reference.shape)}; expected {x.dim()}-D with "
            f"{channels} channels to match the input {tuple(x.shape)}."
        )
    if reference.shape[0] != x.shape[0] or reference.shape[2:] != x.shape[2:]:
        raise ValueError(
            f"reference {tuple(reference.shape)} does not match the input "
            f"{tuple(x.shape)} in batch or spatial size."
        )
    if torch.is_complex(reference) != torch.is_complex(x):
        raise ValueError(
            f"reference dtype {reference.dtype} and input dtype {x.dtype} disagree "
            "on complex versus real-interleaved k-space."
        )
    return torch.cat([reference.to(device=x.device, dtype=x.dtype), x], dim=1)
