"""Micro-batched validation forward, sized by ``validation.loader.chunk_size``.

Validation splits a batch along dim 0 to cap peak memory. Every per-sample
tensor the generator receives must be split with it: slicing the input and the
timesteps while handing each chunk the full-batch ``gen_kwargs`` gave the chunk
full-batch S-maps, masks and measurements (#1914).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

ForwardFn = Callable[[torch.Tensor, Any, dict[str, Any]], Any]


def slice_batch_aligned(
    kwargs: dict[str, Any], start: int, stop: int, batch: int
) -> dict[str, Any]:
    """Slice every tensor whose leading dimension is the batch to ``[start:stop]``.

    A tensor with a different leading size (a broadcast ``(1, ...)`` mask, a
    scalar) is shared by every chunk and passed through unchanged.
    """
    return {
        key: (
            value[start:stop]
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch
            else value
        )
        for key, value in kwargs.items()
    }


def reconstruction_output(out: Any) -> Any:
    """The reconstruction a validation forward returned, out of a tuple or dict output."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("reconstruction", out.get("output", next(iter(out.values()), None)))
    return out


def _primary_output(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("image", out.get("kspace", next(iter(out.values()))))
    return out


def forward_in_chunks(
    forward: ForwardFn,
    x: torch.Tensor,
    timesteps: Any,
    gen_kwargs: dict[str, Any],
    chunk_size: int,
) -> Any:
    """Run ``forward(x, timesteps, gen_kwargs)`` in chunks of ``chunk_size`` samples.

    A batch no larger than one chunk is forwarded as is and its output returned
    untouched; otherwise the primary output of each chunk is concatenated.
    """
    batch = x.shape[0]
    if batch <= chunk_size:
        return forward(x, timesteps, gen_kwargs)
    parts = []
    for start in range(0, batch, chunk_size):
        stop = min(start + chunk_size, batch)
        chunk_ts = (
            timesteps[start:stop]
            if isinstance(timesteps, torch.Tensor)
            and timesteps.dim() > 0
            and timesteps.shape[0] == batch
            else timesteps
        )
        out = forward(x[start:stop], chunk_ts, slice_batch_aligned(gen_kwargs, start, stop, batch))
        parts.append(_primary_output(out))
    return torch.cat(parts, dim=0)


__all__ = ["forward_in_chunks", "reconstruction_output", "slice_batch_aligned"]
