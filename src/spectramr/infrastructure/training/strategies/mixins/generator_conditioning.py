"""Per-sample conditioning ids a generator's ``forward`` names, read off the batch.

A generator that declares ``contrast_idx`` or ``vendor_id`` as a ``forward`` parameter
gets the batch's values; one that names neither gets nothing, so unconditioned
generators are called exactly as before. A named id the batch does not carry raises:
an unconditioned call to a conditioned model would silently drop the conditioning.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field

__all__ = ["CONDITIONING_KEYS", "conditioning_kwargs"]

#: The per-sample ids a generator may name; each is read from the batch by this name.
CONDITIONING_KEYS = ("contrast_idx", "vendor_id")


def _named(generator: nn.Module) -> tuple[str, ...]:
    params = inspect.signature(unwrap_model(generator).forward).parameters
    return tuple(k for k in CONDITIONING_KEYS if k in params)


def conditioning_kwargs(generator: nn.Module, batch: Any, device: Any) -> dict[str, torch.Tensor]:
    """``{name: ids}`` for every conditioning id the generator's ``forward`` names."""
    out: dict[str, torch.Tensor] = {}
    for key in _named(generator):
        value = read_batch_field(batch, key)
        if value is None:
            raise ValueError(
                f"{type(unwrap_model(generator)).__name__}.forward takes {key!r} but the "
                f"batch carries none; the data route must emit it"
            )
        value = torch.as_tensor(value)
        out[key] = value if device is None else value.to(device, non_blocking=True)
    return out
