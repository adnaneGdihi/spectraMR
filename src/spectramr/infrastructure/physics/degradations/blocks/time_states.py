"""Assemble k-space whose lines come from different time states, without stacking the states."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from spectramr.infrastructure.physics.fft_ops import fftnc

__all__ = ["assemble_from_states"]

_ENCODING_DIMS = {"slice": (-2, -1), "volume": (-3, -2, -1)}


def assemble_from_states(
    n_states: int,
    state_index: torch.Tensor,
    render_state: Callable[[int], torch.Tensor],
    *,
    encoding: Literal["slice", "volume"],
) -> torch.Tensor:
    """k-space ``[B, C, D, H, W]`` where line ``(d, h)`` is taken from state ``state_index[d, h]``.

    States are rendered one at a time and scattered into one accumulator, so peak memory is
    O(volume), not O(n_states * volume). ``state_index`` must be a CPU tensor. The caller applies the inverse FFT over the same dims.
    """
    if encoding not in _ENCODING_DIMS:
        raise ValueError(f"encoding must be one of {sorted(_ENCODING_DIMS)}, got {encoding!r}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if state_index.dim() != 2 or state_index.dtype != torch.long:
        raise ValueError("state_index must be a long tensor of shape [D, H]")
    if state_index.device.type != "cpu":
        raise ValueError(
            f"state_index must be a CPU tensor (static schedule, host-side validation), "
            f"got device {state_index.device}"
        )
    if int(state_index.min()) < 0 or int(state_index.max()) >= n_states:
        raise ValueError(f"state_index values must lie in [0, {n_states})")
    dims = _ENCODING_DIMS[encoding]
    out: torch.Tensor | None = None
    first_shape: tuple[int, ...] | None = None
    sched: torch.Tensor | None = None
    for p in range(n_states):
        img = render_state(p)
        if img.dim() != 5 or tuple(img.shape[-3:-1]) != tuple(state_index.shape):
            raise ValueError(
                f"render_state({p}) must be [B, C, D, H, W] with (D, H)={tuple(state_index.shape)}, "
                f"got {tuple(img.shape)}"
            )
        if first_shape is None:
            first_shape = tuple(img.shape)
        elif tuple(img.shape) != first_shape:
            raise ValueError(
                f"render_state({p}) shape {tuple(img.shape)} differs from state 0 {first_shape}"
            )
        k = fftnc(img, dims=dims)
        del img
        if sched is None:
            sched = state_index.to(k.device)
        sel = (sched == p)[None, None, :, :, None]
        out = torch.where(sel, k, torch.zeros_like(k)) if out is None else torch.where(sel, k, out)
        del k
    assert out is not None
    return out
