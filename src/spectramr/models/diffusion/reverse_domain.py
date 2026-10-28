"""Linear complex k-space views of a reverse-loop estimate.

The sampler holds k-space as interleaved real channels, ``log1p``-compressed in
magnitude when ``data.processing.enable_log_scaling`` is on. The physics steps of
the reverse loop -- a coil-consistency operator, a total-variation proximal
step, a Wiener blend -- are defined on LINEAR complex k-space, and applying any
of them to compressed values computes a different operator. These two functions
are the round trip; the compression itself is owned by
``spectramr.data.transforms.normalization``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spectramr.data.transforms.normalization import compress_kspace_log, decompress_kspace_log

if TYPE_CHECKING:
    from torch import Tensor


def linear_kspace(x: Tensor, *, log_scaled: bool) -> Tensor:
    """Complex ``[B, C//2, H, W]`` (or ``x`` itself when complex) in linear magnitude."""
    from spectramr.models.diffusion.kspace_process import paired_complex

    lin = decompress_kspace_log(x, channel_dim=1) if log_scaled else x
    return paired_complex(lin)


def stacked_kspace(z: Tensor, like: Tensor, *, log_scaled: bool) -> Tensor:
    """Inverse of :func:`linear_kspace`: ``z`` in the layout and domain of ``like``."""
    out = z if torch.is_complex(like) else torch.stack([z.real, z.imag], dim=2).flatten(1, 2)
    out = out.to(like.dtype)
    return compress_kspace_log(out, channel_dim=1) if log_scaled else out


__all__ = ["linear_kspace", "stacked_kspace"]
