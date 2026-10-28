"""Shared plumbing of the hardware catalogue axes: line timing and per-line phases.

A 2-D slice runs as a one-slice ``SLICE_2D`` volume (``motion_timing.run_as_volume``).
``shift_lines`` gives every ``(slice or kz, ky)`` line its own displacement along any of the
encoding axes through the shift theorem: ``fftnc(x) * exp(-2 pi i f d)`` shifts by ``+d`` px
(Bernstein, Handbook of MRI Pulse Sequences, 2004, 1.2).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import ENCODING_DIMS
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc

__all__ = [
    "line_taus",
    "shift_lines",
    "shot_taus",
]


def line_taus(geometry: AcquisitionGeometry, d: int, h: int, device: torch.device) -> Tensor:
    """Normalised acquisition time ``[D, H]`` in [0, 1] of each line (last line = 1)."""
    t = line_times(geometry, d, h, tr_s=1.0).to(torch.float64)
    return (t / t.max().clamp(min=1e-30)).to(device)


def shot_taus(geometry: AcquisitionGeometry, d: int, h: int, device: torch.device) -> Tensor:
    """Normalised time ``[D, H]`` of each EPI shot, constant along ky; the last shot is 1.

    A shot (echo train) is the slice, or the kz partition: it lasts tens of ms, so a scan-scale
    drift is constant within it. Its time is the mean acquisition time of its lines.
    """
    t = line_times(geometry, d, h, tr_s=1.0).to(torch.float64).mean(dim=-1, keepdim=True)
    return (t / t.max().clamp(min=1e-30)).expand(d, h).to(device)


def shift_lines(
    x: Tensor,
    geometry: AcquisitionGeometry,
    *,
    kz: Tensor | None = None,
    pe: Tensor | None = None,
    ro: Tensor | None = None,
) -> Tensor:
    """Displace line ``(d, ky)`` by ``kz/pe/ro[b, d, ky]`` px along that axis (``[B or 1, D, H]``)."""
    dims = ENCODING_DIMS[geometry.mode]
    if kz is not None and geometry.mode is not AcquisitionMode.VOLUME_3D:
        raise ValueError("a kz displacement needs a 3-D (kz, ky) encoding")
    rdt = x.real.dtype
    d, h, w = x.shape[-3:]
    axes = ((kz, d, 2), (pe, h, 3), (ro, w, 4))
    arg = torch.zeros((), device=x.device, dtype=rdt)
    for shift, size, dim in axes:
        if shift is None:
            continue
        view = [1] * 5
        view[dim] = size
        f = centred_freqs(size, x.device, dtype=rdt).view(view)
        arg = arg + f * shift.to(device=x.device, dtype=rdt)[:, None, :, :, None]
    return ifftnc(fftnc(x, dims=dims) * torch.exp(-2j * math.pi * arg), dims=dims)
