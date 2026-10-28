"""Acquisition time (s) of every phase-encode line, and its grouping into shots.

``times`` has shape ``[D, H]``: D = slice (multislice) or kz (3-D), H = ky.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    SliceOrder,
)

__all__ = ["line_times", "shot_index"]


def _slice_2d(geom: AcquisitionGeometry, d: int, h: int) -> torch.Tensor:
    if d != 1:
        raise ValueError(f"SLICE_2D needs n_slices_or_kz == 1, got {d}")
    return torch.arange(h, dtype=torch.long)[None, :]


def _multislice(geom: AcquisitionGeometry, d: int, h: int) -> torch.Tensor:
    ky = torch.arange(h, dtype=torch.long)[None, :]
    if geom.slice_order is SliceOrder.SEQUENTIAL:
        return torch.arange(d, dtype=torch.long)[:, None] * h + ky
    order = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    rank = torch.empty(d, dtype=torch.long)
    rank[order] = torch.arange(d)
    # Interleaved: all slices share each TR, offset by rank/D of a TR.
    return ky + rank[:, None].to(torch.float64) / d


def _volume(geom: AcquisitionGeometry, d: int, h: int) -> torch.Tensor:
    kz = torch.arange(d, dtype=torch.long)[:, None]
    ky = torch.arange(h, dtype=torch.long)[None, :]
    if geom.line_order in (LineOrder.LINEAR, LineOrder.KY_INNER):
        return kz * h + ky
    if geom.line_order is LineOrder.KZ_INNER:
        return ky * d + kz
    # Exact integer key proportional to the normalised radius squared: (dz/n_kz)^2 + (dy/n_ky)^2.
    dz = kz - d // 2
    dy = ky - h // 2
    radius = ((dz * h) ** 2 + (dy * d) ** 2).flatten()
    rank = torch.empty(d * h, dtype=torch.long)
    rank[torch.argsort(radius, stable=True)] = torch.arange(d * h)
    return rank.view(d, h)


# Builders return line times in units of TR.
_SLOTS: dict[AcquisitionMode, Callable[[AcquisitionGeometry, int, int], torch.Tensor]] = {
    AcquisitionMode.SLICE_2D: _slice_2d,
    AcquisitionMode.MULTISLICE_2D: _multislice,
    AcquisitionMode.VOLUME_3D: _volume,
}


def line_times(
    geometry: AcquisitionGeometry, n_slices_or_kz: int, n_ky: int, *, tr_s: float
) -> torch.Tensor:
    """Acquisition time in seconds of each ``(slice or kz, ky)`` line, shape ``[D, H]``.

    ``SLICE_2D``: ``ky*TR``. ``MULTISLICE_2D`` sequential: ``(d*n_ky+ky)*TR``; interleaved:
    ``ky*TR + rank(d)*TR/D`` with rank over the order 0, 2, 4, ..., 1, 3, 5, ....
    ``VOLUME_3D``: linear/ky_inner ``(kz*n_ky+ky)*TR``, kz_inner ``(ky*n_kz+kz)*TR``, centric by
    increasing normalised (kz, ky) radius (ties by (kz, ky)). ``line_order`` applies only to 3-D.
    """
    if not tr_s > 0:
        raise ValueError(f"tr_s must be > 0, got {tr_s}")
    if n_slices_or_kz < 1 or n_ky < 1:
        raise ValueError(f"sizes must be >= 1, got ({n_slices_or_kz}, {n_ky})")
    mode = geometry.mode
    if mode not in _SLOTS:
        raise ValueError(f"unsupported acquisition mode {mode!r}")
    return (_SLOTS[mode](geometry, n_slices_or_kz, n_ky).to(torch.float64) * tr_s).to(torch.float32)


def shot_index(times: torch.Tensor, n_shots: int) -> torch.Tensor:
    """Group lines into ``n_shots`` consecutive-in-time groups of (near-)equal count, ``[D, H]``."""
    if times.dim() != 2:
        raise ValueError(f"times must be [D, H], got shape {tuple(times.shape)}")
    n = times.numel()
    if not 1 <= n_shots <= n:
        raise ValueError(f"n_shots must be in [1, {n}], got {n_shots}")
    rank = torch.empty(n, dtype=torch.long, device=times.device)
    rank[torch.argsort(times.flatten(), stable=True)] = torch.arange(n, device=times.device)
    return (rank * n_shots // n).view_as(times)
