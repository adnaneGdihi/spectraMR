"""Vessel maps and shared plumbing of the flow axes (flow_ghosts, flow_voids, inflow_enhancement).

A vessel map is a ``[0, 1]`` weight map broadcastable to the input. When the caller gives none,
1-3 seeded discs per sample (cylinders along D in a stack or volume) stand in for through-plane
vessels; the draws come from one ``per_sample_rand`` call so sample ``b`` never sees the batch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _generator
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = ["K_DIMS", "as_5d", "check_positive", "resolve_vessels", "sample_draws"]

_MAX_VESSELS = 3
# Per sample: [n_vessels] + (cy, cx, radius) per slot + [spare phase].
N_DRAWS = 1 + 3 * _MAX_VESSELS + 1
# Vessel centres sit in the central half of the FOV; radius is 4-8 % of the short side (>= 1.5 px).
_CENTRE_LO, _CENTRE_SPAN = 0.25, 0.5
_RADIUS_LO, _RADIUS_SPAN = 0.04, 0.04


#: Acquisition mode -> the k-space axes one excitation encodes.
K_DIMS = {
    AcquisitionMode.SLICE_2D: (-2, -1),
    AcquisitionMode.MULTISLICE_2D: (-2, -1),
    AcquisitionMode.VOLUME_3D: (-3, -2, -1),
}


def check_positive(name: str, **values: float) -> None:
    """Raise unless every named kwarg is a finite number > 0."""
    for key, v in values.items():
        if not (isinstance(v, int | float) and not isinstance(v, bool) and 0 < v < float("inf")):
            raise ValueError(f"{name}: {key} must be a finite number > 0; got {v!r}")


def as_5d(x: Tensor) -> Tensor:
    """``(B, C, H, W)`` -> ``(B, C, 1, H, W)``; a 5-D tensor passes through."""
    return x if x.dim() == 5 else x.unsqueeze(2)


def sample_draws(x: Tensor, seed: int) -> Tensor:
    """``[B, N_DRAWS]`` uniforms, one sample at a time: vessel layout then one spare phase."""
    return per_sample_rand(
        x.shape[0], N_DRAWS, generator=_generator(x.device, seed), device=x.device
    )


def _discs(u: Tensor, h: int, w: int, dtype: torch.dtype) -> Tensor:
    """``[B, 1, 1, H, W]`` union of the first ``n`` of three seeded discs per sample."""
    b = u.shape[0]
    n = (1 + (u[:, 0] * _MAX_VESSELS).floor().clamp(max=_MAX_VESSELS - 1)).view(b, 1)
    slot = u[:, 1 : 1 + 3 * _MAX_VESSELS].view(b, _MAX_VESSELS, 3)
    cy = (_CENTRE_LO + _CENTRE_SPAN * slot[..., 0]) * h
    cx = (_CENTRE_LO + _CENTRE_SPAN * slot[..., 1]) * w
    r = ((_RADIUS_LO + _RADIUS_SPAN * slot[..., 2]) * min(h, w)).clamp(min=1.5)
    yy = torch.arange(h, device=u.device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=u.device, dtype=dtype).view(1, 1, 1, w)
    inside = (yy - cy[..., None, None]) ** 2 + (xx - cx[..., None, None]) ** 2 <= (
        r[..., None, None] ** 2
    )
    active = torch.arange(_MAX_VESSELS, device=u.device).view(1, -1) < n
    return (inside & active[..., None, None]).any(1).to(dtype).view(b, 1, 1, h, w)


def resolve_vessels(
    x5: Tensor, vessel_map: Tensor | None, u: Tensor, name: str, *, ndim_in: int
) -> Tensor:
    """The vessel weight map ``[B or 1, 1 or C, D or 1, H, W]``: the caller's, else seeded discs.

    ``vessel_map`` must have the input's rank and broadcast to it; its values are taken as
    weights, clamped to ``[0, 1]`` on the device (a range check would need a host sync).
    """
    dtype = x5.real.dtype
    if vessel_map is None:
        return _discs(u, x5.shape[-2], x5.shape[-1], dtype)
    if not isinstance(vessel_map, Tensor) or vessel_map.dim() != ndim_in:
        raise ValueError(
            f"{name}: vessel_map must be a tensor of rank {ndim_in} like the input; "
            f"got {type(vessel_map).__name__} of rank "
            f"{vessel_map.dim() if isinstance(vessel_map, Tensor) else 'n/a'}"
        )
    v = as_5d(vessel_map).to(device=x5.device, dtype=dtype).clamp(0.0, 1.0)
    try:
        ok = torch.broadcast_shapes(tuple(v.shape), tuple(x5.shape)) == tuple(x5.shape)
    except RuntimeError:
        ok = False
    if not ok:
        raise ValueError(
            f"{name}: vessel_map of shape {tuple(vessel_map.shape)} does not broadcast to the "
            f"input {tuple(x5.shape) if ndim_in == 5 else tuple(x5.shape[:2] + x5.shape[3:])}"
        )
    return v
