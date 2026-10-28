"""Where a k-space bin is, and what is known around it, as input channels.

A cold-diffusion backbone is asked to fill the bins a rung has not revealed, but
convolutions are translation-equivariant and the masked input zeroes exactly the
bins whose values it must invent, so the network cannot tell an unknown bin from
a bin whose coefficient is small. These planes give every backbone the geometry
the reverse process is defined over:

========  ==============================================================
channel   meaning
========  ==============================================================
support   ``M_t``: the bins known at this rung (training: the rung mask;
          reverse sampling: the observed support plus every committed bin)
acquired  bins the scanner wrote at all (``measured_lines``); all ones when
          the acquisition has no line axis, matching ``measured_lines=None``
ky        phase-encode coordinate along W, ``0`` at ``W // 2``, ``-1`` at 0
kx        readout coordinate along H, ``0`` at ``H // 2``, ``-1`` at 0
radius    ``log(1 + r) / log(1 + r_max)`` with ``r`` in bins from the DC bin
pe_gap    phase-encode distance to the nearest known column, over ``W``
========  ==============================================================

The centre is the ``fft2c`` convention (DC at ``N // 2``), which ``KSpaceCrop``
preserves at every U-Net level. A separate never-acquirable flag is not emitted:
with ``acquired`` defined as the scanner-written support it is ``1 - acquired``
exactly, a linear function of a channel already present.
"""

from __future__ import annotations

import functools

import torch

#: Order of the planes :func:`kspace_geometry_channels` returns.
KSPACE_GEOMETRY_PLANES: tuple[str, ...] = ("support", "acquired", "ky", "kx", "radius", "pe_gap")

#: Real planes per sample; each becomes one complex channel ``(plane, 0)`` on an
#: interleaved backbone input.
KSPACE_GEOMETRY_CHANNELS: int = len(KSPACE_GEOMETRY_PLANES)


def resolve_geometry_planes(names: object) -> tuple[str, ...]:
    """The plane names a ``kspace_geometry_planes`` declaration selects, validated.

    ``None`` is every plane. An empty list is legal and builds no planes.

    Raises:
        ValueError: ``names`` is not a list of plane names, repeats one, or names
            a plane that does not exist.
    """
    if names is None:
        return KSPACE_GEOMETRY_PLANES
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise ValueError(
            f"kspace_geometry_planes must be a list drawn from {list(KSPACE_GEOMETRY_PLANES)}, "
            f"got {names!r}."
        )
    unknown = [n for n in names if n not in KSPACE_GEOMETRY_PLANES]
    if unknown or len(set(names)) != len(names):
        raise ValueError(
            f"kspace_geometry_planes {list(names)} must be distinct names from "
            f"{list(KSPACE_GEOMETRY_PLANES)}; unknown: {unknown}."
        )
    return tuple(names)


def mask_plane(mask: torch.Tensor, batch: int, shape: tuple[int, int]) -> torch.Tensor:
    """A ``[B, 1, H, W]`` float plane of ``mask``: known if any channel knows the bin.

    Raises:
        ValueError: ``mask`` is not 4-D, does not match ``shape``, or carries a
            batch that is neither 1 nor ``batch``.
    """
    if mask.dim() != 4 or tuple(mask.shape[-2:]) != tuple(shape):
        raise ValueError(
            f"geometry mask must be [B, C, {shape[0]}, {shape[1]}], got {tuple(mask.shape)}."
        )
    if mask.shape[0] not in (1, batch):
        raise ValueError(f"geometry mask batch {mask.shape[0]} does not match {batch}.")
    plane = (mask.real if torch.is_complex(mask) else mask) > 0
    plane = plane.any(dim=1, keepdim=True)
    return plane.expand(batch, -1, -1, -1).float()


@functools.lru_cache(maxsize=16)
def _coordinate_planes(h: int, w: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """``[1, 3, H, W]`` ky, kx and log-radius planes; static per grid, so cached."""
    cy, cx = h // 2, w // 2
    dy = torch.arange(h, dtype=torch.float64) - cy
    dx = torch.arange(w, dtype=torch.float64) - cx
    kx = (dy / max(cy, 1))[:, None].expand(h, w)
    ky = (dx / max(cx, 1))[None, :].expand(h, w)
    r = torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    radius = torch.log1p(r) / torch.log1p(r.max()).clamp_min(1e-12)
    planes = torch.stack([ky, kx, radius])[None]
    return planes.to(device=torch.device(device), dtype=dtype)


def nearest_known_column_distance(known: torch.Tensor) -> torch.Tensor:
    """``[B, W]`` distance from each column to the nearest known one, over ``W``.

    The nearest known column on each side comes from one running max (left) and
    one running min over the reversed axis (right), so no Python loop walks the
    columns. A known column is at distance 0; a sample with no known column is
    at 1 everywhere.

    Args:
        known: ``[B, W]`` bool, a column is known when any of its bins is.
    """
    w = known.shape[-1]
    idx = torch.arange(w, device=known.device, dtype=torch.float32).expand_as(known)
    far = float(2 * w)
    left = torch.where(known, idx, torch.full_like(idx, -far)).cummax(dim=-1).values
    right_src = torch.where(known, idx, torch.full_like(idx, far)).flip(-1)
    right = right_src.cummin(dim=-1).values.flip(-1)
    dist = torch.minimum(idx - left, right - idx)
    return dist.clamp(max=float(w)) / float(w)


def kspace_geometry_channels(
    support: torch.Tensor,
    acquired: torch.Tensor | None,
    shape: tuple[int, int],
    *,
    batch: int,
    dtype: torch.dtype = torch.float32,
    planes: tuple[str, ...] = KSPACE_GEOMETRY_PLANES,
) -> torch.Tensor:
    """The ``[B, len(planes), H, W]`` real geometry planes (all of them by default).

    Args:
        support: ``M_t``, ``[B or 1, C, H, W]``; a bin is known when any channel
            marks it.
        acquired: The scanner-written support in the same layout, or ``None``
            when the acquisition has no line axis (every bin acquirable).
        shape: ``(H, W)`` of the k-space grid, phase encode along W.
        batch: Batch size of the tensor the planes are concatenated onto.
        dtype: Real dtype of the returned planes.
        planes: Which planes to return, in order; see :func:`resolve_geometry_planes`.
    """
    m_t = mask_plane(support, batch, shape)
    m_acq = (
        torch.ones_like(m_t)
        if acquired is None
        else mask_plane(acquired.to(m_t.device), batch, shape)
    )
    coords = _coordinate_planes(int(shape[0]), int(shape[1]), str(m_t.device), torch.float32)
    known_cols = m_t[:, 0].amax(dim=-2) > 0
    gap = nearest_known_column_distance(known_cols)[:, None, None, :].expand_as(m_t)
    stacked = torch.cat([m_t, m_acq, coords.expand(batch, -1, -1, -1), gap], dim=1)
    index = [KSPACE_GEOMETRY_PLANES.index(name) for name in planes]
    return stacked[:, index].to(dtype)


def as_interleaved_complex(planes: torch.Tensor) -> torch.Tensor:
    """``[B, K, H, W]`` real planes as ``[B, 2K, H, W]`` ``(plane, 0)`` pairs.

    Backbone inputs here hold complex coefficients as interleaved ``(Re, Im)``
    channels and several backbones parse them in pairs, so a geometry plane
    joins as the real part of a complex channel with a zero imaginary part.
    """
    return torch.stack([planes, torch.zeros_like(planes)], dim=2).flatten(1, 2)


def model_consumes_acquired_plane(model: object) -> bool:
    """Whether ``model`` (unwrapped from DDP/EMA ``.module``) reads ``acquired=``.

    The same models build their support plane from ``mask=`` and raise without one.
    """
    inner = getattr(model, "module", model)
    return int(getattr(inner, "kspace_geometry_channels", 0) or 0) > 0


def measurement_support(measurement: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W]`` float plane of the bins a zero-filled measurement carries.

    The observation mask of a measurement handed over without one: a bin is
    observed when any channel is non-zero. The generator's ``sample`` and
    ``generate`` and every caller that holds only the measurement use this one
    rule, so the support a backbone is conditioned on cannot depend on the caller.
    """
    return (measurement.abs().sum(dim=1, keepdim=True) > 1e-6).to(torch.float32)


def support_forward_kwargs(model: object, measurement: torch.Tensor) -> dict[str, torch.Tensor]:
    """``{"mask": measurement_support(measurement)}`` for a geometry-plane model, else ``{}``.

    For callers that hold a measurement and no mask; a model without geometry
    planes receives nothing it did not receive before.
    """
    if not model_consumes_acquired_plane(model):
        return {}
    return {"mask": measurement_support(measurement)}


__all__ = [
    "KSPACE_GEOMETRY_CHANNELS",
    "KSPACE_GEOMETRY_PLANES",
    "as_interleaved_complex",
    "kspace_geometry_channels",
    "mask_plane",
    "measurement_support",
    "model_consumes_acquired_plane",
    "nearest_known_column_distance",
    "resolve_geometry_planes",
    "support_forward_kwargs",
]
