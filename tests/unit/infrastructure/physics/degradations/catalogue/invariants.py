"""The invariants every catalogue axis must satisfy in every mode it supports.

Each catalogue test module calls :func:`assert_axis_invariants` once per (axis, geometry).
"""

from __future__ import annotations

from itertools import pairwise

import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    SliceOrder,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid

M = AcquisitionMode
VOXEL = (5.0, 1.0, 1.0)
GEOMETRIES: dict[str, AcquisitionGeometry | None] = {
    "slice_2d": None,
    "ms_interleaved": AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL),
    "ms_sequential": AcquisitionGeometry(
        mode=M.MULTISLICE_2D,
        voxel_mm=VOXEL,
        slice_order=SliceOrder.SEQUENTIAL,
    ),
    "volume": AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL),
    "volume_centric": AcquisitionGeometry(
        mode=M.VOLUME_3D, voxel_mm=VOXEL, line_order=LineOrder.CENTRIC
    ),
}
THETAS = (0.0, 0.3, 0.6, 1.0)


def phantom(geometry: AcquisitionGeometry | None, *, batch: int = 1, seed: int = 3) -> torch.Tensor:
    """Smooth, asymmetric, phase-bearing object: ``(B, 2, 32, 32)`` or ``(B, 2, 6, 32, 32)``."""
    d = 1 if geometry is None else 6
    z, y, x = mm_grid((d, 32, 32), None)
    g = torch.Generator().manual_seed(seed)
    amp = 1.0 + torch.rand((batch, 2, 1, 1, 1), generator=g)
    mag = torch.exp(-((x - 0.1) ** 2 + (1.3 * y) ** 2 + z**2) / 0.25) + 0.3 * (x > 0.3).float()
    out = torch.polar(amp * mag, (1.5 * x - y + z).expand(batch, 2, d, 32, 32))
    return out.squeeze(2) if geometry is None else out


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().pow(2).mean().sqrt())


def assert_axis_invariants(
    name: str, geometry: AcquisitionGeometry | None, *, seed: int = 1, **kwargs
) -> None:
    """theta=0 identity, monotone effect, finite, shape, deterministic, batch-invariant."""

    def run(x: torch.Tensor, theta: float) -> torch.Tensor:
        return apply_degradation(name, x, theta, seed=seed, geometry=geometry, **kwargs)

    x = phantom(geometry)
    assert torch.equal(run(x, 0.0), x), f"{name}: theta=0 is not the identity"
    outs = [run(x, t) for t in THETAS]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y)).all()
    res = [rms(x, y) for y in outs]
    assert all(a < b for a, b in pairwise(res)), f"{name}: not monotone in theta: {res}"
    assert torch.equal(run(x, 0.6), outs[2]), f"{name}: not deterministic"
    # Sample 0 alone equals sample 0 beside a neighbour ten times brighter.
    pair = torch.cat([x, 10.0 * phantom(geometry, seed=11)])
    torch.testing.assert_close(run(pair, 0.6)[:1], outs[2], rtol=0, atol=1e-5)
