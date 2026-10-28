"""Shared fixtures of the flow tests: geometries, phantoms and the apply helper."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)

M = AcquisitionMode
FLOW = ("flow_ghosts", "flow_voids", "inflow_enhancement")
VOXEL = (5.0, 1.0, 1.0)
GEOMS = {
    M.SLICE_2D: None,
    M.MULTISLICE_2D: AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL),
    M.VOLUME_3D: AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL),
}
PAIRS = [(n, m) for n in FLOW for m in M if m in DEGRADATION_REGISTRY[n].supported_modes]


def _phantom(mode: M, batch: int = 1, seed: int = 3) -> torch.Tensor:
    shape = (batch, 1, 24, 24) if mode is M.SLICE_2D else (batch, 1, 6, 24, 24)
    g = torch.Generator().manual_seed(seed)
    amp = 0.5 + torch.rand(shape, generator=g)
    return torch.polar(amp, torch.rand(shape, generator=g) - 0.5)


def _apply(name, x, theta, mode, seed=1, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=GEOMS[mode], **kw)


def _rms(a, b) -> float:
    return float((a - b).abs().pow(2).mean().sqrt())
