"""Synthesize a pose set with commutator residuals from a planted hardware warp.

One owner for the fixture the hardware-warp tests share: a warp with known canonical
coefficients, six random rigid motions, and the first-order residuals ``ad_xi v`` sampled on a
cloud that is a strided subset of the canonical ball grid (so the second basis and the
projection back are exercised). ``write_planted_pose_set`` lays the same record out on disk in
the pose-set contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
)

__all__ = ["synthesize_planted_pose_set", "write_planted_pose_set"]


def synthesize_planted_pose_set(
    *,
    n_pairs: int = 6,
    seed: int = 7,
    max_degree: int = 3,
    min_degree: int = 2,
    ball_radius_mm: float = 100.0,
    basis_points_per_axis: int = 11,
    unit_id: str = "swoop-01",
    software_version: str = "1.2.3",
    stride: int = 2,
    n_subjects: int = 1,
) -> dict[str, Any]:
    """Return ``{"canonical", "truth", "residuals", "pose_set"}`` for a planted warp.

    Pair ``k`` belongs to subject ``sub-{k % n_subjects + 1:02d}``, so ``n_subjects > 1`` gives
    a hold-one-subject-out fold something to hold out; the default keeps every pair on
    ``sub-01`` as the older fixtures expect.
    """
    canonical = HarmonicGradientBasis(
        ball_points(basis_points_per_axis, ball_radius_mm), max_degree, min_degree
    )
    gen = torch.Generator().manual_seed(seed)
    truth = torch.randn(canonical.n_basis, dtype=torch.float64, generator=gen)
    axes = torch.randn(n_pairs, 3, dtype=torch.float64, generator=gen)
    omegas = 0.05 * axes / axes.norm(dim=1, keepdim=True)
    translations = 0.5 * torch.randn(n_pairs, 3, dtype=torch.float64, generator=gen)
    subset = torch.arange(0, canonical.points.shape[0], stride)
    residuals = []
    for omega, t in zip(omegas, translations, strict=True):
        field = torch.einsum("b,bni->ni", truth, canonical.adjoint_action(omega, t))
        residuals.append((canonical.points[subset], field[subset]))
    pose_set = {
        "unit_id": unit_id,
        "software_version": software_version,
        "ball_radius_mm": float(ball_radius_mm),
        "center_mm": [0.0, 0.0, 0.0],
        "pairs": [
            {
                "subject_id": f"sub-{k % n_subjects + 1:02d}",
                "sessions": ["ses-01", f"ses-{k // n_subjects + 2:02d}"],
                "omega": omega.tolist(),
                "translation": t.tolist(),
                "residual_path": f"pair_{k}.pt",
            }
            for k, (omega, t) in enumerate(zip(omegas, translations, strict=True))
        ],
    }
    return {"canonical": canonical, "truth": truth, "residuals": residuals, "pose_set": pose_set}


def write_planted_pose_set(directory: Path, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    """Write the planted record under ``directory``; return ``(pose_set_path, planted)``."""
    planted = synthesize_planted_pose_set(**kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    for pair, (points, residual) in zip(
        planted["pose_set"]["pairs"], planted["residuals"], strict=True
    ):
        torch.save({"points": points, "residual": residual}, directory / pair["residual_path"])
    path = directory / "pose_set.json"
    path.write_text(json.dumps(planted["pose_set"]))
    return path, planted
