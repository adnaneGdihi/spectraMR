"""The default multi-coil Cartesian catalogue: a noise ladder and three mask families.

Lines are phase-encode rows (dim -2), fully sampled along the readout. All masks keep a
central calibration block, as accelerated scans do. Variable-density masks threshold one
seeded field at falling densities, so they are nested by construction. Equispaced and
partial-Fourier masks are generally not nested, which is where coil overlap decides
between incomparable and uncertified. Random masks draw each line independently (the
calibration block always), so they are ordered by coupling (Theorem C1.4). A uniform
Bernoulli mask at ``1/R`` matches equispaced ``R`` in expected density without either
coupling to the other.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .multicoil import (
    WITNESS_MARGIN,
    WITNESS_MAXITER,
    CoilExperiment,
    CoilRealiser,
    certify_coils,
    coil_maps,
)
from .relations import RelationCatalogue


def _rows(h: int, keep: torch.Tensor, w: int) -> torch.Tensor:
    return keep.double()[:, None].expand(h, w).clone()


def _prob_rows(p: torch.Tensor, w: int) -> torch.Tensor:
    return p.double()[:, None].expand(p.shape[0], w).clone()


def _calibration(h: int, frac: float) -> torch.Tensor:
    centre, half = h // 2, max(1, round(frac * h / 2))
    keep = torch.zeros(h, dtype=torch.bool)
    keep[centre - half : centre + half] = True
    return keep


def coil_catalogue(
    contents: Sequence[torch.Tensor],
    *,
    n_coils: int = 4,
    base_noise: float = 0.1,
    noise_levels: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
    accelerations: Sequence[int] = (2, 3, 4, 6),
    densities: Sequence[float] = (0.6, 0.45, 0.3),
    partial_fractions: Sequence[float] = (0.875, 0.75, 0.625),
    random_densities: Sequence[float] = (0.6, 0.3),
    random_accelerations: Sequence[int] = (2, 4),
    calibration_frac: float = 0.08,
    seed: int = 0,
) -> tuple[list[CoilExperiment], torch.Tensor, dict[str, object]]:
    """Experiments for ``contents`` (2-D, one shape), the shared coil maps, provenance."""
    if not contents or any(c.dim() != 2 for c in contents):
        raise ValueError("coil_catalogue needs 2-D (H, W) contents")
    h, w = int(contents[0].shape[-2]), int(contents[0].shape[-1])
    acs = _calibration(h, calibration_frac)
    rows = torch.arange(h)
    out = [
        CoilExperiment(f"noise_{s:g}", "noise", s, torch.ones(h, w, dtype=torch.float64), s)
        for s in noise_levels
    ]
    for r in accelerations:
        keep = ((rows - h // 2) % r == 0) | acs
        out.append(
            CoilExperiment(f"equispaced_R{r}", "equispaced", r, _rows(h, keep, w), base_noise)
        )
    field = torch.rand(h, generator=torch.Generator().manual_seed(seed))
    centrality = 1.0 - (rows - h // 2).abs().double() / (h / 2)
    for d in densities:
        keep = (field < d * (0.5 + centrality)) | acs
        out.append(
            CoilExperiment(f"vd_{d:g}", "variable_density", d, _rows(h, keep, w), base_noise)
        )
    for f in partial_fractions:
        keep = (rows < round(f * h)) | acs
        out.append(CoilExperiment(f"pf_{f:g}", "partial_fourier", f, _rows(h, keep, w), base_noise))
    for d in random_densities:
        p = torch.where(acs, 1.0, (d * (0.5 + centrality)).clamp(0.0, 1.0))
        out.append(
            CoilExperiment(f"random_vd_{d:g}", "random_vd", d, _prob_rows(p, w), base_noise, True)
        )
    for r in random_accelerations:
        p = torch.where(acs, 1.0, torch.full((h,), 1.0 / r, dtype=torch.float64))
        out.append(
            CoilExperiment(f"bernoulli_R{r}", "bernoulli", r, _prob_rows(p, w), base_noise, True)
        )
    provenance: dict[str, object] = {
        "catalogue": "multicoil_cartesian",
        "n_coils": n_coils,
        "coil_source": "synthetic shape-only maps, sum |S|^2 = 1, shared by every content",
        "noise_model": "white per coil and per acquired coefficient",
        "phase_encode_axis": -2,
        "calibration_frac": calibration_frac,
        "seed": seed,
        "witness": {"margin": WITNESS_MARGIN, "maxiter": WITNESS_MAXITER, "space": "real"},
        "sampled_fraction": {e.name: float(e.mask.mean()) for e in out},
        "random_masks": [e.name for e in out if e.random],
    }
    return out, coil_maps((h, w), n_coils, contents[0].device), provenance


def coil_relation_catalogue(
    contents: Sequence[torch.Tensor], **kwargs: object
) -> RelationCatalogue:
    """:func:`coil_catalogue` with its witnessed relations and SENSE-combine realiser."""
    exps, smaps, provenance = coil_catalogue(contents, **kwargs)  # type: ignore[arg-type]
    return RelationCatalogue(
        experiments=exps,
        graph=certify_coils(exps, smaps),
        realiser=lambda n_draws, seed: CoilRealiser(smaps, n_draws, seed),
        noise_dims=(int(smaps.shape[1]),),
        provenance=provenance,
    )


__all__ = ["coil_catalogue", "coil_relation_catalogue"]
