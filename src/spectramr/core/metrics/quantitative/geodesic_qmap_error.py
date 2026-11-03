"""Geodesic-qMap error metric (Idea 5).

The geometrically correct analogue of Euclidean MAE on (M0, T1, T2)
maps. Computes the Fisher-metric geodesic distance between predicted
and ground-truth parameter triples at every voxel, then reduces to a
mean.
"""

from __future__ import annotations

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold


@register_metric(
    "geodesic_qmap_error",
    aliases=["GeodesicQMapMAE"],
    workflows=frozenset({Regime.QUANTITATIVE}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class GeodesicQMapErrorMetric:
    """Mean geodesic distance on the Bloch relaxation manifold.

    Tagged ``mri_quantitative``: the channel axis is read as ``(M0, T1, T2)`` and
    the distance is taken on the relaxation manifold, so this grades a
    quantitative parameter map and nothing else.
    """

    #: Uncached ``metric_tensor`` loops in Python per point; above this a call hangs.
    UNCACHED_POINT_LIMIT: int = 4096

    def __init__(self, manifold: BlochRelaxationManifold | None = None) -> None:
        # A caller holding a cached manifold (the Riemannian Bloch strategy) passes it,
        # so the metric grades in the same Fisher geometry the model trained in.
        self.manifold = manifold if manifold is not None else BlochRelaxationManifold()
        self._name = "geodesic_qmap_error"

    @property
    def name(self) -> str:
        return self._name

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        foreground: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Mean distance over all voxels, or over ``foreground`` ([B, H, W] bool) if given."""
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(prediction.shape)} vs target {tuple(target.shape)}"
            )
        # Expected layout [B, 3, H, W] — channels (M0, T1, T2).
        p = prediction.permute(0, 2, 3, 1).reshape(-1, 3)
        q = target.permute(0, 2, 3, 1).reshape(-1, 3)
        if foreground is not None:
            if foreground.shape != (target.shape[0], *target.shape[2:]):
                raise ValueError(
                    f"foreground mask {tuple(foreground.shape)} does not match map "
                    f"{tuple(target.shape)} without its channel axis."
                )
            keep = foreground.reshape(-1)
            p, q = p[keep], q[keep]
            if p.shape[0] == 0:
                raise ValueError("geodesic_qmap_error: the foreground mask selects no voxel.")
        if self.manifold.cache_resolution <= 0 and p.shape[0] > self.UNCACHED_POINT_LIMIT:
            raise ValueError(
                f"geodesic_qmap_error on {p.shape[0]} voxels with an uncached Fisher metric "
                f"would loop in Python per voxel. Pass a manifold built with "
                f"cache_resolution > 0, or score at most {self.UNCACHED_POINT_LIMIT} voxels."
            )
        d = self.manifold.geodesic_distance(p, q)
        return float(d.mean().item())


__all__ = ["GeodesicQMapErrorMetric"]
