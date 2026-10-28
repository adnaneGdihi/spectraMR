"""Riemannian Bloch diffusion — Idea 5.

Forward (noising) SDE on the manifold M = {(M0, T1, T2)}, reverse SDE
using the learned tangent-space score. Training objective is the
Riemannian denoising score-matching loss with Varadhan's small-time
heat-kernel approximation for the conditional score:

    ∇_{θ_t} log p_{t|0}(θ_t | θ_0) ≈ −d_g(θ_t, θ_0)² / (4t).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)

#: Points per step above which an uncached Fisher metric stops being viable.
#: ``metric_tensor`` without a cache loops in Python over every point, so a
#: per-voxel reduction on a real patch is a hang, not a slow step.
_UNCACHED_POINT_LIMIT = 4096


class RiemannianBlochDiffusionStrategy(BaseTrainingStrategy):
    """Diffusion on the Bloch relaxation manifold.

    Debug-snapshot contract (CLAUDE.md non-negotiable 14): the model is fed
    ``theta_t``, a tangent-perturbed point on the manifold, not the prepared
    input, so the class declares it and emits the real input under
    ``diffusion_step`` (cohort review 2026-09-02, T0.8).
    """

    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        rbd = getattr(self.config.training, "riemannian_bloch_diffusion", None)
        self.t_min = float(getattr(rbd, "t_min", 1e-3)) if rbd else 1e-3
        self.t_max = float(getattr(rbd, "t_max", 1.0)) if rbd else 1.0
        self.step_size_init = float(getattr(rbd, "step_size_init", 0.05)) if rbd else 0.05
        self.injectivity_safety_factor = (
            float(getattr(rbd, "injectivity_safety_factor", 0.5)) if rbd else 0.5
        )
        cache_res = int(getattr(rbd, "metric_cache_resolution", 0)) if rbd else 0
        self.metric_refresh_interval = (
            int(getattr(rbd, "metric_refresh_interval", 50)) if rbd else 50
        )
        use_ckpt = bool(getattr(rbd, "enable_bloch_checkpoint", False)) if rbd else False
        self.manifold = BlochRelaxationManifold(
            cache_resolution=cache_res,
            use_bloch_checkpoint=use_ckpt,
        )
        self._last_refresh_epoch = -1
        self._logged_point_count = -1

    def _to_manifold_points(self, target_batch: torch.Tensor) -> torch.Tensor:
        """Reduce the target to ``[N, 3]`` points on M = {(M0, T1, T2)}.

        ``[B, 3]`` is already a stack of manifold points. A parameter-map stack
        ``[B, 3, *spatial]`` -- what ``dataset_type: quantitative`` serves --
        carries one manifold point per voxel, and the parameter axis is the
        CHANNEL axis.

        The previous reduction was ``flatten(start_dim=1)[..., :3]``, which on a
        map stack returns the first three voxels of the FIRST map and reads them
        as (M0, T1, T2). That is finite, plausible and wrong: the DSM residual
        falls smoothly while the model fits three T1 pixels. It never fired
        before because ``QuantitativeMapDataset`` emitted no ``target`` at all,
        so no batch ever reached this strategy.
        """
        dim = self.manifold.dim
        if target_batch.dim() < 2:
            raise ValueError(
                f"RiemannianBlochDiffusionStrategy expects a target of at least "
                f"2 dims; got shape {tuple(target_batch.shape)}."
            )
        if target_batch.dim() == 2:
            if target_batch.shape[-1] != dim:
                raise ValueError(
                    f"RiemannianBlochDiffusionStrategy expects [B, {dim}] "
                    f"manifold points (M0, T1, T2); got "
                    f"{tuple(target_batch.shape)}."
                )
            return target_batch
        if target_batch.shape[1] != dim:
            raise ValueError(
                f"RiemannianBlochDiffusionStrategy expects the parameter axis "
                f"on dim 1 with exactly {dim} channels (M0, T1, T2); got "
                f"{tuple(target_batch.shape)}. With dataset_type=quantitative "
                f"that means quantitative.target_maps must declare exactly "
                f"{dim} maps, in (M0/PD, T1, T2) order."
            )
        points = target_batch.movedim(1, -1).reshape(-1, dim)
        n_points = points.shape[0]
        if self.manifold.cache_resolution <= 0 and n_points > _UNCACHED_POINT_LIMIT:
            raise ValueError(
                f"{n_points} manifold points per step with the Fisher-metric "
                f"cache disabled. BlochRelaxationManifold.metric_tensor falls "
                f"back to a Python loop over every point, so this would not "
                f"finish. Set training.riemannian_bloch_diffusion."
                f"metric_cache_resolution (32 is the corpus value) to use the "
                f"interpolated grid, or reduce data.batch_size / patch_size."
            )
        if self._logged_point_count != n_points:
            logger.info(
                "[RIEMANNIAN-BLOCH] target %s -> %d manifold points/step "
                "(one per voxel, parameter axis = channels)",
                tuple(target_batch.shape),
                n_points,
            )
            self._log_coordinate_ranges(points)
            self._logged_point_count = n_points
        return points

    def _log_coordinate_ranges(self, points: torch.Tensor) -> None:
        """Report each coordinate's median against the manifold's own bounds.

        ``_interpolate_metric_cache`` normalises to the grid and then
        ``clamp``s, so a coordinate that is systematically outside
        ``cache_bounds`` -- a wrong channel order, or M0 in arbitrary rather
        than unit scale -- pins every point to the grid edge and returns a
        corner metric. That is silent and the loss stays plausible. We cannot
        decide from here whether a given corpus is mis-scaled or merely
        wide-ranging, so the measured fact is logged rather than guessed at.

        The ``.median()``/``.item()`` calls here are GPU syncs, so this runs
        once per distinct target shape (guarded by ``_logged_point_count``),
        not per step -- it is startup reporting, not steady-state work.
        """
        names = ("M0", "T1", "T2")
        for d, (lo, hi) in enumerate(self.manifold.cache_bounds):
            axis = points[:, d]
            outside = ((axis < lo) | (axis > hi)).float().mean().item()
            logger.info(
                "[RIEMANNIAN-BLOCH]   %s: median=%.4g range=[%.4g, %.4g] "
                "declared bounds=[%.4g, %.4g] outside=%.1f%%",
                names[d],
                axis.median().item(),
                axis.min().item(),
                axis.max().item(),
                lo,
                hi,
                100.0 * outside,
            )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Refresh the metric cache at the configured interval. The first
        # call always rebuilds so the cache is consistent with any
        # pulse-sequence parameters that may have changed since the
        # constructor.
        if self.metric_refresh_interval > 0 and (
            self._last_refresh_epoch < 0
            or epoch - self._last_refresh_epoch >= self.metric_refresh_interval
        ):
            self.manifold.refresh_metric_cache()
            self._last_refresh_epoch = epoch
        # The target carries the clean manifold points. ``input`` is unused:
        # this is unconditional score matching on p(M0, T1, T2), so the arm
        # legitimately declares ``quantitative.input_source: maps``.
        del input_batch
        theta0 = self._to_manifold_points(target_batch)
        b = theta0.shape[0]
        device = theta0.device
        t = torch.rand(b, device=device) * (self.t_max - self.t_min) + self.t_min
        # Riemannian forward step: sample a tangent perturbation in the
        # tangent space at θ_0, advance by exp_map.
        v = torch.randn_like(theta0) * t.unsqueeze(-1).sqrt()
        v = self.manifold.project_tangent(theta0, v)
        theta_t = self.manifold.exp_map(theta0, v, n_steps=4)
        # Heat-kernel approximation: ∇_θ_t log p_{t|0} ≈ -log_map(θ_t, θ_0) / t.
        target_tangent = -self.manifold.log_map(theta_t, theta0) / t.unsqueeze(-1)
        self._declare_model_input({"theta_t": theta_t, "t": t}, extra={"model_input_key": "theta_t"})
        predicted_tangent = self.generator_model(theta_t, t, None)
        # Inner product is in the Fisher metric; we use a per-point linearisation.
        G = self.manifold.metric_tensor(theta_t)  # [B, 3, 3]
        diff = (predicted_tangent - target_tangent).unsqueeze(-1)  # [B, 3, 1]
        quad = (diff.transpose(-1, -2) @ G @ diff).squeeze(-1).squeeze(-1)
        loss = quad.mean()
        return {
            "g_total_loss": loss,
            "rdsm_quadratic_residual": loss.detach(),
            "rdsm_mean_t": t.mean().detach(),
        }


__all__ = ["RiemannianBlochDiffusionStrategy"]
