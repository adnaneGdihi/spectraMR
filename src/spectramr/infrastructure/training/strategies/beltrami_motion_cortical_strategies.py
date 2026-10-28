"""Beltrami motion correction + conformal cortical strategies — §§4 and 5.

* :class:`BeltramiMotionCorrectionStrategy`
  (``beltrami_motion_correction``) — motion correction parameterised
  by a Beltrami coefficient ``μ`` recovered through the
  :class:`LinearBeltramiSolver`. Guarantees the resulting motion
  field is a ``K``-quasi-conformal homeomorphism by construction.
* :class:`CorticalConformalReconStrategy`
  (``cortical_conformal_recon``) — reconstruction in conformally-
  flattened cortical coordinates. The flattening operator
  :math:`\\Phi_{\\text{cort}}` is supplied to the strategy as a
  pre-computed sampling grid per batch; the strategy itself only
  routes the loss through the in-disk reconstructor and adds the
  conformal-curvature consistency term.

References (numbers match the plan):
    [1]  Ahlfors, *Lectures on Quasiconformal Mappings*, AMS, 2006.
    [8]  Lui et al., "Teichmüller mapping", *SIAM J. Imag. Sci.*, 2014.
    [10] Gu et al., "Genus zero surface conformal mapping...",
         *IEEE Trans. Med. Imag.*, 23(8), 2004.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.physics.conformal_geometry import (
    LinearBeltramiSolver,
    enforce_beltrami_constraint,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


def _warp_with_phi(image: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Warp ``image[B, C, H, W]`` by a complex displacement ``phi[B, H, W]``.

    Uses ``grid_sample`` with normalised coordinates from the real and
    imaginary parts of ``phi``.
    """
    B, C, H, W = image.shape
    # phi values are in canonical [-1, 1] (LinearBeltramiSolver convention).
    grid = torch.stack([phi.real, phi.imag], dim=-1)
    grid = grid.clamp(-1.0, 1.0)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


class BeltramiMotionCorrectionStrategy(BaseTrainingStrategy):
    """Motion correction with a Beltrami-parameterised displacement.

    A small head predicts an unconstrained complex tensor
    :math:`\\tilde\\mu` per sample, projected to the Beltrami ball by
    :func:`enforce_beltrami_constraint`. The Linear Beltrami Solver
    recovers :math:`\\varphi(z)`, which warps the moving image to a
    fixed reference for the data-consistency residual.
    """

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    # parallel_compat: mu_head is an owned module, DDP-wrapped and run via call_owned.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: object) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("mu_head", OWNED_MODULE),)

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "beltrami_motion_correction", None)
        in_channels = int(getattr(cfg, "in_channels", 1)) if cfg else 1
        grid_size = int(getattr(cfg, "grid_size", 32)) if cfg else 32
        self.k_max = float(getattr(cfg, "k_max", 0.9)) if cfg else 0.9
        self.lambda_smooth = float(getattr(cfg, "lambda_smooth", 1e-3)) if cfg else 1e-3
        self.solver = LinearBeltramiSolver(max_iter=15, tol=1e-3)
        self.grid_size = grid_size
        self.mu_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        if self.device is not None:
            self.mu_head = self.mu_head.to(self.device)
        # The mu-head is the strategy's only learnable component: register it for
        # DDP, then put its parameters on opt_g so it actually trains.
        self.register_owned_module("mu_head", self.mu_head)
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new_params = [p for p in self.mu_head.parameters() if p not in existing]
            if new_params:
                opt.add_param_group({"params": new_params})

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        gs = self.grid_size
        moving = F.interpolate(input_batch, size=(gs, gs), mode="bilinear", align_corners=False)
        fixed = F.interpolate(target_batch, size=(gs, gs), mode="bilinear", align_corners=False)
        raw = self.call_owned("mu_head", moving)
        mu = enforce_beltrami_constraint(torch.complex(raw[:, 0], raw[:, 1]), k_max=self.k_max)
        phi = self.solver(mu).phi
        warped = _warp_with_phi(moving, phi)
        residual = F.l1_loss(warped, fixed)
        smooth = (mu.diff(dim=-1).abs().pow(2).mean() + mu.diff(dim=-2).abs().pow(2).mean()) * 0.5
        total = residual + self.lambda_smooth * smooth
        return {
            "g_total_loss": total,
            "beltrami_motion_residual": residual.detach(),
            "beltrami_smoothness": smooth.detach(),
        }


class CorticalConformalReconStrategy(BaseTrainingStrategy):
    """Reconstruction in conformally-flattened cortical coordinates.

    The conformal flattening operator :math:`\\Phi_{\\text{cort}}`
    and its inverse are precomputed offline per subject (cf. the
    Gu et al. uniformisation pipeline) and shipped to this strategy
    as ``batch['cortex_flatten_grid']``. When absent the strategy
    runs on whatever sampling the dataloader provides, exercising
    only the conformal-curvature consistency loss as an additional
    regulariser.
    """

    # parallel_compat: Builds no parameters (the old note described the Beltrami mu_head); the
    # generator runs only through its wrapped forward.
    parallel_compat = frozenset({"replicated"})

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "cortical_conformal_recon", None)
        self.lambda_curv = float(getattr(cfg, "lambda_curvature", 0.1)) if cfg else 0.1

    @staticmethod
    def _disk_laplacian(x: torch.Tensor) -> torch.Tensor:
        # 5-point Laplacian on the canonical disk grid.
        kernel = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernel = kernel.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        batch = kwargs.get("batch", {})
        grid = batch.get("cortex_flatten_grid")
        if isinstance(grid, torch.Tensor):
            grid = grid.to(input_batch.device)
            x_disk = F.grid_sample(
                input_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            y_disk = F.grid_sample(
                target_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        else:
            x_disk, y_disk = input_batch, target_batch

        recon = self.generator_model(x_disk)
        recon_loss = F.l1_loss(recon, y_disk)
        u = batch.get("conformal_factor")
        if isinstance(u, torch.Tensor):
            u = u.to(input_batch.device)
            curv_residual = (
                (self._disk_laplacian(recon) - torch.exp(-2.0 * u) * self._disk_laplacian(y_disk))
                .pow(2)
                .mean()
            )
        else:
            # Without u we still penalise differential disagreement of
            # the Laplacian — a weaker but still useful prior.
            curv_residual = (
                (self._disk_laplacian(recon) - self._disk_laplacian(y_disk)).pow(2).mean()
            )
        total = recon_loss + self.lambda_curv * curv_residual
        return {
            "g_total_loss": total,
            "cortical_recon": recon_loss.detach(),
            "cortical_conformal_curvature": curv_residual.detach(),
        }


__all__ = [
    "BeltramiMotionCorrectionStrategy",
    "CorticalConformalReconStrategy",
]
