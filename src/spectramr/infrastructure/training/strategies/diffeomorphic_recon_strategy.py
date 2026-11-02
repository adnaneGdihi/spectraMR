"""Diffeomorphic-Registration-Coupled Reconstruction (ULF-PR-8 / II.5).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-8.

Joint reconstruction + intra-volume diffeomorphic registration via an LDDMM-style
stationary velocity field ``v`` (Arsigny et al. 2006) and the exponential map
``φ = exp(v)`` computed by scaling-and-squaring (Ashburner 2007).

``v`` is emitted by a strategy-owned ``velocity_field_head`` block from the
reconstruction ``x̂`` and its target ``y``, so ``φ`` registers the reconstruction onto
the target and absorbs the residual non-rigid misalignment that a rigid
``nex_registration`` leaves between repetitions.

Loss = parent reconstruction fidelity
     + ``λ_smooth · ‖∇v‖²``              (smoothness)
     + ``λ_fold  · mean(max(-det Jφ, 0))``  (hinge penalty on folds of φ)
     + ``λ_warp  · ‖x̂ ∘ φ - y‖₁``           (warped-image fidelity).

``v`` and every displacement are in pixel units; they are converted to the normalised
``grid_sample`` frame only inside :meth:`exp_velocity`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.models.blocks import create_block

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

_DIFFEO_KEYS = frozenset(
    {
        "loss_velocity_smoothness",
        "loss_diffeomorphism_folding",
        "loss_warped_fidelity",
        "diffeo_fold_fraction",
    }
)


class DiffeomorphicReconStrategy(ReconstructionTrainingStrategy):
    """Recon + diffeomorphic registration via scaling-and-squaring."""

    # parallel_compat: velocity_head is an owned module, DDP-wrapped and run via call_owned;
    # it is built after the optimizer, so it cannot be sharded.
    parallel_compat = frozenset({"replicated"})

    n_squarings: int = 6
    lambda_smooth: float = 1e-3
    lambda_fold: float = 1e-2
    lambda_warp: float = 1.0
    head_hidden_channels: int = 16

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("velocity_head", OWNED_MODULE))

    def __init__(self, env: Any = None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        spatial_dims = int(getattr(self.config.model, "spatial_dims", 2) or 2)
        if spatial_dims != 2:
            raise ValueError(
                "diffeomorphic_recon integrates a 2-D stationary velocity field; "
                f"model.spatial_dims={spatial_dims} is not supported."
            )
        self._build_velocity_head(int(self.config.model.out_channels))
        logger.info(
            "diffeomorphic_recon: n_squarings=%d lambda_smooth=%g lambda_fold=%g "
            "lambda_warp=%g head_hidden_channels=%d",
            self.n_squarings,
            self.lambda_smooth,
            self.lambda_fold,
            self.lambda_warp,
            self.head_hidden_channels,
        )

    def declared_metric_keys(self) -> frozenset[str]:
        """The warp terms and fold fraction every step stamps; no config key names them."""
        return frozenset(super().declared_metric_keys()) | _DIFFEO_KEYS

    def _build_velocity_head(self, image_channels: int) -> None:
        """Build the head, register it for DDP and put its parameters on ``opt_g``."""
        head = create_block(
            "velocity_field_head",
            in_channels=2 * image_channels,
            hidden_channels=self.head_hidden_channels,
        )
        if self.device is not None:
            head = head.to(self.device)
        self.velocity_head = head
        self.register_owned_module("velocity_head", head)
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new_params = [p for p in head.parameters() if p not in existing]
            if new_params:
                opt.add_param_group({"params": new_params})

    @staticmethod
    def _identity_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        return torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    @staticmethod
    def _pixel_to_grid_scale(H: int, W: int, ref: torch.Tensor) -> torch.Tensor:
        """Pixel -> normalised-coordinate factor for (x, y) under ``align_corners=True``."""
        return ref.new_tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)])

    def exp_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Compute φ = exp(v) by scaling-and-squaring.

        v: (B, 2, H, W) velocity (x, y) in pixels. Returns the sampling grid
        (B, H, W, 2) of φ in normalised coordinates, for ``F.grid_sample``.
        """
        B, _, H, W = v.shape
        scale = self._pixel_to_grid_scale(H, W, v)
        grid_id = self._identity_grid(B, H, W, v.device).to(v.dtype)
        grid = grid_id + v.permute(0, 2, 3, 1) * scale / (2**self.n_squarings)
        for _ in range(self.n_squarings):
            # φ ← φ ∘ φ: the position field sampled at its own positions.
            grid = F.grid_sample(
                grid.permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)
        return grid

    def grid_displacement(self, grid: torch.Tensor) -> torch.Tensor:
        """Displacement of φ in pixels, (B, 2, H, W), from its sampling grid."""
        B, H, W, _ = grid.shape
        grid_id = self._identity_grid(B, H, W, grid.device).to(grid.dtype)
        return ((grid - grid_id) / self._pixel_to_grid_scale(H, W, grid)).permute(0, 3, 1, 2)

    @staticmethod
    def apply_warp(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    @staticmethod
    def _jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
        """det(I + ∇u) of a pixel-unit displacement ``flow`` (B, 2, H, W) -> (B, H-1, W-1)."""
        d_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        d_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        H, W = flow.shape[-2] - 1, flow.shape[-1] - 1
        d_dx = d_dx[:, :, :H, :W]
        d_dy = d_dy[:, :, :H, :W]
        return (1 + d_dx[:, 0]) * (1 + d_dy[:, 1]) - d_dx[:, 1] * d_dy[:, 0]

    @staticmethod
    def _smoothness(v: torch.Tensor) -> torch.Tensor:
        gx = v[:, :, :, 1:] - v[:, :, :, :-1]
        gy = v[:, :, 1:, :] - v[:, :, :-1, :]
        return gx.pow(2).mean() + gy.pow(2).mean()

    @staticmethod
    def _as_2d(x: torch.Tensor, name: str) -> torch.Tensor:
        """(B, C, H, W) view of a 2-D slice; a singleton depth axis is dropped explicitly."""
        if x.ndim == 5 and x.shape[2] == 1:
            return x.squeeze(2)
        if x.ndim != 4:
            raise ValueError(
                f"diffeomorphic_recon needs 2-D (B, C, H, W) {name}; got {tuple(x.shape)}."
            )
        return x

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        pred, target = self._last_prediction, self._last_target
        if pred is None or target is None:
            raise RuntimeError(
                "diffeomorphic_recon: the reconstruction seam (_last_prediction/_last_target) "
                "is empty after the parent step, so the warp terms cannot be computed."
            )
        pred = self._as_2d(pred, "prediction")
        target = self._as_2d(target, "target").to(pred.dtype)

        v = self.call_owned("velocity_head", torch.cat([pred, target], dim=1))
        grid = self.exp_velocity(v)
        warped = self.apply_warp(pred, grid)
        det = self._jacobian_determinant(self.grid_displacement(grid))

        smooth = self.lambda_smooth * self._smoothness(v)
        fold = self.lambda_fold * F.relu(-det).mean()
        warp_loss = self.lambda_warp * (warped - target).abs().mean()
        extra = smooth + fold + warp_loss

        if "g_total_loss" not in base:
            raise RuntimeError(
                f"diffeomorphic_recon: no g_total_loss in {sorted(base)} to add the warp terms to."
            )
        base["loss_velocity_smoothness"] = smooth
        base["loss_diffeomorphism_folding"] = fold
        base["loss_warped_fidelity"] = warp_loss
        base["diffeo_fold_fraction"] = (det <= 0).float().mean().detach()
        base["g_total_loss"] = base["g_total_loss"] + extra
        if "loss" in base:
            base["loss"] = base["g_total_loss"]
        return base

    def _first_steps_extra(self, epoch: int, batch: Any = None) -> dict[str, Any]:
        """Stamp the applied LDDMM constants into the snapshot provenance."""
        extra = super()._first_steps_extra(epoch, batch=batch)
        extra["lddmm"] = {
            "n_squarings": self.n_squarings,
            "lambda_smooth": self.lambda_smooth,
            "lambda_fold": self.lambda_fold,
            "lambda_warp": self.lambda_warp,
            "head_hidden_channels": self.head_hidden_channels,
            "velocity_source": "velocity_field_head(reconstruction, target)",
            "velocity_units": "pixels",
            "warp": "reconstruction -> target",
        }
        return extra
