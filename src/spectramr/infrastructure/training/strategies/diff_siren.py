from typing import ClassVar

import torch

from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer
from spectramr.models.losses.cross_contrast_losses import LNCCLoss


class DIFFSirenStrategy(BaseTrainingStrategy):
    """
    Distributed Implicit Fiducial Framework (DIFF) Curriculum Strategy.

    Implements a 2-stage curriculum for SIREN-VarNet architectures:
      - Epochs 1-10: Freeze VarNet. Train only the SIREN Field Interpolator
        using Boundary MSE + PINN Laplacian Loss.
      - Epochs 11+: Unfreeze VarNet. Train End-to-End, receiving gradients
        from image reconstruction (SSIM, HFEN, LNCC) through the Virtual-Coil DC
        blocks down into the SIREN coordinates.
    """

    forwards_loss_components: ClassVar[bool] = True

    def __init__(self, env=None, device=None, **kwargs):
        """Initialize DIFFSirenStrategy.

        Args:
            env: TrainingEnvironment containing config, models, optimizers.
            device: Optional device override.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(env=env, device=device, **kwargs)

        # Strategy specific flags
        self.curriculum_switch_epoch = getattr(self.config.training, "curriculum_switch_epoch", 10)

        # Safely access loss config attributes with fallbacks
        pinn_config = getattr(self.config, "losses", None)
        pinn_sub = getattr(pinn_config, "pinn", None) if pinn_config else None
        recon_sub = getattr(pinn_config, "reconstruction", None) if pinn_config else None

        self.lambda_bound = getattr(pinn_sub, "lambda_bound", 1.0) if pinn_sub else 1.0
        self.lambda_pde = getattr(pinn_sub, "lambda_laplacian", 1e-3) if pinn_sub else 1e-3
        self.lambda_img = getattr(recon_sub, "lambda_l1", 1.0) if recon_sub else 1.0

        # Models
        self.model = self.env.generator

        # SSOT: Use UnifiedReconstructionLossComputer for reconstruction losses
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )

        # Paradigm-specific losses (LNCC is cross-contrast, not standard reconstruction)
        self.lncc = LNCCLoss(window_size=9)

        # Determine current stage (handled in _on_epoch_start or manually)
        self.is_stage_2 = False

    def _on_epoch_start(self, epoch: int):
        """Handle Curriculum Switch at the epoch boundary."""
        if epoch >= self.curriculum_switch_epoch and not self.is_stage_2:
            self.logger.info(f"--- STAGE 2 INITIATED at Epoch {epoch} ---")
            self.logger.info("Unfreezing VarNet cascades. Enabling End-to-End Image Gradients.")
            self.is_stage_2 = True

            # Ensure VarNet is unfrozen
            for param in self.model.cascades.parameters():
                param.requires_grad = True

        elif epoch < self.curriculum_switch_epoch and self.is_stage_2:
            # Revert if needed
            self.is_stage_2 = False
            for param in self.model.cascades.parameters():
                param.requires_grad = False

    def _compute_laplacian_loss(self, siren_model):
        """PINN Laplacian Loss: enforces \\nabla^2 S = 0 on random internal coords."""
        rand_coords = (torch.rand(2000, 2, device=self.device) * 2 - 1).requires_grad_(True)
        S_pred = siren_model(rand_coords)  # [N, 2]

        grad_real = torch.autograd.grad(S_pred[:, 0].sum(), rand_coords, create_graph=True)[0]
        grad_imag = torch.autograd.grad(S_pred[:, 1].sum(), rand_coords, create_graph=True)[0]

        laplace_real_x = torch.autograd.grad(grad_real[:, 0].sum(), rand_coords, create_graph=True)[
            0
        ][:, 0]
        laplace_real_y = torch.autograd.grad(grad_real[:, 1].sum(), rand_coords, create_graph=True)[
            0
        ][:, 1]

        laplace_imag_x = torch.autograd.grad(grad_imag[:, 0].sum(), rand_coords, create_graph=True)[
            0
        ][:, 0]
        laplace_imag_y = torch.autograd.grad(grad_imag[:, 1].sum(), rand_coords, create_graph=True)[
            0
        ][:, 1]

        lap_sq = (laplace_real_x + laplace_real_y) ** 2 + (laplace_imag_x + laplace_imag_y) ** 2
        return torch.mean(lap_sq)

    def _compute_losses_impl(self, input_batch, target_batch, epoch, **kwargs):
        """Compute DIFF-SIREN curriculum losses.

        Reconstruction losses are delegated to UnifiedReconstructionLossComputer (SSOT).
        Paradigm-specific losses (LNCC, PDE Laplacian) remain strategy-managed.

        Args:
            input_batch: Input tensor (undersampled k-space or image).
            target_batch: Target tensor (fully-sampled image).
            epoch: Current epoch for curriculum switching.
            **kwargs: Additional arguments (batch dict, etc.).

        Returns:
            dict[str, torch.Tensor]: Loss dict with g_total_loss key.
        """
        losses = {}

        # Check curriculum stage
        if epoch >= self.curriculum_switch_epoch and not self.is_stage_2:
            self.is_stage_2 = True

        # Standard forward pass through model
        pred = self.model(input_batch)
        if isinstance(pred, tuple):
            pred = pred[0]

        # SSOT: Delegate reconstruction loss to loss_computer
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        # Thread the live loop iteration (the loop passes ``iteration``, never
        # ``step``, so ``kwargs.get("step", 0)`` froze any step-gated schedule
        # at 0 — pitfall #16).
        loss_output = self.loss_computer.compute(
            pred=pred,
            target=target_batch,
            epoch=epoch,
            iteration=resolve_loop_iteration(self),
            losses_dict=env_losses,
        )

        loss_recon = loss_output.total
        losses.update(loss_output.components)
        losses["loss_recon"] = loss_recon

        # Paradigm-specific: LNCC loss (cross-contrast, not standard
        # reconstruction). No except-swallow: the earlier bare `except Exception:
        # loss_lncc = 0` collapsed the arm's distinctive cross-contrast objective
        # to a plain reconstruction on any error while smoke still PASSed
        # (pitfalls #9/#16). Let a genuine failure surface.
        loss_lncc = self.lncc(pred, target_batch)
        losses["loss_lncc"] = loss_lncc * 0.1

        # PDE regularization if SIREN module exists (paradigm-specific physics).
        # Same rule: a failing SIREN Laplacian must raise, not vanish silently.
        if hasattr(self.model, "siren"):
            loss_pde = self._compute_laplacian_loss(self.model.siren)
            losses["loss_pde"] = loss_pde * self.lambda_pde

        # Total loss
        g_total = self.lambda_img * loss_recon + losses.get(
            "loss_lncc", torch.tensor(0.0, device=self.device)
        )
        if "loss_pde" in losses:
            g_total = g_total + losses["loss_pde"]

        losses["g_total_loss"] = g_total
        return losses
