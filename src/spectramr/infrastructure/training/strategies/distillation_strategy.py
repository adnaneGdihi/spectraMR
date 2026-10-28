"""Method C: Teacher-Student Latent Distillation Strategy.

The Teacher (Method A: CrossAttentionOracleUNet) uses markers to extract
a perfect degradation token Z_teacher. The blind Student model learns to
predict the same latent from anatomy alone — no markers needed at test time.

Loss:
    L_total = L_recon(student_pred, clean)
            + λ_distill · ||Z_student - Z_teacher.detach()||²
            + λ_marker · L_marker(student_pred, ideal_marker)

All simulator parameters come from ``config.physics.digital_twin`` (SSOT).

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network," 2015.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.checkpointing.restore import weights
from spectramr.infrastructure.distributed.owned_modules import sync_frozen_module
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.simulator_builder import (
    build_simulator_from_config,
)
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


def _build_teacher(config: Any) -> torch.nn.Module:
    from spectramr.models.generators.cross_attention_oracle import CrossAttentionOracleUNet

    return CrossAttentionOracleUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        **(config.model.model_kwargs or {}),
    )


class ConcreteDistillationStrategy(BaseTrainingStrategy):
    """Teacher-Student distillation for marker-free inference.

    Phase 1 in init: loads the frozen Teacher (Method A) from checkpoint.
    Phase 2 in training: Teacher generates Z_teacher from ΔM, Student
    generates Z_student from anatomy only. Latent MSE loss distills
    the physics knowledge.

    The Student is the generator_model from the environment.
    The Teacher is loaded separately from a checkpoint.

    Attributes:
        teacher: Frozen CrossAttentionOracleUNet (or loaded from checkpoint).
        simulator: Digital Twin for generating training pairs (config-driven).
    """

    # parallel_compat: The teacher is frozen and runs under no_grad, so it needs no reduction,
    # only rank 0's values (sync_frozen_module). The student is the wrapped env generator.
    parallel_compat = frozenset({"replicated"})

    synthesises_model_input: ClassVar[bool] = True

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    def _setup_strategy_specific_components(self) -> None:
        """Initialize Teacher, simulator (from config), and losses."""
        # ── Build simulator from config.physics.digital_twin (SSOT) ──
        self.simulator = build_simulator_from_config(self.config, self.device)

        # ── Losses ──
        self.loss_l1 = create_loss("l1").to(self.device)
        self.loss_ssim = create_loss("ssim").to(self.device)
        self.loss_hfen = create_loss("hfen").to(self.device)
        self.loss_marker = create_loss("marker_corruption").to(self.device)

        recon = self.config.losses.reconstruction
        self._lambda_l1 = recon.lambda_l1
        self._lambda_ssim = recon.lambda_ssim
        self._lambda_hfen = recon.lambda_hfen
        self._lambda_marker = recon.lambda_marker  # SSOT: field added to schema
        self._lambda_distill = recon.lambda_distill  # SSOT: field added to schema

        # Teacher model: loaded from checkpoint
        self._teacher: torch.nn.Module | None = None
        self._load_teacher()

        logger.info(
            "[DistillationStrategy] λ_distill=%.2f, λ_marker=%.2f, teacher_loaded=%s",
            self._lambda_distill,
            self._lambda_marker,
            self._teacher is not None,
        )

    @staticmethod
    def _student_returns_latent(student: torch.nn.Module) -> bool:
        """Whether the student's forward takes ``return_latent``.

        Read on the inner module: a DDP/DP wrapper's forward takes only ``*inputs,
        **kwargs``, so probing it dropped the distillation term without an error.
        """
        inner = unwrap_model(student)
        return hasattr(inner, "forward") and "return_latent" in inner.forward.__code__.co_varnames

    def _load_teacher(self) -> None:
        """Load the frozen teacher from ``training.distillation.teacher_checkpoint``."""
        distillation = self.config.training.distillation
        teacher_path = distillation.teacher_checkpoint if distillation is not None else None
        if not teacher_path:
            # INFO, not WARNING: no-teacher is a supported mode and the distillation loss
            # is simply 0.0; a WARNING trips the smoke audit (F7b).
            logger.info(
                "[DistillationStrategy] No teacher checkpoint configured. Set "
                "training.distillation.teacher_checkpoint to the Method A checkpoint. "
                "Distillation loss will be skipped."
            )
            return
        try:
            teacher = _build_teacher(self.config)
            weights(teacher_path, "generator", teacher, policy="overlap")
            teacher.to(self.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            sync_frozen_module(teacher)  # "overlap" leaves unmatched tensors rank-seeded
            self._teacher = teacher
            logger.info("[DistillationStrategy] Teacher loaded from %s", teacher_path)
        except Exception as e:
            # A configured teacher that fails to load must not collapse the loss to 0.0
            # and train a plain reconstruction under the distillation label (pitfall #16).
            raise RuntimeError(
                f"[DistillationStrategy] training.distillation.teacher_checkpoint="
                f"'{teacher_path}' was configured but the teacher failed to load ({e}). "
                "Fix the path, or unset it to run without distillation."
            ) from e

    def _to_complex(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert real-stacked tensor to complex."""
        if torch.is_complex(batch):
            return batch
        C = batch.shape[1]
        if C >= 2 and C % 2 == 0:
            return torch.complex(batch[:, 0::2], batch[:, 1::2])
        return batch.to(torch.complex64)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute distillation + reconstruction + marker losses.

        Args:
            input_batch: Unused (corrupted internally).
            target_batch: Clean ground truth [B, C, H, W].
            epoch: Current epoch.
            **kwargs: Additional kwargs.

        Returns:
            Loss dict.
        """
        target_complex = self._to_complex(target_batch)
        # Bring a k-space-delivered target into image domain once. The Digital
        # Twin, the reconstruction loss and the cached visuals are image-domain;
        # without this an svd-coil arm's clean target / REAL reference is raw
        # |k-space| and the FAKE collapses to black (method_c, smoke audit
        # 2026-06-13). The decision is the SSOT needs_ifft_for_visualization, so
        # this is a NO-OP for the rss_image distillation arms (eval_c2/c3/c7,
        # exp_c4) whose dataset already emits a magnitude image.
        target_complex = self._ensure_image_domain_target(target_complex)

        # 1. Digital Twin simulation (expects an image)
        corrupted_image, marker_prior, _ = self.simulator(target_complex)

        # 2. Marker residual
        marker_mask = self.simulator.marker_mask.to(self.device)
        corrupted_marker = corrupted_image * marker_mask
        ideal_marker = marker_prior * marker_mask
        marker_residual = torch.cat(
            [
                corrupted_marker.real - ideal_marker.real,
                corrupted_marker.imag - ideal_marker.imag,
            ],
            dim=1,
        )

        # 3. Corrupted input (real-stacked, blind student)
        corrupted_input = torch.cat([corrupted_image.real, corrupted_image.imag], dim=1)

        # 4. Student forward (blind — no marker)
        student = self.generator_model
        if self._student_returns_latent(student):
            result = student(corrupted_input, return_latent=True)
            if isinstance(result, tuple):
                student_pred, z_student = result
            else:
                student_pred = result
                z_student = None
        else:
            student_pred = student(corrupted_input)
            z_student = None

        # 5. Teacher forward (with marker — frozen)
        z_teacher = None
        if self._teacher is not None:
            with torch.no_grad():
                _, z_teacher = self._teacher(
                    corrupted_input,
                    marker_residual=marker_residual,
                    return_latent=True,
                )

        # 6. Clean target (real-stacked)
        clean_target = torch.cat([target_complex.real, target_complex.imag], dim=1)

        # 7. Reconstruction loss
        loss_l1 = self.loss_l1(student_pred, clean_target)
        loss_ssim = self.loss_ssim(student_pred, clean_target)
        loss_hfen = self.loss_hfen(student_pred, clean_target)

        recon_loss = (
            self._lambda_l1 * loss_l1
            + self._lambda_ssim * loss_ssim
            + self._lambda_hfen * loss_hfen
        )

        # 8. Marker corruption loss
        loss_marker = self.loss_marker(
            student_pred,
            clean_target,
            marker_mask=marker_mask,
            ideal_marker=marker_prior,
        )

        # 9. Distillation loss (if both latents available)
        if z_student is not None and z_teacher is not None:
            loss_distill = F.mse_loss(z_student, z_teacher.detach())
        else:
            loss_distill = torch.tensor(0.0, device=self.device)

        g_total_loss = (
            recon_loss + self._lambda_distill * loss_distill + self._lambda_marker * loss_marker
        )

        return {
            "g_total_loss": g_total_loss,
            "loss_l1": loss_l1.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_hfen": loss_hfen.detach(),
            "loss_distill": loss_distill.detach(),
            "loss_marker": loss_marker.detach(),
        }

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate the blind student via a real forward pass (no markers).

        The previous implementation called ``_compute_losses_impl(target,
        target)`` — comparing the target to itself — so it returned only loss
        scalars and never an actual reconstruction metric. Early stopping
        monitors ``val_robust_mri_psnr``, which was therefore never emitted
        ("Early Stopping monitor metric ... not found", smoke audit 2026-06-03,
        F7a). This now mirrors the training forward (corrupt the clean target
        via the digital twin, run the student, blind to markers) and computes
        ``val_``-prefixed metrics via the SSOT validation-metrics computer
        (same seam as the VAE strategy), so early stopping resolves the monitor.
        """
        target_batch = self._to_device(target_batch)
        if target_batch.ndim == 5:  # volumetric -> mid-slice (as in training)
            mid = target_batch.shape[-1] // 2
            target_batch = target_batch[..., mid]

        with torch.no_grad():
            # Digital-twin corruption of the clean target (same as _compute_losses_impl)
            target_complex = self._to_complex(target_batch)
            # k-space → image once (see _compute_losses_impl) so the cached visual
            # target and val metrics are image-domain, not |k-space|.
            target_complex = self._ensure_image_domain_target(target_complex)
            corrupted_image, _marker_prior, _ = self.simulator(target_complex)
            corrupted_input = torch.cat([corrupted_image.real, corrupted_image.imag], dim=1)

            # Blind student forward (no marker), honouring the return_latent API
            student = self.generator_model
            if (
                hasattr(student, "forward")
                and "return_latent" in student.forward.__code__.co_varnames
            ):
                result = student(corrupted_input, return_latent=True)
                student_pred = result[0] if isinstance(result, tuple) else result
            else:
                student_pred = student(corrupted_input)

            # Real-stacked clean target (matches student_pred channel layout)
            clean_target = torch.cat([target_complex.real, target_complex.imag], dim=1)

            # SSOT validation metrics -> val_-prefixed (incl. val_robust_mri_psnr)
            computer = self._get_validation_metrics_computer(self.config)
            computed = computer.compute(student_pred, clean_target)

            # F2 (smoke 2026-06-16): cache image-domain magnitude visuals for the
            # pipeline's image logging (the train.py:2714 override seam ~10 VF
            # siblings already use). Without it, train.py falls back to
            # generator(raw 1-ch input) into a 2-ch conv → raises → swallowed →
            # "visual_samples was never captured" (eval_c2/c3/c7, exp_c4).
            self._last_visual_pred = self._to_complex(student_pred).abs().detach().cpu()
            self._last_visual_target = self._to_complex(clean_target).abs().detach().cpu()

        return {f"val_{k}": float(v) for k, v in computed.items()}


__all__ = ["ConcreteDistillationStrategy"]
