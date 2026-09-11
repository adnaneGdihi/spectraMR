"""Motion Meta-Training Strategy.

Stage 1 of the Virtual Fiducial + HyperMamba motion correction pipeline.
Trains the HyperMambaUNet on clean M4Raw data with synthetic motion corruption.

Training Loop:
    1. Sample random motion trajectory θ
    2. Corrupt clean anatomy + Virtual Fiducial with KinematicForwardOperator
    3. Feed corrupted fiducial → HyperMambaBridge → SSM params
    4. Feed corrupted anatomy → HyperMambaUNet → reconstructed image
    5. Compute composite loss: L1 + λ₁·SSIM + λ₂·HFEN

Reference:
    Batchelor et al., "Matrix description of general motion correction," MRM 2005.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.kinematic_forward import KinematicForwardOperator
from spectramr.infrastructure.physics.virtual_fiducial import VirtualFiducial
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.loss_folding import declared_loss_weights
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


class ConcreteMotionMetaTrainingStrategy(BaseTrainingStrategy):
    """Strategy for meta-training the HyperMambaUNet on synthetic motion.

    Uses clean M4Raw averaged volumes as ground truth. Random rigid-body
    motion is generated per batch and applied via the KinematicForwardOperator
    to both the clean anatomy and the Virtual Fiducial simultaneously.

    Attributes:
        kinematic_op: Differentiable motion forward operator.
        fiducial: Virtual Fiducial (Gaussian grid probe).
        loss_l1: L1 reconstruction loss.
        loss_ssim: Structural similarity loss.
        loss_hfen: High-frequency error norm loss.
    """

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MotionMetaTrainingStrategy.

        Args:
            env: TrainingEnvironment from the builder.
            device: Target device.
            **kwargs: Additional kwargs.
        """
        super().__init__(env=env, device=device, **kwargs)
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize physics operators and loss functions."""
        # Physics operators
        _ps = self.config.data.sampling.patch_size
        _is = getattr(self.config.data, "image_size", None)
        if _ps and len(_ps) >= 2:
            img_size = (int(_ps[0]), int(_ps[1]))
        elif _is and len(_is) >= 2:
            img_size = (int(_is[0]), int(_is[1]))
        else:
            img_size = (256, 256)

        self.kinematic_op = KinematicForwardOperator(
            im_size=img_size,
        ).to(self.device)

        self.fiducial = VirtualFiducial(
            im_size=img_size,
            grid_spacing=16,
            sigma=2.0,
            learnable=False,
        ).to(self.device)

        # Loss functions (all registered in the loss registry)
        self.loss_l1 = create_loss("l1").to(self.device)
        self.loss_ssim = create_loss("ssim").to(self.device)
        self.loss_hfen = create_loss("hfen").to(self.device)

        # Loss weights from the loss-weight SSOT — NOT from ``config.losses.
        # reconstruction`` directly. That read (whose comment used to call itself
        # "SSOT") saw only the *category* paradigm: an arm on the domain paradigm
        # (``losses.image_losses: [{name: l1, weight: ...}]``, no ``reconstruction:``
        # block) still gets a fully-defaulted ``ReconstructionLossesConfig`` back, so
        # the strategy silently applied ``lambda_l1``'s SCHEMA DEFAULT of 10.0 and
        # ignored the weight the arm actually declared. The one live arm
        # (``experiment_vf_hyper_mamba_meta_v2``) escapes only by the coincidence that
        # its declared weight is also 10.0 — and its own in-repo comment ("the run
        # actually used 10.0 (lambda_l1 won). Materialized.") records the divergence
        # having already fired once and been papered over by editing the YAML.
        # ``declared_loss_weights`` reads BOTH surfaces and raises when they disagree;
        # ``.get(..., 0.0)`` keeps "undeclared means off" intact (an undeclared term
        # contributes ``0.0 * loss``, byte-identical to not being summed).
        declared = declared_loss_weights(self.config)
        self._lambda_l1 = declared.get("l1", 0.0)
        self._lambda_ssim = declared.get("ssim", 0.0)
        self._lambda_hfen = declared.get("hfen", 0.0)

        # Motion parameters from config or defaults
        if hasattr(self.config.training, "motion"):
            self._max_translation = self.config.training.motion.max_translation
            self._max_rotation = self.config.training.motion.max_rotation
            self._motion_type = getattr(self.config.training.motion, "motion_type", "smooth")
        else:
            self._max_translation = 5.0
            self._max_rotation = 0.05
            self._motion_type = "smooth"

        logger.info(
            "[MotionMetaTraining] Setup complete: im_size=%s, "
            "λ_l1=%.2f, λ_ssim=%.2f, λ_hfen=%.2f, trans=%.1f, rot=%.3f",
            img_size,
            self._lambda_l1,
            self._lambda_ssim,
            self._lambda_hfen,
            self._max_translation,
            self._max_rotation,
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute motion meta-training losses.

        Steps:
            1. target_batch is clean, fully-sampled image (ground truth)
            2. Generate random motion θ
            3. Corrupt target via kinematic operator → motion-corrupted input
            4. Corrupt fiducial via same θ → corrupted_fiducial
            5. Forward through HyperMambaUNet with corrupted_fiducial
            6. Compute composite loss vs clean target

        Args:
            input_batch: Input tensor (not used — we corrupt target instead).
            target_batch: Clean ground truth [B, C, H, W].
            epoch: Current epoch.
            **kwargs: Additional kwargs (batch dict, etc.).

        Returns:
            Loss dict with 'g_total_loss' and component losses.
        """
        B = target_batch.shape[0]

        # 1. Generate random motion trajectory
        theta = self.kinematic_op.generate_random_motion(
            batch_size=B,
            max_translation=self._max_translation,
            max_rotation=self._max_rotation,
            motion_type=self._motion_type,
            device=self.device,
        )

        # 2. Convert target to complex for kinematic operator
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        # Handle real-stacked [B, 2*C, H, W] → complex [B, C, H, W]
        if not torch.is_complex(target_batch):
            C = target_batch.shape[1]
            if C % 2 == 0:
                real = target_batch[:, 0::2, :, :]
                imag = target_batch[:, 1::2, :, :]
                target_complex = torch.complex(real, imag)
            else:
                target_complex = target_batch.to(torch.complex64)
        else:
            target_complex = target_batch

        # 2b. The data is k-space (``dataset_type: kspace``). The kinematic
        # operator, the loss and the cached visuals are all image-domain, so the
        # target must be IFFT'd to image domain ONCE here. The earlier code
        # asserted "target_complex is already an image" and dropped the IFFT —
        # false for k-space data: the REAL reference then rendered as raw
        # |k-space| and the FAKE collapsed to black (smoke audit 2026-06-13).
        # Idempotent for image-domain data (SSOT seam on BaseTrainingStrategy).
        target_complex = self._ensure_image_domain_target(target_complex)

        # 3. Corrupt anatomy with kinematic operator (expects an image)
        corrupted_kspace = self.kinematic_op(target_complex, theta)

        # 4. Corrupt Virtual Fiducial with same motion
        fiducial_image = self.fiducial(batch_size=B)
        corrupted_fiducial_kspace = self.kinematic_op(fiducial_image.to(self.device), theta)
        # Convert to real-stacked for bridge input [B, 2, H, W]
        corrupted_fiducial_real = torch.stack(
            [corrupted_fiducial_kspace.real, corrupted_fiducial_kspace.imag],
            dim=1,
        ).squeeze(2)  # Remove original channel dim

        # 5. Convert corrupted k-space to real-stacked for UNet input
        corrupted_image = ifft2c(corrupted_kspace)
        corrupted_input = torch.cat([corrupted_image.real, corrupted_image.imag], dim=1)

        # 6. Forward through HyperMambaUNet
        gen = self.generator_model
        pred = gen(
            corrupted_input,
            corrupted_fiducial=corrupted_fiducial_real,
        )

        # 7. Prepare target for loss (real-stacked). target_complex was brought
        # into image domain at step 2b, so it is now a genuine image (k-space
        # data is IFFT'd; image data passes through). Use it directly.
        target_image = target_complex
        target_real = torch.cat([target_image.real, target_image.imag], dim=1)

        # 8. Compute composite loss
        loss_l1 = self.loss_l1(pred, target_real)
        loss_ssim = self.loss_ssim(pred, target_real)
        loss_hfen = self.loss_hfen(pred, target_real)

        g_total_loss = (
            self._lambda_l1 * loss_l1
            + self._lambda_ssim * loss_ssim
            + self._lambda_hfen * loss_hfen
        )

        return {
            "g_total_loss": g_total_loss,
            "loss_l1": loss_l1.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_hfen": loss_hfen.detach(),
        }

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run validation with fixed motion for reproducibility.

        Args:
            val_batch: Validation batch.
            batch_idx: Batch index.

        Returns:
            Validation metrics dict including val_psnr.
        """
        val_batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)
        if isinstance(val_batch, (list, tuple)):
            _, target = val_batch[0], val_batch[1]
        elif isinstance(val_batch, dict):
            target = val_batch.get("target", val_batch.get("hr"))
        elif hasattr(val_batch, "target"):
            target = val_batch.target
        else:
            target = val_batch

        target = self._to_device(target)

        # Handle 5D TorchIO volumes [B, C, H, W, D] → select middle slice
        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]

        with torch.no_grad():
            # #1190: calling ``_compute_losses_impl`` directly bypasses the
            # ``_compute_losses`` wrapper that emits the ``model_output``
            # snapshot, so arm it here. ``snapshot_source("val")`` is the OUTER
            # manager so the phase is still "val" when the emit runs on exit --
            # without it the record would claim the training data chain.
            with (
                self.snapshot_source("val"),
                self._capture_model_output(
                    module=self.generator_model,
                    input_batch=target,
                    target_batch=target,
                    step=batch_idx,
                ),
            ):
                losses = self._compute_losses_impl(
                    input_batch=target,
                    target_batch=target,
                    epoch=0,
                )

            # Compute image quality metric (PSNR) for early stopping
            # Lightweight forward pass mirroring _compute_losses_impl
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            if not torch.is_complex(target):
                C = target.shape[1]
                if C % 2 == 0:
                    target_complex = torch.complex(target[:, 0::2, :, :], target[:, 1::2, :, :])
                else:
                    target_complex = target.to(torch.complex64)
            else:
                target_complex = target

            # k-space → image (see _compute_losses_impl step 2b). Without this
            # the val PSNR was computed against |k-space| and the cached visual
            # target rendered as a k-space blob.
            target_complex = self._ensure_image_domain_target(target_complex)

            B = target.shape[0]
            theta = self.kinematic_op.generate_random_motion(
                batch_size=B,
                max_translation=self._max_translation,
                max_rotation=self._max_rotation,
                motion_type=self._motion_type,
                device=self.device,
            )
            corrupted_kspace = self.kinematic_op(target_complex, theta)
            corrupted_image = ifft2c(corrupted_kspace)
            corrupted_input = torch.cat([corrupted_image.real, corrupted_image.imag], dim=1)

            fiducial_image = self.fiducial(batch_size=B)
            corrupted_fiducial_kspace = self.kinematic_op(fiducial_image.to(self.device), theta)
            corrupted_fiducial_real = torch.stack(
                [corrupted_fiducial_kspace.real, corrupted_fiducial_kspace.imag],
                dim=1,
            ).squeeze(2)

            pred = self.generator_model(
                corrupted_input,
                corrupted_fiducial=corrupted_fiducial_real,
            )

            # target_complex is image-domain after the step-2b conversion above.
            target_image = target_complex
            target_real = torch.cat([target_image.real, target_image.imag], dim=1)

            mse = torch.nn.functional.mse_loss(pred, target_real)
            psnr = -10.0 * torch.log10(mse + 1e-10)

        result = {k: float(v) for k, v in losses.items() if isinstance(v, torch.Tensor)}
        result["val_psnr"] = float(psnr)

        # Emit YAML-configured ``validation.metrics`` so early-stopping /
        # best-metric monitors (val_robust_mri_psnr, val_hfen, …) resolve;
        # without this only val_psnr existed and monitors never fired
        # (dispatch 6944227, 2026-05-25). Metrics run on magnitude images.
        # compute() guards each metric; failures are warned, not swallowed
        # (CLAUDE.md #9/#10).
        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None:
            with torch.no_grad():
                # pred / target_real are block-encoded ``cat([real, imag],
                # dim=1)``, so the real and imaginary halves are the first /
                # second C channels — NOT interleaved 0::2 / 1::2 (which
                # silently mis-pairs channels for C>1; latent today only
                # because the reference arm uses C=1).
                cp = pred.shape[1] // 2
                ct = target_real.shape[1] // 2
                pred_mag = torch.complex(pred[:, :cp], pred[:, cp:]).abs()
                target_mag = torch.complex(target_real[:, :ct], target_real[:, ct:]).abs()
            try:
                result.update(
                    {k: float(v) for k, v in computer.compute(pred_mag, target_mag).items()}
                )
            except Exception as exc:
                logger.warning(
                    "[MotionMetaStrategy] validation_metrics_computer failed; "
                    "only val_psnr available this step: %s",
                    exc,
                )
        return result


__all__ = ["ConcreteMotionMetaTrainingStrategy"]
