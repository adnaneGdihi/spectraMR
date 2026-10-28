"""Guided Super-Resolution Training Strategy.

Reference-guided ULF-to-HF volumetric enhancement strategy. The model receives:
- Full ULF volume (low-field, noisy, low-contrast) → anatomical content
- Partial HF reference (a single slice or small slab from a real HF scan of
  the SAME subject) → quality/contrast exemplar

The output is a full HF-quality volume preserving the contrast of the ULF input.

During training, the partial HF reference is randomly sampled from the paired
ground truth. During inference, the actual partial HF acquisition is provided.

Key insight: The existing disentangled architecture's content/style separation
maps naturally to this problem:
- Content encoder → extracts anatomical structure from full ULF (field-invariant)
- Style encoder → extracts quality/contrast features from partial HF (field-specific)
- Generator → combines both to produce HF-quality output with ULF anatomy

Reference:
    Li et al., "Reference-Based Image Super-Resolution with Deformable
    Attention Transformer," ECCV, 2022.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.adversarial import AdversarialMixin
from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
    ReconstructionMixin,
)
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer
from spectramr.models.losses.hfen_loss import HFENLoss
from spectramr.models.losses.registry import create_loss
from spectramr.models.losses.ssim_loss import SSIMLoss

logger = logging.getLogger(__name__)


class GuidedSuperResolutionStrategy(BaseTrainingStrategy, ReconstructionMixin, AdversarialMixin):
    """Reference-guided ULF-to-HF super-resolution training strategy.

    Training loop:
        1. Sample paired (ULF_full, HF_full) from dataset
        2. Randomly extract K contiguous slices from HF_full as reference
        3. Content = ContentEncoder(ULF_full)
        4. Style = StyleEncoder(HF_reference)
        5. Output = Generator(Content, Style)
        6. Loss = Reconstruction(Output, HF_full) + Adversarial + Reference consistency

    Inference:
        Content from ULF input + Style from actual partial HF scan → full HF volume
    """

    def _setup_strategy_specific_components(self) -> None:
        """Initialize guided SR-specific components."""
        super()._setup_strategy_specific_components()
        model_kwargs = self.config.model.model_kwargs or {}

        # Reference extraction config
        self._ref_slices = int(model_kwargs.get("reference_slices", 1))
        self._ref_slab_size = int(model_kwargs.get("reference_slab_size", 1))
        self._ref_mode = model_kwargs.get("reference_mode", "random_slice")

        # Loss weights
        self._lambda_recon = float(model_kwargs.get("lambda_recon", 10.0))
        self._lambda_perceptual = float(model_kwargs.get("lambda_perceptual", 1.5))
        self._lambda_adv = float(model_kwargs.get("lambda_adv", 1.0))
        self._lambda_ref_consistency = float(model_kwargs.get("lambda_ref_consistency", 2.0))
        self._lambda_content_preservation = float(
            model_kwargs.get("lambda_content_preservation", 5.0)
        )

        # SSOT: Use UnifiedReconstructionLossComputer for reconstruction losses
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )
        # SSOT: Use registry losses for adversarial and reference consistency
        self._bce_loss = create_loss("bce_with_logits")
        self._ref_criterion = create_loss("mse")
        self._content_criterion = create_loss("l1")

        # Structural / edge losses are constructed ONCE here and moved to the
        # training device. Previously they were imported and re-instantiated
        # inside ``gen_closure`` every step (an nn.Module alloc in the hot path)
        # AND constructed with an unsupported ``channel=`` kwarg behind a bare
        # ``except Exception: pass`` — so ``SSIMLoss`` raised ``TypeError`` on
        # every call and the term was silently dropped (pitfall #10:
        # silently-omitted losses). Build them correctly, once, on-device.
        self._ssim_loss = SSIMLoss(window_size=7).to(self.device)
        self._hfen_loss = HFENLoss().to(self.device)

        logger.info(
            "GuidedSR setup: ref_mode=%s, ref_slices=%d, "
            "λ_recon=%.1f, λ_percept=%.1f, λ_adv=%.1f, λ_ref=%.1f, λ_content=%.1f",
            self._ref_mode,
            self._ref_slices,
            self._lambda_recon,
            self._lambda_perceptual,
            self._lambda_adv,
            self._lambda_ref_consistency,
            self._lambda_content_preservation,
        )

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        """Project a (possibly 5-D) volume to ``[B, C, H, W]`` for 2-D losses.

        ``SSIMLoss``/``HFENLoss`` are 2-D image losses (``B, C, H, W = x.shape``
        in ``HFENLoss.forward``), so a 5-D ``[B, C, D, H, W]`` volume is folded
        depth-into-batch (the same projection the 2-D PatchGAN path uses). 4-D
        tensors pass through unchanged.
        """
        if x.ndim == 5:
            b, c, d, h, w = x.shape
            return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        return x

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Not called directly — GuidedSuperResolutionStrategy overrides train_step().

        Returns an empty dict to satisfy the base class contract without crashing.
        All loss computation is handled inline within :meth:`train_step`.
        """
        return {}

    def _extract_reference(
        self,
        hf_volume: torch.Tensor,
        mode: str = "random_slice",
    ) -> torch.Tensor:
        """Extract partial HF reference from the full HF ground truth.

        During training, we simulate having only a partial HF scan by
        extracting random contiguous slices from the full HF volume.

        Args:
            hf_volume: Full HF volume [B, C, (D,) H, W].
            mode: Extraction mode:
                - 'random_slice': Single random axial slice
                - 'random_slab': Contiguous slab of ref_slab_size slices
                - 'center_slice': Center slice (deterministic)
                - 'center_slab': Center slab (deterministic)

        Returns:
            Partial HF reference tensor. For 3D: [B, C, K, H, W] where K=ref_slices.
            For 2D: returns the input (no slicing needed).
        """
        # 2D case: no slicing possible
        if hf_volume.ndim == 4:
            return hf_volume

        B, C, D, H, W = hf_volume.shape
        k = min(self._ref_slab_size, D)

        if mode == "random_slice":
            idx = random.randint(0, D - 1)
            return hf_volume[:, :, idx : idx + 1, :, :]
        elif mode == "random_slab":
            start = random.randint(0, max(0, D - k))
            return hf_volume[:, :, start : start + k, :, :]
        elif mode == "center_slice":
            mid = D // 2
            return hf_volume[:, :, mid : mid + 1, :, :]
        elif mode == "center_slab":
            mid = D // 2
            start = max(0, mid - k // 2)
            return hf_volume[:, :, start : start + k, :, :]
        else:
            msg = f"Unknown reference mode: {mode}"
            raise ValueError(msg)

    def _build_reference_input(
        self,
        ulf_volume: torch.Tensor,
        hf_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Build the combined model input from ULF volume + HF reference.

        Creates a zero-padded HF reference volume matching the ULF spatial
        dimensions, then concatenates channel-wise for the model input.

        For 3D: ULF [B, C, D, H, W] + padded HF ref [B, C, D, H, W] → [B, 2C, D, H, W]
        For 2D: ULF [B, C, H, W] + HF ref [B, C, H, W] → [B, 2C, H, W]

        Args:
            ulf_volume: Full ULF input.
            hf_reference: Partial HF reference (fewer slices).

        Returns:
            Combined input tensor.
        """
        if ulf_volume.ndim == 4:
            # 2D: concatenate directly
            if hf_reference.shape[2:] != ulf_volume.shape[2:]:
                hf_reference = F.interpolate(
                    hf_reference,
                    size=ulf_volume.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return torch.cat([ulf_volume, hf_reference], dim=1)

        # 3D: zero-pad the reference to match ULF depth
        B, C, D, H, W = ulf_volume.shape
        _, _, D_ref, _, _ = hf_reference.shape

        # Create zero-padded HF reference volume
        hf_padded = torch.zeros_like(ulf_volume)

        # Place reference slices at the corresponding position
        # Center the reference spatially in the volume
        start = max(0, (D - D_ref) // 2)
        end = min(D, start + D_ref)
        hf_padded[:, :, start:end, :, :] = hf_reference[:, :, : end - start, :, :]

        return torch.cat([ulf_volume, hf_padded], dim=1)

    @accepts_step_io
    def train_step(
        self,
        batch: Any,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute one training step for guided super-resolution.

        Args:
            batch: Training batch containing ULF and HF paired data.
            epoch: Current epoch.
            **kwargs: Additional arguments.

        Returns:
            Dict of loss values and metrics.
        """
        device = self.device
        generator = self.env.generator
        generator.train()

        # Extract paired data
        ulf_input = batch.get("input", batch.get("source"))
        hf_target = batch.get("target", batch.get("source_b"))

        if ulf_input is None or hf_target is None:
            msg = "Batch must contain paired ULF input and HF target"
            raise ValueError(msg)

        ulf_input = ulf_input.to(device, non_blocking=True)
        hf_target = hf_target.to(device, non_blocking=True)

        # Extract partial HF reference from ground truth (training simulation)
        hf_reference = self._extract_reference(hf_target, mode=self._ref_mode)

        # Build combined input: ULF + zero-padded HF reference
        combined_input = self._build_reference_input(ulf_input, hf_reference)

        # Generator forward pass
        def gen_closure() -> torch.Tensor:

            # Generate HF output from combined input
            hf_output = generator(combined_input)
            # Handle tuple output from models like DisentangledMRI (output, mu, logvar)
            if isinstance(hf_output, tuple):
                hf_output = hf_output[0]

            losses = {}

            # 1. Reconstruction loss via SSOT loss_computer
            env_losses = {}
            if self.env and hasattr(self.env, "losses"):
                env_losses = self.env.losses or {}

            loss_output = self.loss_computer.compute(
                pred=hf_output,
                target=hf_target,
                epoch=epoch,
                losses_dict=env_losses,
            )
            # Only the SCALED aggregate enters the summed loss. ``loss_output``
            # components are *detached* logging values (LossOutput contract) and
            # ``loss_output.total == Σ weight·component`` — re-adding the
            # components here previously double-counted the reconstruction
            # energy in ``total``.
            losses["recon"] = loss_output.total * self._lambda_recon

            # 2. SSIM-based structural loss (module built once on-device in
            #    _setup_strategy_specific_components). 2-D loss → project 5-D
            #    volumes to [B*D, C, H, W]. No try/except: a genuine failure
            #    must surface, not be silently dropped (pitfall #10).
            losses["ssim"] = self._ssim_loss(self._to_4d(hf_output), self._to_4d(hf_target)) * 2.0

            # 3. HFEN (High Frequency Error Norm) for edge preservation
            losses["hfen"] = self._hfen_loss(self._to_4d(hf_output), self._to_4d(hf_target)) * 2.0

            # 4. Reference consistency via SSOT registry loss
            if hf_output.ndim == 5 and hf_reference.ndim == 5:
                B, C, D, H, W = hf_output.shape
                D_ref = hf_reference.shape[2]
                start = max(0, (D - D_ref) // 2)
                end = min(D, start + D_ref)
                ref_slice_output = hf_output[:, :, start:end, :, :]
                ref_loss = self._ref_criterion(
                    ref_slice_output, hf_reference[:, :, : end - start, :, :]
                )
                losses["ref_consistency"] = ref_loss * self._lambda_ref_consistency

            # 5. Content preservation via SSOT registry loss
            if hf_output.ndim == ulf_input.ndim:
                if hf_output.ndim == 5:
                    ulf_sim = F.interpolate(
                        hf_output,
                        size=ulf_input.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    ulf_sim = F.interpolate(
                        hf_output,
                        size=ulf_input.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                content_loss = self._content_criterion(ulf_sim, ulf_input)
                losses["content_preserve"] = content_loss * self._lambda_content_preservation

            # 6. Adversarial loss (PatchGAN)
            if hasattr(self.env, "discriminator") and self.env.discriminator is not None:
                disc = self.env.discriminator
                disc_input = hf_output
                if disc_input.ndim == 5:
                    # Flatten 3D for 2D PatchGAN
                    B, C, D, H, W = disc_input.shape
                    disc_input = disc_input.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                pred_fake = disc(disc_input)
                adv_loss = self._bce_loss(pred_fake, torch.ones_like(pred_fake))
                losses["adv_gen"] = adv_loss * self._lambda_adv

            total = sum(losses.values())

            # A non-finite total is left to the executor's pre-backward guard, the
            # one owner of that rule; a second check here raised under AMP too,
            # where the GradScaler is meant to skip the step (#1471).
            return total

        # Discriminator step (if available)
        def disc_closure() -> torch.Tensor:
            if not hasattr(self.env, "discriminator") or self.env.discriminator is None:
                return torch.tensor(0.0, device=device)
            if not hasattr(self.env, "opt_d") or self.env.opt_d is None:
                return torch.tensor(0.0, device=device)
            disc = self.env.discriminator

            with torch.no_grad():
                hf_output = generator(combined_input)
                if isinstance(hf_output, tuple):
                    hf_output = hf_output[0]
                hf_output = hf_output.detach()

            # Prepare for 2D PatchGAN if 3D
            real_disc = hf_target
            fake_disc = hf_output
            if real_disc.ndim == 5:
                B, C, D, H, W = real_disc.shape
                real_disc = real_disc.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                fake_disc = fake_disc.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

            pred_real = disc(real_disc)
            pred_fake = disc(fake_disc)

            loss_real = self._bce_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = self._bce_loss(pred_fake, torch.zeros_like(pred_fake))
            d_loss = (loss_real + loss_fake) * 0.5

            return d_loss

        # Return OptimizationStepConfigs for the Trainer orchestrator
        step_configs = [
            {
                "optimizer": self.env.opt_g,
                "closure": gen_closure,
                "model": generator,
                "name": "generator",
            }
        ]

        # Add discriminator step if available
        if (
            hasattr(self.env, "discriminator")
            and self.env.discriminator is not None
            and hasattr(self.env, "opt_d")
            and self.env.opt_d is not None
        ):
            step_configs.append(
                {
                    "optimizer": self.env.opt_d,
                    "closure": disc_closure,
                    "model": self.env.discriminator,
                    "name": "discriminator",
                }
            )

        return step_configs

    @accepts_step_io
    def validation_step(
        self,
        batch: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for guided SR.

        The reference is zeros with the training slab's shape: the HF ground
        truth must not reach the generator input, or val_psnr measures a leak.

        Args:
            batch: Validation batch.
            **kwargs: Additional arguments.

        Returns:
            Dict of validation metrics.
        """
        device = self.device
        generator = self.env.generator
        generator.eval()

        ulf_input = batch.get("input", batch.get("source"))
        hf_target = batch.get("target", batch.get("source_b"))

        if ulf_input is None or hf_target is None:
            return {}

        ulf_input = ulf_input.to(device, non_blocking=True)
        hf_target = hf_target.to(device, non_blocking=True)

        # Shape-only zero reference (no GT values); see the docstring.
        ref_shape = list(hf_target.shape)
        if hf_target.ndim == 5:
            ref_shape[2] = min(self._ref_slab_size, hf_target.shape[2])
        hf_reference = torch.zeros(ref_shape, device=device, dtype=ulf_input.dtype)
        combined_input = self._build_reference_input(ulf_input, hf_reference)

        with torch.no_grad():
            hf_output = generator(combined_input)
            if isinstance(hf_output, tuple):
                hf_output = hf_output[0]

        # Compute validation metrics via SSOT loss_computer
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        loss_output = self.loss_computer.compute(
            pred=hf_output,
            target=hf_target,
            epoch=0,
            losses_dict=env_losses,
        )
        mse = loss_output.total.item()
        psnr = -10 * torch.log10(torch.tensor(mse + 1e-10)).item()

        return {
            "val_psnr": psnr,
            "val_loss": mse,
        }
