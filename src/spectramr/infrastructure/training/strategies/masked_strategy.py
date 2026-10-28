"""Masked Pretraining Strategy
===========================

Training strategy for Experiment 5.2: Foundation Model Pre-training.
Implements Masked Image Modeling (MIM) in image or k-space.
"""

import logging
from typing import Any, ClassVar

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


class MaskedPretrainingStrategy(ReconstructionTrainingStrategy):
    """Training strategy for Masked Image Modeling (MIM) pretraining.

    Implements self-supervised pretraining via masked image/k-space reconstruction.
    Masks patches and trains model to predict missing regions, learning rich
    anatomical representations without paired supervision.

    ## Core Concept

    **Objective**: Mask random 50% of image patches, train model to predict masked
    content using visible context. Encourages learning of robust spatial features
    and anatomical structure understanding.

    **Key Insight**: Predicting pixels is easier than learning semantics, but forces
    model to capture fine-grained spatial relationships essential for reconstruction.

    ## Pre-training Process

    1. **Masking**: Random masking of image patches (50% by default)
       - Each patch: 16×16 pixels (typical)
       - Masking ratio: Configurable (default 0.5)
       - Masking strategy: Random per-patch masking

    2. **Forward Pass**: Model processes masked image
       - Encoder sees visible patches + mask tokens
       - Decoder predicts all patches (including visible as auxiliary signal)
       - Loss computed only on masked regions

    3. **Reconstruction Loss**: Per-patch L2/L1 loss
       - Target: Original pixel values in masked patches
       - Reconstruction quality measured by PSNR/SSIM

    4. **Fine-tuning Transfer**: Representations transfer to downstream tasks
       - Freeze encoder, train task-specific decoder
       - Typically achieves 10-20% better performance than random init

    ## Variants & Domain Adaptation

    **Image-Space MIM**:
    - Mask patches in spatial domain
    - Predicts RGB/grayscale pixel values
    - Preserves spatial locality of missing patches

    **K-Space MIM** (MRI-specific):
    - Mask k-space frequency patterns (cartesian/radial)
    - Predicts complex k-space coefficients
    - Captures both magnitude and phase information
    - More aligned with MRI physics than image-space

    ## Configuration

    - `training.training_mode`: 'masked' or 'reconstruction'
    - `training.masked.mask_ratio`: Fraction masked (0.5 = 50%)
    - `training.masked.patch_size`: Patch dimensions (16, 32, etc.)
    - `training.masked.masking_strategy`: 'random', 'block', 'non_overlapping'
    - `objectives.reconstruction.lambda_l1`: Reconstruction weight

    ## Loss Components

    - **Masked Region Loss**: L1/L2 on predicted masked patches (primary)
    - **Reconstruction Loss**: Full image L1/L2 (optional auxiliary)
    - **Perceptual Loss**: VGG-based feature matching (optional)
    - **Frequency Loss**: K-space consistency (for k-space MIM)

    ## Key Features

    - **Patch-Based Masking**: Independent per-patch masking enables efficient batching
    - **Asymmetric Encoder-Decoder**: Lightweight encoder, heavy decoder (Vision Transformer style)
    - **Efficient Implementation**: Uses mask tokens instead of zeros (reduces information leakage)
    - **Curriculum Learning**: Optional gradual mask ratio increase

    ## Pretraining Benefits

    ✅ **Self-Supervised**: No labels required, uses raw image structure
    ✅ **Transferable**: Representations benefit diverse downstream tasks
    ✅ **Robust**: Learns denoising and inpainting-like skills
    ✅ **Scalable**: Works with unlabeled data (large image databases)
    ✅ **Interpretable**: Visualization shows learned patch patterns

    ## Fine-tuning Strategy

    1. **Feature Extraction**: Use pre-trained encoder for downstream task
    2. **Task Head**: Either:
       - Train lightweight decoder for specific task
       - Train classifier on encoded features
    3. **Warmup**: Keep encoder frozen for 1-2 epochs, then unfreeze for full fine-tuning
    4. **Learning Rate**: Use lower LR for pre-trained blocks

    Attributes:
        state: TrainingState with generator supporting mask input
        loss_computer: UnifiedReconstructionLossComputer for reconstruction
        mask_ratio: Fraction of patches to mask (default 0.5)
        patch_size: Size of patches (e.g., 16 for 16×16 pixels)
        patch_mask: Cached patch mask tensor for current batch
        device: Computation device (CUDA/CPU)

    References:
        - He et al. (2022): Masked Autoencoders Are Scalable Vision Learners (MAE)
        - Baevski et al. (2023): Data2vec: A General Framework for Self-supervised Learning
    """

    #: Loss ownership (issue #1918). Hands env.losses to UnifiedMAELossComputer as losses_dict -- which ACCEPTS
    #: the argument and never reads it, so route 4 is a facade here and every
    #: declared image loss is silently discarded (issue #1918).
    #: The computer does compute image-space l2 + l1 against the target, hence the
    #: inline declaration. CAVEAT: their weights come from the legacy
    #: config.losses.reconstruction.lambda_l2 / lambda_l1 section, NOT from the
    #: loss-weight table, so a declared lambda on those two names is not honoured.
    inline_losses: ClassVar[frozenset[str]] = frozenset({"l1", "l2"})
    folds_image_losses: ClassVar[bool] = False

    def __init__(
        self,
        env=None,
        mask_ratio: float = 0.5,
        patch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            env (Any): Description.
            mask_ratio (float): Description.
            patch_size (int): Description.
        """
        super().__init__(env=env, **kwargs)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def _create_patch_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create patch-based mask for MIM.

        Args:
            B: Batch size
            H, W: Image height and width
            device: Device

        Returns:
            mask: [B, 1, H, W] boolean mask (True = masked)
        """
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w
        num_masked_patches = int(total_patches * self.mask_ratio)

        # Create patch-level mask for each sample in batch
        masks = []
        for _ in range(B):
            # Random selection of patches to mask
            patch_indices = torch.randperm(total_patches, device=device)[:num_masked_patches]

            # Create patch mask.
            #
            # The row/column form of this write is a flat scatter: for a
            # contiguous 2-D tensor, ``[idx // W, idx % W]`` addresses exactly
            # ``flat[idx]``, so scattering into the flat view and reshaping is
            # the same set of ones. Doing it per element meant iterating a
            # CUDA tensor -- one host sync AND one kernel launch per masked
            # patch (192 of each per sample at 256x256 / patch 16), inside the
            # training step (non-negotiable 9).
            patch_mask = torch.zeros(total_patches, device=device)
            patch_mask[patch_indices] = 1
            patch_mask = patch_mask.view(num_patches_h, num_patches_w)

            # Upsample to image resolution (repeat each patch)
            mask = patch_mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(
                self.patch_size, dim=1
            )

            # Crop to exact size if needed
            mask = mask[:H, :W]
            masks.append(mask)

        # Stack and add batch and channel dimensions
        mask = torch.stack(masks, dim=0).unsqueeze(1)  # [B, 1, H, W]
        return mask > 0.5

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute MIM losses with patch-based masking.

        Args:
            input_batch: Input images [B, C, H, W]
            target_batch: Target images (usually same as input_batch for self-supervised)
        """
        B, C, H, W = input_batch.shape
        device = input_batch.device

        # 1. Create Patch-Based Mask
        mask = self._create_patch_mask(B, H, W, device)

        # 2. Apply Mask with learnable mask token
        # Instead of zeros, use a small random value as mask token
        mask_token = 0.5 + 0.1 * torch.randn(B, C, 1, 1, device=device)
        masked_input = input_batch * (~mask).float() + mask_token * mask.float()

        # 3. Forward Pass
        reconstructed = self.env.generator(masked_input)

        # 4. Compute Loss
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Check for configured losses
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}
        elif hasattr(self, "context") and self.context and hasattr(self.context, "loss_fn"):
            env_losses = self.context.loss_fn or {}

        if not hasattr(self, "loss_computer") or self.loss_computer is None:
            from spectramr.models.losses.computers import UnifiedMAELossComputer

            self.loss_computer = UnifiedMAELossComputer(config=self.config, device=self.device)

        # Use Unified Loss Computer for configured losses
        loss_output = self.loss_computer.compute(
            pred=reconstructed,
            target=target_batch,
            epoch=epoch,
            losses_dict=env_losses,
        )
        total_loss = loss_output.total
        losses.update(loss_output.components)

        # Add masked info for telemetry
        mask_expanded = mask.expand_as(input_batch)
        if mask_expanded.any():
            masked_mse = nn.functional.mse_loss(
                reconstructed[mask_expanded], target_batch[mask_expanded]
            )
            losses["masked_mse"] = masked_mse

        losses["g_total_loss"] = total_loss
        losses["mask_ratio"] = mask.float().mean()

        return losses
