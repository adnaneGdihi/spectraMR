"""MAEPretrainingStrategy Training Strategy Module

This module contains MAEPretrainingStrategy training strategies.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.sampling import MaskGenerator, MaskType
from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.models.losses.computers import UnifiedMAELossComputer

from .base import BaseTrainingStrategy
from .patch_masking import create_patch_mask
from .rng_streams import rank_seeded_generator

#: Validation masks are fixed per validation sample: seed = base + batch_idx * stride + i.
_VAL_MASK_SEED = 42
_VAL_MASK_STRIDE = 1 << 16


def kspace_masked_input(
    images: torch.Tensor, seeds: Sequence[int], mask_ratio: float
) -> torch.Tensor:
    """Undersample each sample's centred k-space with its own variable-density mask.

    ``seeds[i]`` seeds sample ``i``'s mask, so a batch never shares one mask (#2393).
    The acceleration is ``1 / (1 - mask_ratio)``. A real input returns the real part.
    """
    if len(seeds) != images.shape[0]:
        raise ValueError(f"need one mask seed per sample: {len(seeds)} for {images.shape[0]}")
    acceleration = 1.0 / (1.0 - mask_ratio) if mask_ratio < 1.0 else 4.0
    shape = tuple(images.shape[-2:])
    masks = torch.stack(
        [
            MaskGenerator(seed=int(seed)).generate_mask(
                mask_type=MaskType.VARIABLE_DENSITY, shape=shape, acceleration_factor=acceleration
            )
            for seed in seeds
        ]
    )
    masks = masks.view(images.shape[0], *([1] * (images.dim() - 3)), *shape)
    masks = masks.to(images.device, non_blocking=True)
    was_real = not torch.is_complex(images)
    x = torch.complex(images, torch.zeros_like(images)) if was_real else images
    recovered = ifft2c(fft2c(x) * masks)
    return recovered.real if was_real else recovered


class MAEPretrainingStrategy(BaseTrainingStrategy):
    """Masked Autoencoder (MAE) pretraining strategy for self-supervised learning.

    Implements MAE pretraining where random patches are masked and the model learns
    to reconstruct missing patches from visible context. Achieves strong unsupervised
    feature learning without labels or contrastive pairs.

    ## MAE Concept

    **Objective**: Mask 75% of image patches, train encoder-decoder to predict
    missing patches from visible ones. Forces learning of rich semantic features
    and spatial relationships without supervision.

    **Key Insight**: Unlike contrastive learning (SSL), MAE uses generative task
    (patch prediction) as self-supervision. More directly related to downstream
    reconstruction tasks.

    ## Architecture

    **Asymmetric Encoder-Decoder**:
    - **Encoder**: Processes only visible patches (→3x speed vs full image)
    - **Decoder**: Lightweight, reconstructs from encoder + mask tokens
    - Masking reduces computation significantly during pretraining

    **Patch Tokenization**:
    - Input image → Non-overlapping patches (16×16 typical)
    - Linear embedding for each patch
    - Positional embeddings preserve spatial information

    **Mask Tokens**:
    - Learnable tokens for masked patches
    - Decoder sees concatenation of [encoder output, mask tokens]
    - Symmetry restoration happens in decoder

    ## Training Process

    1. **Input**: Original image (e.g., 224×224)
    2. **Patching**: Divide into 196 patches (16×16 each)
    3. **Masking**: Randomly mask 60-75% of patches
    4. **Encoding**: Pass visible patches through encoder (3x faster)
    5. **Decoding**: Reconstruct all patches via decoder
    6. **Loss**: MSE on masked regions (visible regions as context)
    7. **Backward**: Update encoder and decoder jointly

    ## Configuration

    - `training.training_mode`: 'mae' or 'self_supervised'
    - `training.mae.mask_ratio`: Fraction masked (0.75 = 75%, typical)
    - `training.mae.patch_size`: Patch dimensions (16, 32)
    - `training.mae.decoder_depth`: Decoder layers (8-12)
    - `objectives.reconstruction.lambda_l1`: Reconstruction weight

    ## Loss Components

    - **Reconstruction Loss**: MSE on masked patch pixels (primary)
       - L_mae = ||x_hat - x||² computed only on masked patches
       - Visible patches provide context without loss signal

    - **Optional Perceptual Loss**: VGG features (if configured)

    - **Optional Adversarial Loss**: Discriminator on reconstructed image

    ## Key Features

    ✅ **Asymmetric**: Encoder sees only 25%, decoder reconstructs all (efficient)
    ✅ **Self-Supervised**: No labels or pairs required
    ✅ **Simple**: Generation task more interpretable than contrastive
    ✅ **Transferable**: Learned representations transfer to 15+ downstream tasks
    ✅ **Scalable**: Works with any image size or modality

    ## Pretraining Benefits

    ✅ **Faster Downstream**: 10-20% faster convergence on target task
    ✅ **Better Convergence**: Higher final accuracy than random init
    ✅ **Data Efficient**: Effective with limited labeled data (few-shot)
    ✅ **Robust**: Learned features generalize across domains

    ## Fine-tuning Strategy

    1. **Feature Extraction**: Use pre-trained encoder for downstream
    2. **Task Head**: Train classifier or regression head on encoder output
    3. **Warmup**: Freeze encoder for 1 epoch, then unfreeze for full fine-tuning
    4. **Learning Rate**: Lower LR for encoder (0.1-0.01x task LR)
    5. **Scheduler**: Cosine annealing with warmup restart per epoch

    ## Inference

    1. Input: Image patch (visible regions only)
    2. Encoder: Extract features from visible patches (→3x faster)
    3. Decoder: Reconstruct masked patches
    4. Output: Complete reconstructed image with filled-in patches

    ## Advanced Variations

    **Hierarchical MAE**:
    - Multi-scale patches: small + large simultaneously
    - Coarse-to-fine reconstruction

    **Token Masking**:
    - Instead of patch masking, mask individual tokens
    - Finer-grained reconstruction task

    **Hybrid MAE+Contrastive**:
    - Combine MAE with contrastive (SimCLR, MoCo)
    - Better representations than either alone

    Attributes:
        state: TrainingState with MAE model
        loss_computer: UnifiedMAELossComputer for reconstruction
        mask_ratio: Fraction of patches to mask (default 0.75)
        patch_size: Size of patches (e.g., 16 for 16×16)
        device: Computation device (CUDA/CPU)

    References:
        - He et al. (2022): Masked Autoencoders Are Scalable Vision Learners (MAE)
        - Xie et al. (2022): An Empirical Study of Training Self-Supervised Vision Transformers
    """

    #: The training-mask stream: one per rank, carried through the checkpoint.
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_mask_rng",)

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
        """
        super().__init__(env=env, **kwargs)

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedMAELossComputer(config=self.config, device=self.device)
        self._mask_rng = rank_seeded_generator(int(self.config.run.seed))
        self._setup_strategy_specific_components()

    def _mae_masking(self) -> tuple[str, float, int]:
        """``(mask_domain, mask_ratio, patch_size)``: the typed ``training.mae`` block, else task_settings."""
        training = self.config.training
        settings = getattr(self.config, "task_settings", None)
        if not isinstance(settings, dict) and training is not None:
            settings = getattr(self.config.training, "task_settings", None)
        settings = settings if isinstance(settings, dict) else {}
        mae_cfg = self.config.training.mae if training is not None else None
        if mae_cfg is not None:
            return mae_cfg.mask_domain, float(mae_cfg.mask_ratio), int(mae_cfg.patch_size)
        domain = settings.get("mask_domain", "image")
        if domain not in ("image", "kspace"):
            raise ValueError(f"mask_domain must be 'image' or 'kspace', got {domain!r}")
        return domain, float(settings.get("mask_ratio", 0.75)), int(settings.get("patch_size", 16))

    def _masked_generator_input(
        self, images: torch.Tensor, seeds: Sequence[int] | None, gen_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Corrupt ``images`` as ``training.mae`` declares; ``seeds=None`` draws from the training stream.

        A generator that drops patch tokens itself (``owns_spatial_masking``) receives the
        clean image on the image route, so the image is never masked twice.
        """
        mask_domain, mask_ratio, patch_size = self._mae_masking()
        if mask_domain == "kspace":
            if seeds is None:
                seeds = torch.randint(
                    0, 2**31 - 1, (images.shape[0],), generator=self._mask_rng
                ).tolist()
            # The k-space mask is the corruption; the model must not drop patches too.
            gen_kwargs["disable_spatial_masking"] = True
            return kspace_masked_input(images, seeds, mask_ratio)
        if getattr(unwrap_model(self.env.generator), "owns_spatial_masking", False):
            return images
        stream = self._mask_rng if seeds is None else torch.Generator().manual_seed(int(seeds[0]))
        height, width = images.shape[-2:]
        mask = create_patch_mask(
            images.shape[0], height, width, patch_size, mask_ratio, images.device, stream
        )
        mask = mask.view(images.shape[0], *([1] * (images.dim() - 3)), height, width)
        return images.masked_fill(mask, 0.0)

    def _setup_strategy_specific_components(self) -> None:
        """Initialize MAE-specific components and perform validation."""
        self._verify_strategy_config(
            expected_modes=(
                "self_supervised",
                "ssl_pretraining",
                "ssl",
                "mae",
                "mae_pretraining",
                "reconstruction",
            ),
        )
        self._log_config_features(self.logging_service)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for MAE pretraining step with AMP support.

        Phase 4b-2: Enhanced with LossResult infrastructure for metrics tracking.
        Self-supervised contrastive loss computation.

        Args:
            input_batch: Low-resolution input tensor batch (unused for MAE).
            target_batch: High-resolution input tensor batch (used as input and target).
            epoch: Current epoch index.
            **kwargs: Additional optional context (may include batch dict).

        Returns:
            Dictionary of loss tensors.
        """
        # Handle both batch formats via the SSOT seam (rather than re-reading
        # raw kwargs): _resolve_legacy_batch checks kwargs["batch"] then
        # input_batch.
        resolved = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(resolved, dict) and "target" in resolved:
            # MAE uses batch dict format: 'target' is the High-Res image
            input_batch = resolved["target"]
        else:
            # Use target_batch as input for MAE (self-supervised on high-res images)
            input_batch = target_batch

        # The target is the clean image, captured before masking overwrites the
        # generator input; a corrupted target makes the identity map optimal (pitfall #20).
        mae_target = input_batch

        # A fresh mask per sample and step, drawn from the run's seeded stream.
        input_batch = self._masked_generator_input(input_batch, None, kwargs)
        outputs = self.env.generator(input_batch, **kwargs)

        # Handle both dict and tensor outputs
        if isinstance(outputs, dict):
            reconstruction = outputs.get("reconstruction", outputs)
        elif isinstance(outputs, tuple):
            # Some models return (reconstruction, ...) tuple
            reconstruction = outputs[0]
        else:
            # Standard tensor output
            reconstruction = outputs

        # [FIX] Use Unified Loss Computer (SSOT)
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        iteration = int(kwargs.get("iteration", 0) or 0)
        loss_output = self.loss_computer.compute(
            pred=reconstruction,
            target=mae_target,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )

        total_loss = loss_output.total
        components = loss_output.components

        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(components)
        self._loss_dict_reuse["g_total_loss"] = total_loss

        return self._loss_dict_reuse

    @torch.no_grad()
    def validation_step(
        self,
        val_batch: Any,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for MAE pretraining.

        Accepts the standard pipeline calling convention:
        ``strategy.validation_step(val_batch, batch_idx=i)``

        Args:
            val_batch: Batch from validation dataloader (TrainingBatch or dict).
            batch_idx: Index of the current batch (unused but required by interface).
            **kwargs: Additional arguments.

        Returns:
            Dictionary of validation metrics (PSNR, SSIM, MSE).
        """
        # Unpack batch using base class helper (handles TrainingBatch and dict formats)
        input_data, target_data = self._unpack_batch(val_batch)

        # MAE uses target as both input and reconstruction target.
        # Fall back to input if target is None (e.g. unlabelled SSL data).
        source = target_data if target_data is not None else input_data
        if source is None:
            # Last resort: val_batch itself might be a raw tensor
            if isinstance(val_batch, torch.Tensor):
                source = val_batch
            else:
                raise ValueError(
                    f"MAEPretrainingStrategy.validation_step: could not extract "
                    f"tensor from batch of type {type(val_batch).__name__}"
                )

        input_batch = source.to(self.device, non_blocking=True)

        # Metrics grade against the clean image, captured before masking (pitfall #20).
        mae_target = input_batch

        gen_kwargs: dict[str, Any] = {}
        seeds = [
            _VAL_MASK_SEED + batch_idx * _VAL_MASK_STRIDE + i for i in range(input_batch.shape[0])
        ]
        input_batch = self._masked_generator_input(input_batch, seeds, gen_kwargs)

        self.env.generator.eval()
        outputs = self.env.generator(input_batch, **gen_kwargs)

        # Handle different output formats
        if isinstance(outputs, dict):
            reconstruction = outputs.get("reconstruction", outputs)
        elif isinstance(outputs, tuple):
            reconstruction = outputs[0]
        else:
            reconstruction = outputs

        # Standardized Metric Computation
        # ``validation`` is optional, so this must be guarded -- the shape every
        # sibling strategy already uses (``vae.py``, ``gan.py``,
        # ``field_cocycle_strategy.py``, ``graph_cold_diffusion_strategy.py``).
        # Unguarded, an arm with no ``validation:`` block raised
        # ``AttributeError: 'NoneType' object has no attribute 'scoring'`` inside
        # the validation forward, which the loop reports only as "all N validation
        # batch(es) raised an exception" -- the cause named nowhere in the message.
        _val_config = getattr(self.config, "validation", None)
        compute_img_metrics = _val_config.scoring.enable_image_metrics if _val_config else True

        metrics = {}
        if compute_img_metrics:
            # [Refactor] Use SSOT ValidationMetricsComputer
            try:
                computer = self._get_validation_metrics_computer(self.config)
                computed = computer.compute(reconstruction, mae_target)

                # Apply prefix
                for k, v in computed.items():
                    metrics[f"val_{k}"] = v
            except Exception as e:
                # Do not silently swallow a validation-metric failure: a
                # masked shape/AMP/OOM error here yields a green run with
                # silently-degraded validation (the early-stopping / best
                # monitor then reads missing metrics) — pitfall #10. Log and
                # re-raise so the failure surfaces.
                if hasattr(self, "logging_service"):
                    self.logging_service.log_warning(f"Validation metrics failed: {e}")
                raise

        return metrics
