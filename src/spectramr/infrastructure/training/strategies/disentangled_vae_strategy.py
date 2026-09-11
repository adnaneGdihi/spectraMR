"""Disentangled VAE Training Strategy

This strategy implements physics-aware disentangled VAE training, enforcing
anatomy consistency across different acquisition physics (e.g., field strengths)
while ensuring physics-fidelity in the latent space.
"""

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.acquisition_registry import AcquisitionRegistry
from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


class DisentangledVAETrainingStrategy(BaseTrainingStrategy):
    """Disentangled VAE training strategy for cross-physics anatomy learning.

    Separates anatomical content (invariant across field strengths) from physics
    acquisition parameters (field-specific). Learns unified anatomical representations
    that generalize across different MRI protocols and field strengths (e.g., 3T→0.05T).

    ## Core Concept

    **Problem**: Same anatomical structure appears different under different physics:
    - High-field (3T) vs ultra-low-field (0.05T) MRI show different contrast/SNR
    - T1/T2 relaxation effects vary with field strength
    - Standard VAE conflates anatomy with physics, poor generalization

    **Solution**: Factorize representation into:
    1. **Content Code (z_c)**: Anatomy-invariant features (tissue structure, shape)
    2. **Physics Code (z_p)**: Field-strength-specific acquisition parameters

    ## Architecture

    **Encoder(s)**:
    - ContentEncoder: Maps image → z_c (invariant across field strengths)
    - PhysicsEncoder: Maps image → z_p (field-strength-specific)

    **Latent Space**:
    - z_c: Anatomy code (continuous, 64-256 dims)
    - z_p: Physics code (continuous, 16-64 dims)
    - Both reparameterized via μ and logvar (VAE-style)

    **Generator(s)**:
    - Generator(z_c, z_p) → Image (synthesis with explicit physics control)

    ## Training Objectives

    **1. Self-Reconstruction** (Domain consistency):
    - Low-field image → encode_{low} → (z_c, z_p) → generate(z_c, z_p) → Low-field reconstruction
    - High-field image → encode_{high} → (z_c, z_p) → generate(z_c, z_p) → High-field reconstruction
    - L1 loss enforces fidelity within domain

    **2. Cross-Domain Translation** (Physics transfer):
    - Low-field image encode → z_c^{low}, z_p^{low}
    - Transfer physics: z_c^{low} + z_p^{high} → Generate high-field-like image
    - Enables synthesis of high-field data from low-field without paired acquisition

    **3. Anatomy Consistency** (Content invariance):
    - KL divergence between z_c^{low} and z_c^{high} small (same anatomy→same encoding)
    - Enforces anatomical features are field-strength invariant
    - Critical for transfer learning and domain generalization

    **4. Physics Consistency** (Acquisition fidelity):
    - z_p must encode actual physics (T1, T2, tissue types)
    - Optional physics regularization: z_p ∈ [realistic ranges]
    - Prevents trivial solutions where content encodes all information

    **5. Reconstruction Diversity** (VAE regularization):
    - KL(q(z_c|x)||p(z_c)): Standard VAE prior for content
    - KL(q(z_p|x)||p(z_p)): Standard VAE prior for physics
    - Prevents posterior collapse

    ## Configuration

    - `training.training_mode`: 'disentangled_vae'
    - `training.vae.content_dim`: Anatomy code dimensions (default 128)
    - `training.vae.physics_dim`: Physics code dimensions (default 32)
    - `training.vae.kl_weight_content`: β for content KL (0.01-1.0)
    - `training.vae.kl_weight_physics`: β for physics KL (0.001-0.1)
    - `training.disentangled.anatomy_consistency_weight`: λ_anatomy (0.1-1.0)

    ## Loss Components

    - **Reconstruction Loss**: L1(x, x̂) for self and cross-domain (weight: 1.0)
    - **Content KL**: KL(q_c||p_c) regularization (β_c: 0.1-1.0)
    - **Physics KL**: KL(q_p||p_p) regularization (β_p: 0.01-0.1)
    - **Anatomy Consistency**: KL(z_c^{low} || z_c^{high}) (λ_anatomy: 0.1-1.0)
    - **Perceptual Loss**: VGG-based feature matching (optional)

    ## Key Features

    ✅ **Field-Agnostic**: Content code works uniformly across field strengths
    ✅ **Physics Control**: Explicit physics parameters tunable at inference
    ✅ **Interpretable**: Physics code corresponds to MRI parameters (T1, T2, etc.)
    ✅ **Transferable**: Learn on 3T, apply to 0.05T with physics adjustment
    ✅ **Invertible**: Both directions (HF↔LF) supported via physics swap

    ## Inference

    1. Input: Low-field image
    2. Encode: z_c^{low}, z_p^{low} = Encoder(x_{low})
    3. **Option A - Self-Reconstruct**: Generate(z_c^{low}, z_p^{low}) → Denoised low-field
    4. **Option B - Cross-Domain**: Generate(z_c^{low}, z_p^{high}) → High-field-like image
    5. Output: Either denoised same-field or synthesized cross-field

    ## Advantages

    - ✅ **Generalization**: Content learned from any field → applies to all fields
    - ✅ **Physics Awareness**: Parameters interpretable, tunable
    - ✅ **Synthesis**: Generate realistic cross-field data without new acquisition
    - ✅ **Interpretable**: Physics code provides quantitative tissue info

    ## Disadvantages

    - ❌ **Complexity**: Dual encoders + generators hard to train
    - ❌ **Mode Collapse**: Content can collapse to ignore physics code
    - ❌ **Limited Data**: Requires well-paired low-field and high-field data

    Attributes:
        state: TrainingState with disentangled VAE model
        content_encoder: Encodes anatomy-invariant features
        physics_encoder: Encodes field-specific parameters
        generator: Decoder with explicit physics control
        device: Computation device (CUDA/CPU)

    References:
        - Higgins et al. (2018): beta-TCVAE: Learning Disentangled Representations
        - Kumar et al. (2018): Variational Lossy Autoencoder
        - Achituve et al. (2021): Disentangling By Factorising
    """

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
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize strategy-specific components."""
        self._verify_strategy_config(expected_modes=("disentangled_vae",))
        self._log_config_features(self.logging_service)

        # SSOT: Use loss computer for reconstruction losses
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )
        # Paradigm-specific: anatomy consistency (latent-space MSE)
        self._anatomy_criterion = create_loss("mse")

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute disentangled VAE losses.

        Args:
            input_batch: Low-field image (source domain).
            target_batch: High-field image (target domain).
            epoch: Current epoch.
            **kwargs: Additional context (e.g., 'batch' dict).
        """
        # 1. Unpack & Prepare Inputs
        # input_batch is already moved to device and prepared by BaseTrainingStrategy
        # But we might need original batch keys if passed in kwargs
        img_64 = input_batch
        img_3T = target_batch

        # Handle case where input/target might be same (self-recon only)
        if img_3T is None:
            img_3T = img_64

        B = img_64.shape[0]
        device = img_64.device

        # 2. Get Physics Vectors
        # TODO: Should come from metadata/batch if available
        # For now, preserving legacy hardcoded logic from original implementation
        phys_64 = AcquisitionRegistry.get_physics_vector("T1w", "64mT", device)
        phys_3T = AcquisitionRegistry.get_physics_vector("T1w", "3T", device)

        # Expand vectors to batch size
        phys_64 = phys_64.expand(B, -1)
        phys_3T = phys_3T.expand(B, -1)

        # 3. Forward Pass
        # We assume generator handles (image, physics_vector) signature
        # If generator is wrapped (e.g. Compiled), ensure it supports kwargs or args

        # API contract: this strategy requires the generator to return the
        # disentangled-VAE 3-tuple ``(output, mu, logvar)`` AND expose
        # ``encode_content`` / ``decode``.  Catch the mismatch loudly at
        # the first forward (CLAUDE.md #9) instead of letting the
        # 4-tuple-returning ``sparse_vae`` silently crash with
        # ``too many values to unpack (expected 3)``.
        gen = self.env.generator
        _unwrapped = gen.module if hasattr(gen, "module") else gen
        if not hasattr(_unwrapped, "encode_content") or not hasattr(_unwrapped, "decode"):
            raise TypeError(
                f"DisentangledVAETrainingStrategy requires a generator "
                f"exposing ``encode_content`` and ``decode`` "
                f"(e.g. model_type='disentangled_mri'); got "
                f"{type(_unwrapped).__name__} which does not. Re-check the "
                f"YAML's model.model_type."
            )

        # A. Auto-Encoding (Self-Reconstruction)
        # Note: BaseStrategy calls generator(input), but here we need 2 args.
        # We call state.generator directly.
        recon_64, mu_64, logvar_64 = self.env.generator(img_64, phys_64)
        recon_3T, mu_3T, logvar_3T = self.env.generator(img_3T, phys_3T)

        # B. Cross-Domain Translation (The Disentanglement Test)
        # Encode 64mT anatomy, decode with 3T physics -> Should look like 3T
        # We need access to 'encode_content' and 'decode' methods of the model.
        # If state.generator is DDP/Compiled, we might need to access .module or .orig_mod
        model = self.env.generator
        if hasattr(model, "module"):
            model = model.module

        # [FIX] Use encode_content to get the 4D anatomy code, not the 2D style code (mu_64)
        z_anatomy_64 = model.encode_content(img_64)
        z_anatomy_3T = model.encode_content(img_3T)
        synth_3T = model.decode(z_anatomy_64, phys_3T)

        # 4. Loss Calculation

        # Loss components
        losses = {}

        # A. Reconstruction Loss — SSOT via loss_computer
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        # Thread the live ``iteration`` so the computer's spatial-loss warm-up
        # gate advances; without it the l1/perceptual terms stay at weight 0
        # for the entire run (CLAUDE.md pitfall #16).
        iteration = int(kwargs.get("iteration", 0) or 0)
        recon_output_64 = self.loss_computer.compute(
            pred=recon_64,
            target=img_64,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )
        recon_output_3T = self.loss_computer.compute(
            pred=recon_3T,
            target=img_3T,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )
        loss_recon_self = recon_output_64.total + recon_output_3T.total
        losses["l1_self"] = loss_recon_self

        # B. Cross-Domain Loss — SSOT via loss_computer
        trans_output = self.loss_computer.compute(
            pred=synth_3T,
            target=img_3T,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )
        loss_translation = trans_output.total

        # Perceptual Loss (if available in context)
        perceptual_loss_fn = (self.env.losses if self.env else {}).get("perceptual")
        if perceptual_loss_fn:
            try:
                p_loss = perceptual_loss_fn(synth_3T, img_3T)
                if isinstance(p_loss, torch.Tensor):
                    p_loss = p_loss.mean()
                loss_translation = loss_translation + 0.1 * p_loss
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)
        losses["l_trans"] = loss_translation

        # C. Anatomy Consistency Loss (paradigm-specific latent-space metric)
        loss_anatomy = self._anatomy_criterion(z_anatomy_64, z_anatomy_3T)
        losses["l_anat"] = loss_anatomy

        # D. KL Divergence
        # loss_kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        loss_kl_64 = -0.5 * torch.sum(1 + logvar_64 - mu_64.pow(2) - logvar_64.exp())
        loss_kl_3T = -0.5 * torch.sum(1 + logvar_3T - mu_3T.pow(2) - logvar_3T.exp())
        loss_kl_raw = loss_kl_64 + loss_kl_3T

        # KL Weighting
        lambda_kl = self._get_loss_weight("kl_divergence", epoch)
        if lambda_kl == 1.0:  # Default if not found
            # Use v6.0 training config for KL weight
            if hasattr(self.config.training, "kl_anneal_end"):
                lambda_kl = self.config.training.kl_anneal_end
            else:
                lambda_kl = 0.0001  # Safe default

        loss_kl = loss_kl_raw * lambda_kl / B
        losses["kl"] = loss_kl

        # Total Loss
        # Weights (Legacy hardcoded weights from original strategy)
        w_trans = 10.0
        w_anat = 5.0

        total_loss = loss_recon_self + w_trans * loss_translation + w_anat * loss_anatomy + loss_kl

        losses["g_total_loss"] = total_loss

        # Populate reuse dict
        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(losses)

        # Compute training metrics (PSNR, SSIM) on the translation task.
        # Live iteration (loop_state seam): the loop passes ``iteration=``,
        # never ``step=``, so ``kwargs.get("step", 0)`` was a constant 0 and
        # the interval throttle fired on every batch (pitfall #16).
        current_step = resolve_loop_iteration(self)
        train_metrics = self._compute_training_metrics(
            pred=synth_3T,
            target=img_3T,
            config=self.config,
            current_step=current_step,
        )
        self._loss_dict_reuse.update(train_metrics)

        return self._loss_dict_reuse

    @accepts_step_io
    @torch.no_grad()
    def validation_step(
        self,
        batch: Any,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for Disentangled VAE."""
        if input_batch is None or target_batch is None:
            # Need strict unpacking for paired data if possible
            # But BaseStrategy unpacking usually works for input/target
            input_batch, target_batch = self._unpack_batch(batch)

        if input_batch is None or target_batch is None:
            return {}

        img_64 = input_batch.to(self.device, non_blocking=True)
        img_3T = target_batch.to(self.device, non_blocking=True)
        B = img_64.shape[0]

        phys_3T = AcquisitionRegistry.get_physics_vector("T1w", "3T", self.device)
        phys_3T = phys_3T.expand(B, -1)

        # Use model to translate 64mT -> 3T
        # Need to get physics-conditioned VAE methods
        model = self.env.generator
        model.eval()

        # Encode low field
        # We need to construct phys_64 for encoding?
        # Yes, standard VAE forward needs it.
        phys_64 = AcquisitionRegistry.get_physics_vector("T1w", "64mT", self.device)
        phys_64 = phys_64.expand(B, -1)

        _, mu_64, logvar_64 = model(img_64, phys_64)

        # Cross-domain reconstruction
        # [FIX] Use encode_content to get the 4D anatomy code, not the 2D style code
        z_anatomy = model.encode_content(img_64)
        fake_3T = model.decode(z_anatomy, phys_3T)

        # Compute standard metrics
        metrics = {}

        # Use unified validation computer
        try:
            computer = self._get_validation_metrics_computer(self.config)
            computed = computer.compute(fake_3T, img_3T)
            for k, v in computed.items():
                metrics[f"val_{k}"] = v
        except Exception as _exc:
            logger.warning("[DisentangledVAE] Validation metrics failed: %s", _exc, exc_info=True)

        return self._convert_metrics_to_floats(metrics)
