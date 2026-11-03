"""Disentangled VAE Training Strategy

This strategy implements physics-aware disentangled VAE training, enforcing
anatomy consistency across different acquisition physics (e.g., field strengths)
while ensuring physics-fidelity in the latent space.
"""

import logging
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.distributed.owned_modules import OWNED_CALL
from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.disentangled_diagnostics import expand_row_names
from spectramr.infrastructure.training.strategies.disentangled_vae_physics import (
    SOURCE_FIELD_KEY,
    TARGET_FIELD_KEY,
    FieldSwapReadings,
    PhysicsAxis,
    resolve_physics_axis,
)
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)

_TISSUE_OUTPUT_MESSAGE = (
    "DisentangledVAETrainingStrategy grades image tensors, but the generator returned a "
    "tissue-parameter dict (model_kwargs.output_mode: tissue_params). Render it through "
    "the Bloch layer with training.strategy_class "
    "spectramr.infrastructure.training.strategies.disentangled_strategy."
    "DisentangledTrainingStrategy."
)


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
    - `training.disentangled_vae.physics_source`: where each row's physics vector comes
      from (fixed T1w 64 mT/3 T pair, the loader's per-row field stamps, or none for a
      single-field self-reconstruction); see `disentangled_vae_physics.py`.

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

    # parallel_compat: Every training step calls generator.encode_content / decode, outside the
    # wrapped forward, which the call declaration refuses under ddp.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        own = (("generator.encode_content", OWNED_CALL),)
        return (*super().owned_module_declarations(config), *own)

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
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    def _setup_strategy_specific_components(self) -> None:
        """Initialize strategy-specific components."""
        self._verify_strategy_config(expected_modes=("disentangled_vae",))
        self._log_config_features(self.logging_service)
        self._physics = resolve_physics_axis(self.config)
        logger.info(
            "[DisentangledVAE] training.disentangled_vae.physics_source=%s contrasts=%s",
            self._physics.source,
            self._physics.contrasts,
        )

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
        img_hf = target_batch
        if not self._physics.has_physics:
            return self._single_field_losses(img_64, img_hf, epoch, **kwargs)

        if img_hf is None:
            if self._physics.source == "paired_field_stamps":
                raise ValueError("physics_source=paired_field_stamps needs a target image per row")
            img_hf = img_64  # fixed_t1w_64mt_3t keeps its historical self-recon fallback

        B = img_64.shape[0]
        device = img_64.device

        # 2. Physics vectors of the source (input) and target fields, one row per sample.
        phys_64, phys_hf = self._physics.vectors(kwargs.get("batch"), B, device)

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
        _unwrapped = unwrap_model(gen)
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
        recon_hf, mu_hf, logvar_hf = self.env.generator(img_hf, phys_hf)

        # B. Cross-Domain Translation (The Disentanglement Test)
        # Encode 64mT anatomy, decode with 3T physics -> Should look like 3T
        model = self.inner_for_call(self.env.generator, "generator.encode_content")

        # [FIX] Use encode_content to get the 4D anatomy code, not the 2D style code (mu_64)
        z_anatomy_64 = model.encode_content(img_64)
        z_anatomy_hf = model.encode_content(img_hf)
        synth_hf = model.decode(z_anatomy_64, phys_hf)
        if any(isinstance(out, dict) for out in (recon_64, recon_hf, synth_hf)):
            raise TypeError(_TISSUE_OUTPUT_MESSAGE)

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
        recon_output_hf = self.loss_computer.compute(
            pred=recon_hf,
            target=img_hf,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )
        loss_recon_self = recon_output_64.total + recon_output_hf.total
        losses["l1_self"] = loss_recon_self

        # B. Cross-Domain Loss — SSOT via loss_computer
        trans_output = self.loss_computer.compute(
            pred=synth_hf,
            target=img_hf,
            epoch=epoch,
            iteration=iteration,
            losses_dict=env_losses,
        )
        loss_translation = trans_output.total

        # Perceptual Loss (if available in context)
        perceptual_loss_fn = (self.env.losses if self.env else {}).get("perceptual")
        if perceptual_loss_fn:
            try:
                p_loss = perceptual_loss_fn(synth_hf, img_hf)
                if isinstance(p_loss, torch.Tensor):
                    p_loss = p_loss.mean()
                loss_translation = loss_translation + 0.1 * p_loss
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)
        losses["l_trans"] = loss_translation

        # C. Anatomy Consistency Loss (paradigm-specific latent-space metric)
        loss_anatomy = self._anatomy_criterion(z_anatomy_64, z_anatomy_hf)
        losses["l_anat"] = loss_anatomy

        # D. KL Divergence of both encodings, weighted once.
        loss_kl = self._kl_term(mu_64, logvar_64, epoch) + self._kl_term(mu_hf, logvar_hf, epoch)
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
            pred=synth_hf,
            target=img_hf,
            config=self.config,
            current_step=current_step,
        )
        self._loss_dict_reuse.update(train_metrics)

        return self._loss_dict_reuse

    def _kl_term(self, mu: torch.Tensor, logvar: torch.Tensor, epoch: int) -> torch.Tensor:
        """Weighted KL of one ``(mu, logvar)`` pair, divided by the batch size."""
        kl_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lambda_kl = self._get_loss_weight("kl_divergence", epoch)
        if lambda_kl == 1.0 and hasattr(self.config.training, "kl_anneal_end"):
            lambda_kl = self.config.training.kl_anneal_end
        elif lambda_kl == 1.0:
            lambda_kl = 0.0001
        # A deterministic style code (use_vae false) has no posterior; its "KL" is an L2 on mu.
        if lambda_kl > 0 and not getattr(unwrap_model(self.env.generator), "use_vae", True):
            raise ValueError(
                f"KL weight {lambda_kl} declared on a deterministic latent "
                "(model_kwargs.use_vae: false). Set use_vae: true, or losses.latent.lambda_kl: 0."
            )
        return kl_raw * lambda_kl / mu.shape[0]

    def _single_field_losses(
        self, img: torch.Tensor, target: torch.Tensor | None, epoch: int, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """``single_field``: self-reconstruction with an image-derived style and its KL."""
        if target is None:
            raise ValueError("physics_source=single_field needs a target image per row")
        recon, mu, logvar = self.env.generator(img)
        if isinstance(recon, dict):
            raise TypeError(_TISSUE_OUTPUT_MESSAGE)
        env_losses = (self.env.losses or {}) if self.env and hasattr(self.env, "losses") else {}
        loss_recon = self.loss_computer.compute(
            pred=recon,
            target=target,
            epoch=epoch,
            iteration=int(kwargs.get("iteration", 0) or 0),
            losses_dict=env_losses,
        ).total
        loss_kl = self._kl_term(mu, logvar, epoch)
        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(
            {"l1_self": loss_recon, "kl": loss_kl, "g_total_loss": loss_recon + loss_kl}
        )
        self._loss_dict_reuse.update(
            self._compute_training_metrics(
                pred=recon,
                target=target,
                config=self.config,
                current_step=resolve_loop_iteration(self),
            )
        )
        return self._loss_dict_reuse

    #: Physics-axis source; ``_setup_strategy_specific_components`` resolves it from the config,
    #: and this class value is the schema default.
    _physics: PhysicsAxis = PhysicsAxis("fixed_t1w_64mt_3t")

    #: Rows of the running validation pass on the field axis; ``None`` outside a pass or
    #: when the physics source has no per-row fields.
    _field_readings: FieldSwapReadings | None = None

    def reset_validation_metrics(self) -> None:
        """Open a validation pass; the paired-field source also starts its readings."""
        super().reset_validation_metrics()
        paired = self._physics.source == "paired_field_stamps"
        self._field_readings = FieldSwapReadings() if paired else None

    def finalize_validation(self) -> dict[str, float]:
        """Close the pass and add the field-axis disentanglement readings."""
        summary = dict(super().finalize_validation())
        readings, self._field_readings = self._field_readings, None
        if readings is not None:
            summary.update(readings.summary())
        return summary

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
        img_hf = target_batch.to(self.device, non_blocking=True)
        B = img_64.shape[0]
        model = self.env.generator
        model.eval()

        if not self._physics.has_physics:
            fake_hf, _, _ = model(img_64)
        else:
            phys_64, phys_hf = self._physics.vectors(batch, B, self.device)
            z_anatomy = model.encode_content(img_64)
            fake_hf = model.decode(z_anatomy, phys_hf)
            if self._field_readings is not None and not isinstance(fake_hf, dict):
                kept_64 = model.decode(z_anatomy, phys_64)
                groups = expand_row_names(read_batch_field(batch, "subject_id"), B)
                self._field_readings.add(
                    content=(z_anatomy, model.encode_content(img_hf)),
                    physics=(phys_64, phys_hf),
                    fields=(
                        self._physics.row_fields(batch, SOURCE_FIELD_KEY, B),
                        self._physics.row_fields(batch, TARGET_FIELD_KEY, B),
                    ),
                    groups=groups,
                    swap=(fake_hf, kept_64, img_hf, model.encode_content(fake_hf)),
                )
        if isinstance(fake_hf, dict):
            raise TypeError(_TISSUE_OUTPUT_MESSAGE)

        # A failure here must raise: swallowed, it left val_psnr unemitted and the
        # early-stopping monitor reading nothing for the whole run (#2606).
        computer = self._get_validation_metrics_computer(self.config)
        metrics = {f"val_{k}": v for k, v in computer.compute(fake_hf, img_hf).items()}
        return self._convert_metrics_to_floats(metrics)
