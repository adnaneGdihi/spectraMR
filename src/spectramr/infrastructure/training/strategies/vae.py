"""VAEVQVAE Training Strategy Module

This module contains VAE and VQVAE training strategies.
"""

import logging
from typing import Any

import torch

from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.precision_annotations import (
    VAEPrecisionManager,
    VQVAEPrecisionManager,
)
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.models.losses.computers import (
    UnifiedVAELossComputer,
    UnifiedVQVAELossComputer,
)

from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class VAETrainingStrategy(BaseTrainingStrategy):
    """Variational Autoencoder (VAE) training strategy.

    Implements probabilistic latent variable model with encoder-decoder architecture.
    Combines reconstruction loss with KL divergence regularization for learned
    posterior distribution.

    ## VAE Objective

    The VAE loss combines two complementary objectives:

    1. **Reconstruction Loss**: Negative log-likelihood of decoding
       - Measures reconstruction quality (L1/L2 with target)
       - Encourages latent code informativeness

    2. **KL Regularization**: Divergence from prior distribution
       - Prevents posterior collapse via KL divergence
       - Encourages meaningful latent structure

    Total Loss = Reconstruction + β·KL(q||p)  where β is weighting factor

    ## Configuration Parameters

    - `training.training_mode`: Must be 'vae'
    - `training.vae.latent_dim`: Latent space dimensions (e.g., 128)
    - `training.vae.kl_weight`: KL divergence weight (β scheduling)
    - `model.model_type`: VAE architecture (e.g., 'standard_vae')

    ## Encoding Process

    1. Input image → Encoder → Mean (μ) & Logvariance (σ²)
    2. Sample latent z ~ N(μ, σ²) using reparameterization trick
    3. Latent z → Decoder → Reconstructed image

    ## Loss Components

    - **Reconstruction Loss**: L1 or L2 between prediction and target (configurable)
    - **KL Divergence**: -0.5·Σ[1 + logvar - mu² - exp(logvar)]
    - **Perceptual Loss**: Optional VGG-based feature matching (if configured)
    - **Adversarial Loss**: Optional GAN discriminator (if configured)

    ## Validation & Generation

    - **Validation**: PSNR/SSIM on reconstructed images via compute_metrics()
    - **Generation**: Sample from prior N(0,I) and decode
    - **Interpolation**: Smooth transitions in latent space

    ## Phase 4b-1 Enhancement

    Enhanced with explicit precision annotations to prevent silent casting errors
    and ensure numerically stable KL divergence computation. VAEPrecisionManager
    ensures KL computed in FP32 even when AMP enabled.

    Attributes:
        state: TrainingState with VAE model
        loss_computer: UnifiedVAELossComputer for multi-term loss aggregation
        precision_manager: VAEPrecisionManager for numerical stability
        device: Torch device for computation
    """

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
            device (Optional[torch.device]): Description.
        """
        super().__init__(env=env, device=device, **kwargs)

        # Explicitly access env/config for execution graph validation
        self._env = self.env
        self._config = self.config

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedVAELossComputer(config=self.config, device=self.device)

        # Phase 4b-1: Initialize precision manager for explicit tracking
        self.precision_manager = VAEPrecisionManager(device=self.device)

        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize VAE-specific components and perform validation."""
        self._verify_strategy_config(expected_modes=("vae", "reconstruction"))
        self._log_config_features(self.logging_service)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for VAE training step with AMP support.

        Phase 4b-1: Enhanced with explicit precision annotations.
        Ensures KL divergence maintained in FP32 for numerical stability
        via VAEPrecisionManager.

        Args:
            input_batch: Low-resolution input tensor batch.
            target_batch: High-resolution target tensor batch.
            epoch: Current epoch index.
            **kwargs: Additional optional context.

        Returns:
            Dictionary of loss tensors.
        """
        # Forward pass through VAE model (encoder may use AMP)
        # Output precision depends on autocast context (FP16 if AMP enabled)
        output = self.env.generator(input_batch)

        # Handle different model output formats
        if isinstance(output, tuple) and len(output) >= 3:
            # True VAE model returning (reconstruction, mu, logvar, [z])
            hr_fakes, mu, logvar = output[0], output[1], output[2]
            z_eval = output[3] if len(output) > 3 else None
        elif isinstance(output, dict):
            # Dict output format
            hr_fakes = output.get("reconstruction", output.get("output", input_batch))
            mu = output.get("mu", torch.zeros(input_batch.shape[0], 1, device=input_batch.device))
            logvar = output.get(
                "logvar",
                torch.zeros(input_batch.shape[0], 1, device=input_batch.device),
            )
            z_eval = output.get("z", None)
        else:
            hr_fakes = output
            # A model may return a BARE TENSOR and still be a real VAE, caching its
            # posterior for retrieval instead of widening the forward signature —
            # `slat_vae_slab_to_volume` does exactly this (`last_aux()` returns the
            # cached mu/logvar; its `forward` only emits them under an opt-in
            # `return_structured=True` that nothing on this path passes).
            #
            # Falling straight through to dummy zeros made KL identically 0 for those
            # arms: the beta schedule was routed correctly and then multiplied a
            # constant zero, so the "VAE" trained as a plain autoencoder with an
            # unregularised latent (a silent-fallback facade, pitfalls #9/#16). Ask
            # the model for its posterior before assuming it has none.
            last_aux = getattr(self.env.generator, "last_aux", None)
            aux = last_aux() if callable(last_aux) else None

            if (
                isinstance(aux, dict)
                and aux.get("mu") is not None
                and aux.get("logvar") is not None
            ):
                mu, logvar = aux["mu"], aux["logvar"]
                z_eval = aux.get("z")
            else:
                # Genuinely not a VAE (e.g. a plain UNet) — reconstruction-only, with
                # a zero KL contribution.
                mu = torch.zeros(input_batch.shape[0], 1, device=input_batch.device)
                logvar = torch.zeros(input_batch.shape[0], 1, device=input_batch.device)
                z_eval = None

        # Phase 4b-1: Use precision manager to ensure KL numerical stability
        # Casts to FP32 for KL computation only (avoids 2-5% convergence loss)
        mu_fp32, logvar_fp32 = self.precision_manager.prepare_latent_for_kl_computation(mu, logvar)

        # Reconstruction loss can stay in FP16 (less numerically sensitive)
        reconstruction_input = hr_fakes

        # Autoencoder semantics: an autoencoder arm MUST supply target ≡ input
        # (data.bidirectional_mode: hf_to_hf / ulf_to_ulf drops the opposite
        # arm so the self-supervised branch aliases target=input). Raise on a
        # missing / empty / shape-mismatched target instead of silently
        # substituting input_batch — that former fallback (pitfall #9) MASKED a
        # hf_to_ulf translation config as an autoencoder and corrupted the
        # frozen stage-2 latent (2026-07 ldm_two_stage_ulf_to_hf triage).
        if (
            target_batch is None
            or target_batch.numel() == 0
            or target_batch.shape != reconstruction_input.shape
        ):
            raise ValueError(
                "VAE reconstruction target is missing or shape-mismatched "
                f"(target={None if target_batch is None else tuple(target_batch.shape)}, "
                f"pred={tuple(reconstruction_input.shape)}). An autoencoder arm must "
                "supply target ≡ input: set data.bidirectional_mode: hf_to_hf / "
                "ulf_to_ulf, or use training_mode: reconstruction for a translator."
            )

        # [FIX] Use UnifiedVAELossComputer with losses_dict
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}
        elif hasattr(self, "context") and self.context and hasattr(self.context, "loss_fn"):
            env_losses = self.env.losses if self.env else {}

        # Thread the live ``iteration`` so the computer's KL-anneal schedule
        # (beta ramps from kl_beta_start to kl_beta_end over kl_anneal_steps)
        # actually advances. Dropping it froze beta at beta_start (default 0)
        # for the whole run — an unregularised autoencoder masquerading as a
        # VAE (CLAUDE.md pitfall #16/#20).
        iteration = int(kwargs.get("iteration", 0) or 0)
        loss_output = self.loss_computer.compute(
            pred=reconstruction_input,
            target=target_batch,
            epoch=epoch,
            iteration=iteration,
            posterior=(mu_fp32, logvar_fp32),
            z=z_eval,
            discriminator=kwargs.get("discriminator"),
            losses_dict=env_losses,
        )

        total_loss = loss_output.total
        components = loss_output.components

        # Get KL weight (beta for beta-VAE). Honor the configured value
        # verbatim — a deliberate ``lambda_kl_divergence: 0`` (β=0, e.g. an
        # autoencoder ablation) must stay 0. A previous override that promoted
        # a zero weight to one silently inflated β=0 to β=1, scaling the
        # ``kl_override`` term below against the user's intent (audit 2026-06).
        kl_weight = self._get_loss_weight("kl_divergence", epoch)

        # Build losses dict
        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(components)
        self._loss_dict_reuse.update(
            {
                "g_total_loss": total_loss,
                "g_loss_kl": components.get(
                    "kl_divergence", torch.tensor(0.0, device=total_loss.device)
                ),
                "kl_weight": kl_weight,
            }
        )

        # Phase 3 (2026-05-22): models with a non-Gaussian or aggregated
        # latent (ladder VAE, hyperspherical/vMF VAE, MoE-VAE) cannot express
        # their KL through a single Gaussian (mu, logvar) pair, so they emit a
        # ``kl_override`` (their exact KL) and optional ``aux_loss`` in the
        # output dict. We add these to the total here. The branch is additive
        # and only fires when the keys are present, so all existing Gaussian
        # VAE models are unaffected.
        if isinstance(output, dict):
            kl_override = output.get("kl_override")
            if kl_override is not None:
                total_loss = total_loss + kl_weight * kl_override
                self._loss_dict_reuse["g_loss_kl"] = kl_override
            aux_loss = output.get("aux_loss")
            if aux_loss is not None:
                total_loss = total_loss + aux_loss
                self._loss_dict_reuse["loss_aux"] = aux_loss
            self._loss_dict_reuse["g_total_loss"] = total_loss

        if "loss" not in self._loss_dict_reuse:
            self._loss_dict_reuse["loss"] = total_loss

        # [ENHANCEMENT] Compute training metrics (PSNR, SSIM, MAE) for monitoring.
        # Use the live loop ``iteration`` (resolved above) — ``self.env.step`` is
        # frozen at 0, which would defeat the train_metric_interval throttle and
        # recompute SSIM/PSNR/MAE every step (pitfall #16).
        train_metrics = self._compute_training_metrics(
            pred=hr_fakes,
            target=target_batch,
            config=self.config,
            current_step=iteration,
        )

        # Phase 5: Ensure strict type compliance (dict[str, Tensor])
        for k, v in train_metrics.items():
            if not isinstance(v, torch.Tensor):
                self._loss_dict_reuse[k] = torch.tensor(v, device=hr_fakes.device)
            else:
                self._loss_dict_reuse[k] = v

        return self._loss_dict_reuse

    @accepts_step_io
    def validation_step(
        self,
        batch: Any,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, float]:
        # Use absolute import to avoid resolution errors

        """validation_step.

        Args:
            batch (Any): Description.
            input_batch (Optional[torch.Tensor]): Description.
            target_batch (Optional[torch.Tensor]): Description.
            epoch (int): Description.
        Returns:
            dict[str, float]: Description.
        """
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        with torch.no_grad():
            # VAE validation typically uses reconstruction error
            vae_output = self.env.generator(input_batch)

            # Handle different output formats
            if isinstance(vae_output, tuple) and len(vae_output) >= 3:
                # VAE returns tuple: (reconstruction, mu, logvar)
                hr_fakes, mu, logvar = vae_output[0], vae_output[1], vae_output[2]
            elif isinstance(vae_output, dict):
                # Handle dict output for compatibility
                hr_fakes = vae_output.get("reconstruction", vae_output.get("output", input_batch))
                mu = vae_output.get(
                    "mu",
                    torch.zeros(input_batch.shape[0], 1, device=input_batch.device),
                )
                logvar = vae_output.get(
                    "logvar",
                    torch.zeros(input_batch.shape[0], 1, device=input_batch.device),
                )
            else:
                # Non-VAE model (e.g., UNet) - just use reconstruction
                hr_fakes = vae_output
                mu = torch.zeros(input_batch.shape[0], 1, device=input_batch.device)
                logvar = torch.zeros(input_batch.shape[0], 1, device=input_batch.device)

            # Autoencoder semantics: target ≡ input is guaranteed by the data
            # layer (bidirectional_mode: hf_to_hf / ulf_to_ulf). Raise on a
            # missing / mismatched target rather than silently substituting
            # input — the removed fallback masked a translation config as an
            # autoencoder (pitfall #9); see ``_compute_losses_impl`` above.
            if (
                target_batch is None
                or target_batch.numel() == 0
                or target_batch.shape != hr_fakes.shape
            ):
                raise ValueError(
                    "VAE validation target is missing or shape-mismatched "
                    f"(target={None if target_batch is None else tuple(target_batch.shape)}, "
                    f"pred={tuple(hr_fakes.shape)}). An autoencoder arm must supply "
                    "target ≡ input: set data.bidirectional_mode: hf_to_hf / "
                    "ulf_to_ulf, or use training_mode: reconstruction."
                )

            # Ensure tensors are on the same device
            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            # Standardized Metric Computation
            val_config = getattr(self.config, "validation", None)
            compute_img_metrics = (
                (val_config.scoring.enable_image_metrics if val_config else True)
                if val_config
                else True
            )

            metrics = {}
            if compute_img_metrics:
                # [Refactor] Use SSOT ValidationMetricsComputer
                try:
                    computer = self._get_validation_metrics_computer(self.config)
                    computed = computer.compute(hr_fakes, target_batch)

                    # Apply prefix
                    for k, v in computed.items():
                        metrics[f"val_{k}"] = v
                except Exception as e:
                    if hasattr(self, "logging_service") and logger.isEnabledFor(logging.WARNING):
                        self.logging_service.log_warning("Validation metrics failed: %s", str(e))

            # Add VAE-specific metrics (only if mu/logvar are meaningful)
            if mu.numel() > 0:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                metrics["kl_divergence"] = kl_loss
                metrics["latent_std"] = torch.exp(0.5 * logvar).mean()

            return self._convert_metrics_to_floats(metrics)

    # NOTE: the former ``_get_kl_weight`` annealing helper was removed (audit
    # 2026-06). It had zero callers and was the sole reader of
    # ``anneal_kl_beta``/``kl_beta_start``/``kl_anneal_steps``; the live KL path
    # is ``UnifiedVAELossComputer._get_loss_weight("kl", ..., iteration)``, which
    # now performs the start→end linear annealing those knobs describe.


class VQVAETrainingStrategy(BaseTrainingStrategy):
    """Vector Quantized Variational Autoencoder (VQ-VAE) training strategy.

    Combines discrete latent codes with continuous features for stable,
    high-quality image reconstruction. Avoids posterior collapse via quantization
    constraint instead of KL regularization.

    ## VQ-VAE Architecture

    1. **Encoder**: Image → Continuous latent embedding z_e ∈ ℝ^(H×W×D)
    2. **Vector Quantizer**: Discrete codebook with nearest-neighbor lookup
       - Maps z_e to nearest codebook vector z_q via L2 distance
       - Maintains pool of K diverse code vectors (codebook)
       - Implements straight-through gradient estimator
    3. **Decoder**: Quantized codes z_q → Reconstructed image

    ## Key Advantages Over Standard VAE

    - ✅ Avoids posterior collapse (discrete codes prevent KL→0)
    - ✅ Stable training (no KL annealing scheduling needed)
    - ✅ Better reconstruction quality (fewer samples wasted on variance)
    - ✅ Codes amenable to further generation (e.g., autoregressive prior)
    - ✅ Interpretable code usage (discrete codebook analysis)

    ## Loss Components

    1. **Reconstruction Loss**: MSE/L1 between input and decoder output
       - Main training signal (primary loss)
       - Encourages accurate reconstructions

    2. **Codebook Loss** (commitment): Weight codes toward encoder outputs
       - β·||sg[z_e] - z_q||²  where sg[] = stop-gradient
       - Prevents codebook drift relative to encoder evolution

    3. **Dictionary Loss**: Update codebook toward encoder outputs
       - γ·||z_e - sg[z_q]||²
       - Ensures codebook stays current with changing encoder

    Total Loss = MSE(x, Dec(Q(Enc(x)))) + β·||sg[z_e] - z_q||² + γ·||z_e - sg[z_q]||²

    where Q = quantization operation, z_e = encoder latent, z_q = quantized vector

    ## Training Configuration

    - `training.training_mode`: 'vqvae' or 'reconstruction'
    - `training.vae.latent_dim`: Dimensionality of code vectors (e.g., 4)
    - `training.vae.num_codes`: Size of codebook (e.g., 512, 1024)
    - `training.vae.commitment_cost`: β weighting for codebook loss (default 0.25)
    - `training.vae.commitment_weight`: γ weighting for dictionary loss
    - `model.model_type`: VQ-VAE architecture (e.g., 'vqvae_unet')

    ## Code Pruning & Diversity Monitoring

    Tracks key metrics to identify under-utilized or collapsed codes:
    - **Perplexity**: Effective number of codes used (target: high, e.g., >0.95*K)
    - **Usage**: Count of unique codes assigned per batch (histogram)
    - **Entropy**: Shannon entropy of code distribution (0=collapsed, 1=uniform)
    - **Dead Codes**: Zero-usage codes flagged for potential reset

    ## Inference

    1. **Encoding**: Image → Encoder → z_e {nearest codebook lookup} → z_q
    2. **Decoding**: z_q → Decoder → Reconstructed image
    3. **Generation**: Sample codes from learned prior (if trained) → Decode
    4. **Interpolation**: Linear interpolation in quantized code space

    ## Phase 4b-1 Enhancement

    Enhanced with explicit precision annotations to prevent silent casting errors
    and ensure numerically stable VQ loss computation. VQVAEPrecisionManager tracks
    vector quantization loss in FP32 for gradient stability.

    Attributes:
        state: TrainingState with VQ-VAE model
        loss_computer: UnifiedVQVAELossComputer for multi-term loss aggregation
        precision_manager: VQVAEPrecisionManager for numerical stability
        codebook: Discrete quantization codebook vectors
        num_codes: Size of codebook (e.g., 512)
        latent_dim: Dimensionality of each code vector
        device: Device for computation (CUDA/CPU)
        perplexity: Current codebook usage perplexity metric
    """

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
            device (Optional[torch.device]): Description.
        """
        super().__init__(env=env, device=device, **kwargs)

        # Explicitly access env/config for execution graph validation
        self._env = self.env
        self._config = self.config

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedVQVAELossComputer(config=self.config, device=self.device)

        # Phase 4b-1: Initialize precision manager for VQ loss tracking
        self.precision_manager = VQVAEPrecisionManager(device=self.device)

        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize VQ-VAE-specific components and perform validation."""
        self._verify_strategy_config(expected_modes=("vae", "vqvae", "reconstruction"))
        self._log_config_features(self.logging_service)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for VQ-VAE training step with AMP support.

        Phase 4b-1: Enhanced with explicit precision annotations.
        Vector quantization loss maintained in FP32 for gradient stability
        via VQVAEPrecisionManager.

        Args:
            input_batch: Low-resolution input tensor batch.
            target_batch: High-resolution target tensor batch.
            epoch: Current epoch index.
            **kwargs: Additional optional context.

        Returns:
            Dictionary of loss tensors.
        """
        # Forward pass - use generator, falling back to model if needed
        model = getattr(self.state, "generator", None) or getattr(self.state, "model", None)
        if model is None:
            raise AttributeError("Neither state.generator nor state.model is available")
        outputs = model(input_batch)

        # Handle different output formats
        if isinstance(outputs, dict):
            reconstruction = outputs.get("reconstruction", outputs.get("output", input_batch))
            vq_loss = outputs.get("vq_loss")
            perplexity = outputs.get("perplexity")
        elif isinstance(outputs, tuple) and len(outputs) >= 2:
            reconstruction = outputs[0]
            vq_loss = outputs[1] if len(outputs) > 1 else None
            perplexity = outputs[2] if len(outputs) > 2 else None
        else:
            # Standard tensor output - just reconstruction
            reconstruction = outputs
            vq_loss = None
            perplexity = None

        # Get VQ-specific losses from model if available
        # Phase 4b-1: Use precision manager to ensure VQ loss stability
        if vq_loss is not None:
            vq_loss = self.precision_manager.validate_vq_loss_precision(vq_loss)

        if perplexity is not None and perplexity.dtype != torch.float32:
            perplexity = perplexity.float()

        # [FIX] Use UnifiedVQVAELossComputer with losses_dict
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}
        elif hasattr(self, "context") and self.context and hasattr(self.context, "loss_fn"):
            env_losses = self.env.losses if self.env else {}

        loss_output = self.loss_computer.compute(
            pred=reconstruction,
            target=target_batch,
            epoch=epoch,
            vq_loss=vq_loss,
            perplexity=perplexity,
            losses_dict=env_losses,
        )

        total_loss = loss_output.total
        components = loss_output.components

        # Build losses dict
        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(components)
        self._loss_dict_reuse["g_total_loss"] = total_loss

        if "loss" not in self._loss_dict_reuse:
            self._loss_dict_reuse["loss"] = total_loss

        # [ENHANCEMENT] Compute training metrics (PSNR, SSIM, MAE) for monitoring.
        # Live iteration (loop_state seam), not the frozen ``self.env.step``
        # (=0) — restores the train-metric interval throttle (pitfall #16).
        iteration = resolve_loop_iteration(self)
        train_metrics = self._compute_training_metrics(
            pred=reconstruction,
            target=target_batch,
            config=self.config,
            current_step=iteration,
        )

        # Phase 5: Ensure strict type compliance (dict[str, Tensor])
        for k, v in train_metrics.items():
            if not isinstance(v, torch.Tensor):
                self._loss_dict_reuse[k] = torch.tensor(v, device=reconstruction.device)
            else:
                self._loss_dict_reuse[k] = v

        return self._loss_dict_reuse

    @accepts_step_io
    def validation_step(
        self,
        batch: Any,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for VQ-VAE.

        Args:
            batch: Input batch.
            input_batch: Low-resolution input tensor.
            target_batch: High-resolution target tensor.
            epoch: Current epoch.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of validation metrics.
        """
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)
        # Use absolute import to avoid resolution errors

        with torch.no_grad():
            # VQ-VAE validation uses reconstruction error
            vqvae_output = self.env.generator(input_batch)

            # Handle different output formats
            if isinstance(vqvae_output, tuple) and len(vqvae_output) >= 2:
                # VQ-VAE returns tuple: (reconstruction, vq_loss, ...)
                hr_fakes = vqvae_output[0]
                # CRITICAL FIX: Ensure vq_loss has requires_grad=True for backward
                vq_loss = (
                    vqvae_output[1]
                    if len(vqvae_output) > 1
                    else torch.tensor(0.0, device=hr_fakes.device, requires_grad=True)
                )
            elif isinstance(vqvae_output, dict):
                # Handle dict output
                hr_fakes = vqvae_output.get(
                    "reconstruction", vqvae_output.get("output", input_batch)
                )
                # CRITICAL FIX: Ensure vq_loss has requires_grad=True for backward
                vq_loss = vqvae_output.get("vq_loss")
                if vq_loss is None:
                    vq_loss = torch.tensor(0.0, device=hr_fakes.device, requires_grad=True)
                elif not vq_loss.requires_grad:
                    vq_loss = vq_loss.detach().requires_grad_(True)
            else:
                # Non-VQVAE model (e.g., UNet) - just use reconstruction
                hr_fakes = vqvae_output
                # CRITICAL FIX: Ensure vq_loss has requires_grad=True for backward
                vq_loss = torch.tensor(0.0, device=vqvae_output.device, requires_grad=True)

            # Ensure tensors are on the same device
            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            # Standardized Metric Computation
            val_config = self.config.validation if hasattr(self.config, "validation") else None
            compute_img_metrics = val_config.scoring.enable_image_metrics if val_config else True

            metrics = {}
            if compute_img_metrics:
                # [Refactor] Use SSOT ValidationMetricsComputer
                try:
                    computer = self._get_validation_metrics_computer(self.config)
                    computed = computer.compute(hr_fakes, target_batch)

                    # Apply prefix
                    for k, v in computed.items():
                        metrics[f"val_{k}"] = v
                except Exception as e:
                    if hasattr(self, "logging_service") and logger.isEnabledFor(logging.WARNING):
                        self.logging_service.log_warning("Validation metrics failed: %s", str(e))

            # Add VQ-VAE specific metrics
            if vq_loss is not None:
                # Use detach() to keep as tensor for deferred logging
                if isinstance(vq_loss, torch.Tensor):
                    metrics["vq_loss"] = vq_loss.detach()
                else:
                    metrics["vq_loss"] = vq_loss

            return self._convert_metrics_to_floats(metrics)
