"""DomainAdaptation Training Strategy Module

This module contains DomainAdaptation training strategies.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.models.losses.computers import UnifiedGANLossComputer

from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class DomainAdaptationTrainingStrategy(BaseTrainingStrategy):
    """Domain adaptation training for cross-field-strength MRI synthesis.

    Transfers learned features from source field strength (e.g., 3T high-field)
    to target field strength (e.g., 0.05T ultra-low-field) with minimal target
    supervision via adversarial domain alignment.

    ## Domain Adaptation Objectives

    1. **Source Task Loss**: Supervised learning on labeled source data
       - Generator trained for source-domain reconstruction
       - Establishes strong feature representations in source domain

    2. **Adversarial Alignment**: Domain discriminator forces distribution matching
       - Minimizes domain-specific feature differences
       - Encourages learning of field-strength-invariant features
       - Uses DANN-style gradient reversal

    3. **Target Reconstruction**: Unsupervised adaptation on target field data
       - Reconstruction consistency in target domain
       - Cycle consistency loss (if unpaired data available)

    4. **Distribution Matching**: Optional MMD or CORAL loss
       - Aligns feature distributions across ultrasound domains
       - Reduces systematic domain shift automatically

    ## Training Phases

    **Phase 1: Pre-training** (Optional)
    - Train on source domain (3T) with full supervision
    - Generator learns source-specific features and reconstruction quality
    - Discriminator learns to classify source vs. target domain

    **Phase 2: Adaptation** (Main)
    - Freeze shared encoder, fine-tune decoder on target (0.05T)
    - Adversarial loss on feature representations (gradient reversal)
    - Cycle consistency for unpaired/weakly-paired data
    - Domain discriminator loss guides alignment

    ## Configuration

    - `training.training_mode`: 'gan' or 'domain_adaptation'
    - `training.domain_adaptation.source_domain`: Source field strength (e.g., '3T')
    - `training.domain_adaptation.target_domain`: Target field strength (e.g., '0.05T')
    - `training.domain_adaptation.adaptation_weight`: λ_domain weighting adversarial loss
    - `optimization.optimizer.learning_rate`: Generator LR (domain discriminator uses 0.1x)

    ## Loss Components

    - **Reconstruction**: L1 between prediction and available targets
    - **Adversarial (Generator)**: Fool domain discriminator (minimize)
    - **Adversarial (Discriminator)**: Classify source vs. target domains
    - **Cycle Consistency**: MSE(x, decode(encode(x))) for unpaired adaptation
    - **Feature Alignment**: Optional MMD or CORAL distribution matching

    ## Output Features

    - **Unified Representations**: Generator produces field-agnostic features
    - **Domain-Invariant Encodings**: Encoder output usable in both source/target
    - **Generalization**: Decoder learns to work with diverse feature distributions

    Attributes:
        state: TrainingState with source/target data access
        loss_computer: UnifiedGANLossComputer for multi-term loss aggregation
        domain_discriminator: Adversarial domain classifier (source vs. target)
        domain_optimizer: Optimizer for discriminator (separate LR)
        domain_loss_fn: BCEWithLogitsLoss for binary domain classification
        lambda_domain: Weight factor for adversarial term (default 1.0)
        device: Computation device (CUDA/CPU)
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

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedGANLossComputer(config=self.config, device=self.device)
        self._setup_strategy_specific_components()

        # Set device from context
        self.device = self.device

        # Domain adaptation specific components
        self.domain_discriminator = None
        self.domain_optimizer = None
        self.domain_loss_fn = None
        # SSOT: read domain adversarial weight from config.losses.gan (GANLossesConfig)
        self.lambda_domain = self.config.losses.gan.lambda_domain

    def _setup_strategy_specific_components(self) -> None:
        """Initialize domain adaptation-specific components and perform
        validation."""
        self._verify_strategy_config(expected_modes=("gan", "domain_adaptation"))
        self._log_config_features(self.logging_service)

    def initialize_domain_adaptation(
        self,
        domain_discriminator,
        domain_optimizer,
        lambda_domain=1.0,
    ):
        """Initialize domain adaptation components."""
        self.domain_discriminator = domain_discriminator
        self.domain_optimizer = domain_optimizer
        self.domain_loss_fn = nn.BCEWithLogitsLoss()
        self.lambda_domain = lambda_domain

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for domain adaptation training step.

        Phase 4b-2: Enhanced with LossResult infrastructure for metrics tracking.
        Adversarial domain alignment with DANN-style gradient reversal.

        Args:
            input_batch: Low-resolution input tensor batch.
            target_batch: High-resolution target tensor batch.
            epoch: Current epoch index.
            **kwargs: Additional optional context.

        Returns:
            Dictionary of loss tensors for the template method to wrap with AMP.
        """
        epoch = kwargs.get("epoch", epoch)

        # Move data to device
        lr_reals = input_batch.to(self.device, non_blocking=True)
        hr_reals = target_batch.to(self.device, non_blocking=True)

        # Generate fake images
        hr_fakes = self.env.generator(lr_reals)

        # === Generator and Discriminator Losses ===
        # [FIX] Use UnifiedGANLossComputer (SSOT)
        env_losses = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        # ``iteration`` is the live loop seam, not ``self.env.step`` and not a
        # bare ``kwargs.get`` (pitfall #16, #1937). Omitting it here defaulted
        # ``compute``'s parameter to 0, which held every warm-up gate shut for
        # the whole run -- silently, because a gated loss is absent from
        # ``components`` rather than scaled to zero (#1950).
        loss_output = self.loss_computer.compute(
            pred=hr_fakes,
            target=hr_reals,
            epoch=epoch,
            iteration=resolve_loop_iteration(self),
            discriminator=self.domain_discriminator,
            losses_dict=env_losses,
        )

        gen_loss = loss_output.total
        components = loss_output.components

        # === Domain Adaptation Losses ===
        domain_loss_val = torch.tensor(0.0, device=self.device)
        if self.domain_discriminator is not None:
            # Domain discriminator loss
            domain_loss_val = self.compute_domain_loss(hr_fakes, hr_reals)
            gen_loss = gen_loss + self.lambda_domain * domain_loss_val

        # Build losses dict
        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(components)
        self._loss_dict_reuse.update(
            {
                "g_total_loss": gen_loss,
                "domain_loss": domain_loss_val,
            }
        )

        return self._loss_dict_reuse

    def unpack_balancer_batch(self, item: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Unpack a :class:`Balancer`-emitted item into (input, target).

        Phase 7 adapter (~Amendment G follow-up). The site balancer emits:

        - round_robin: ``{"domain": str, "batch": <loader-batch>}``
        - stratified / concat: ``{<domain_name>: <loader-batch>, ...}``

        For DA, the contract is: the source-domain batch carries paired
        (input, target); target-domain batches carry input only. This
        helper returns the first paired batch found among the dict entries.

        Single-loader (non-Balancer) batches pass through unchanged.

        Returns:
            ``(input, target)`` from the source-domain batch when both
            are present; ``(target_input, None)`` when only the target
            domain batch is present (unsupervised adaptation step).
        """
        # Pass-through for non-Balancer batches.
        if not isinstance(item, dict):
            return self._unpack_batch(item)

        # Round-robin shape: {"domain": "src"|"tgt", "batch": ...}
        if set(item.keys()) == {"domain", "batch"}:
            inner_in, inner_tgt = self._unpack_batch(item["batch"])
            return inner_in, inner_tgt

        # Stratified/concat: dict keyed by domain name. Prefer 'source'
        # if present; otherwise take the first paired key. Fall back to
        # input-only when no domain has a target.
        domain_names = [n for n in ("source", "src", "labeled") if n in item] + [
            k for k in item.keys() if k not in ("source", "src", "labeled")
        ]
        for name in domain_names:
            inner_in, inner_tgt = self._unpack_batch(item[name])
            if inner_in is not None and inner_tgt is not None:
                return inner_in, inner_tgt
        # No paired batch — return first input we can find.
        for v in item.values():
            inner_in, _ = self._unpack_batch(v)
            if inner_in is not None:
                return inner_in, None
        return None, None

    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Optimized training step using closures for Trainer integration."""
        if input_batch is None or target_batch is None:
            # Phase 7: when the trainer is fed by a multi-domain Balancer,
            # batch is a domain-tagged dict — route through the adapter.
            if isinstance(batch, dict) and (
                "domain" in batch or any(isinstance(v, (dict, list, tuple)) for v in batch.values())
            ):
                input_batch, target_batch = self.unpack_balancer_batch(batch)
            else:
                input_batch, target_batch = self._unpack_batch(batch)

        if input_batch is not None:
            input_batch = self._to_device(input_batch)
        if target_batch is not None:
            target_batch = self._to_device(target_batch)

        iteration = kwargs.get("iteration", 0)

        # Generator Closure
        def g_closure() -> torch.Tensor:
            """g_closure.

            Returns:
                torch.Tensor: Description.
            """
            # #1190: this closure calls ``_compute_losses_impl`` DIRECTLY, so it
            # bypasses the ``_compute_losses`` wrapper that normally emits the
            # ``model_output`` debug snapshot. Arm it here instead. The module is
            # ``env.generator`` -- the same one the step config below declares --
            # named explicitly rather than defaulted, so the site states which
            # module its snapshot describes.
            with self._capture_model_output(
                module=self.env.generator,
                input_batch=input_batch,
                target_batch=target_batch,
                step=iteration,
            ):
                losses = self._compute_losses_impl(input_batch, target_batch, epoch, **kwargs)
            total_loss = losses["g_total_loss"]

            # Store metrics for reporting as DETACHED tensors (no host transfer).
            # The old per-key ``v.item()`` fired one blocking GPU→CPU sync per loss
            # key EVERY step; the reporting path converts to float once, later and
            # throttled (base ``_last_step_metrics`` float() boundary), exactly as
            # the adversarial / cycle-Bloch closures already do.
            self._last_step_metrics = {
                k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in losses.items()
            }
            return total_loss

        step_configs = [
            {
                "name": "generator",
                "closure": g_closure,
                "optimizer": self.env.opt_g,
                "model": self.env.generator,
            }
        ]

        # Domain Discriminator Closure (Optional)
        if self.domain_discriminator is not None and self.domain_optimizer is not None:

            def d_closure() -> torch.Tensor:
                # We need hr_fakes from generator without gradients
                """d_closure.

                Returns:
                    torch.Tensor: Description.
                """
                with torch.no_grad():
                    hr_fakes = self.env.generator(input_batch)

                domain_loss = self.compute_domain_loss(hr_fakes, target_batch)

                # Detached tensor, not .item(): avoid a per-step GPU sync in the
                # discriminator closure (converted to float in the reporting path).
                self._last_step_metrics["domain_loss_disc"] = domain_loss.detach()
                return domain_loss

            step_configs.append(
                {
                    "name": "domain_discriminator",
                    "closure": d_closure,
                    "optimizer": self.domain_optimizer,
                    "model": self.domain_discriminator,
                }
            )

        return step_configs

    def compute_domain_loss(
        self,
        hr_fakes: torch.Tensor,
        hr_reals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain adaptation loss."""
        if self.domain_discriminator is None:
            return torch.tensor(0.0, device=self.device)

        # Extract features from discriminator layers for domain classification
        # This assumes the discriminator has intermediate feature extraction
        with torch.no_grad():
            discriminator = self.discriminator_model
            if discriminator and hasattr(
                discriminator,
                "extract_features",
            ):
                real_features = discriminator.extract_features(hr_reals)
                fake_features = discriminator.extract_features(hr_fakes)
            else:
                # Fallback: use discriminator outputs directly
                real_features = discriminator(hr_reals) if discriminator else hr_reals
                fake_features = discriminator(hr_fakes) if discriminator else hr_fakes

        # Domain labels: 0 for source (3T), 1 for target (ULF)
        # In practice, this would be determined by the dataset
        batch_size = hr_reals.shape[0]
        source_labels = torch.zeros(batch_size, 1, device=self.device)
        target_labels = torch.ones(batch_size, 1, device=self.device)

        # Domain discriminator predictions
        source_domain_pred = self.domain_discriminator(real_features.detach())
        target_domain_pred = self.domain_discriminator(fake_features.detach())

        # Domain loss
        if self.domain_loss_fn is None:
            self.domain_loss_fn = nn.BCEWithLogitsLoss()
        source_loss = self.domain_loss_fn(source_domain_pred, source_labels)
        target_loss = self.domain_loss_fn(target_domain_pred, target_labels)

        return (source_loss + target_loss) / 2

    def compute_metrics(
        self,
        hr_fakes: torch.Tensor,
        hr_reals: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics."""
        # [Refactor] Use SSOT ValidationMetricsComputer
        try:
            computer = self._get_validation_metrics_computer(self.config)

            with torch.no_grad():
                if hr_fakes.dim() == 5 and hr_reals.dim() == 5:
                    # 3D evaluation - flatten spatial dimensions
                    # (B, C, H, W, D) -> (B, C, H*W*D) or similar for metrics?
                    # Computer usually expects (B, C, H, W).
                    # If we flatten to (B, C, -1), it becomes (B, C, N).
                    # Standard metrics might expect image-like shapes.
                    # Best effort: flatten 3D to large 2D-like strip or just pass as is if computer supports it?
                    # Legacy code flattened: view(B, C, -1).
                    # If we pass (B, C, L), metrics need to handle 1D spatial.
                    generated_flat = hr_fakes.view(hr_fakes.shape[0], hr_fakes.shape[1], -1)
                    target_flat = hr_reals.view(hr_reals.shape[0], hr_reals.shape[1], -1)
                    # We assume computer's underlying metrics can handle this shape (or legacy did)
                    # Actually standard torchmetrics usually handle (B, C, ...) generic.
                    metrics = computer.compute(generated_flat, target_flat)
                else:
                    # Standard 2D evaluation
                    metrics = computer.compute(hr_fakes, hr_reals)

            return metrics

        except Exception as e:
            if hasattr(self, "logging_service") and logger.isEnabledFor(logging.WARNING):
                self.logging_service.log_warning("Validation metrics failed: %s", str(e))
            return {}

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Performs validation step with evaluation metrics."""
        batch = (input_batch, target_batch)
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        lr_reals = input_batch
        hr_reals = target_batch

        self.env.generator.eval()
        # Generate fake images
        hr_fakes = self.env.generator(lr_reals)

        # Ensure tensors are on the same device
        if hr_fakes.device != hr_reals.device:
            hr_fakes = hr_fakes.to(hr_reals.device)

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
            # Use existing helper which uses EvaluationMetrics but lacks prefix
            raw_metrics = self.compute_metrics(hr_fakes, hr_reals)
            # Apply prefix
            for k, v in raw_metrics.items():
                metrics[f"val_{k}"] = v

        # Add domain-specific metrics if domain discriminator available
        if self.domain_discriminator is not None:
            domain_loss = self.compute_domain_loss(hr_fakes, hr_reals)
            metrics["domain_loss"] = float(domain_loss.detach())

        return metrics


__all__ = ["DomainAdaptationTrainingStrategy"]
