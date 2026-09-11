"""GAN Training Strategy Module

This module contains the GANTrainingStrategy for adversarial training.
"""

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.optimization_utils import AsyncMetricsReporter
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
    _resolve_disc_updates,
    assemble_adversarial_step_configs,
)
from spectramr.infrastructure.training.strategy_helpers import (
    StrategyInitializationHelper,
)

from ..utils.training_utils import clamp_to_range
from .base import BaseTrainingStrategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GANTrainingStrategy(BaseTrainingStrategy, AdversarialMixin):
    """GAN training strategy with dynamic balancing and performance safeguards.

    [FORENSIC FIX] This strategy implements a full GAN training loop with:
    - Discriminator training (real vs. fake classification)
    - Generator training (fool discriminator + reconstruction)
    - Dynamic loss balancing to prevent mode collapse
    - Gradient penalty (optional) for Lipschitz constraint
    - Feature matching loss (optional) for perceptual quality

    Training Procedure:
    1. **Discriminator Step**:
       - Sample real data from batch
       - Generate fake data via generator
       - Compute adversarial loss (BCE, hinge, or WGAN)
       - Apply gradient penalty if enabled
       - Update discriminator weights

    2. **Generator Step**:
       - Generate fake data
       - Compute adversarial loss (fool discriminator)
       - Compute reconstruction loss (L1/L2 vs. target)
       - Optionally add perceptual/feature matching
       - Update generator weights

    Loss Components:
    - **D_real**: Discriminator loss on real samples
    - **D_fake**: Discriminator loss on fake samples
    - **G_adv**: Generator adversarial loss (fool discriminator)
    - **G_recon**: Generator reconstruction loss (L1/L2)
    - **G_perceptual**: Perceptual/feature matching (optional)
    - **GP**: Gradient penalty (WGAN-GP)

    Attributes:
        config_snapshot: Cached config for fast access.
        discriminator_model: Discriminator network (nn.Module).
        generator_model: Generator network (nn.Module).
        opt_d: Discriminator optimizer.
        opt_g: Generator optimizer.
        device: PyTorch device.

    Config Requirements:
        ```yaml
        training:
          training_mode: gan
        objectives:
          adversarial:
            lambda_adv_g: 0.01       # Generator adversarial weight
            lambda_adv_d: 1.0        # Discriminator weight
          reconstruction:
            lambda_l1: 1.0           # Reconstruction weight
        model:
          discriminator_type: "patch_gan"  # Discriminator architecture
        ```

    Raises:
        ValueError: If training_mode is not 'gan' or discriminator is missing.
        RuntimeError: If mode collapse detected (discriminator loss ~0).

    Example:
        >>> from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
        >>> strategy = GANTrainingStrategy(env=training_env)
        >>> losses = strategy.train_step(batch, epoch=0, step=100)
        >>>
        >>> # Monitor GAN training stability
        >>> d_loss = losses['d_loss'].item()
        >>> g_loss = losses['generator_total'].item()
        >>> print(f"D: {d_loss:.4f}, G: {g_loss:.4f}")

    Note:
        GAN training is inherently unstable. Use:
        - Spectral normalization in discriminator
        - Two-timescale update rule (TTUR)
        - Gradient penalty for stabilization
        - Early stopping if discriminator dominates
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

        # Get validated config snapshot to reduce direct attribute access
        config = self.env.config if self.env else self.config

        if hasattr(config, "get_validated_snapshot"):
            self.config_snapshot = config.get_validated_snapshot()
        else:
            if isinstance(config, dict):
                self.config_snapshot = config
            elif hasattr(config, "model_dump"):
                self.config_snapshot = config.model_dump()
            else:
                self.config_snapshot = config.__dict__ if hasattr(config, "__dict__") else {}

        self.async_metrics_reporter = AsyncMetricsReporter()
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize GAN-specific components and perform validation."""
        self.setup_adversarial(expected_modes=("gan",))
        # Step counter used by ``train_step`` for D/G alternation cadence.
        # Without this, the first training step raises AttributeError. See
        # findings booklet 2026-05-05 ST-1.
        self._step_counter = 0

    def _profile(self, name: str, **metadata: Any):
        """Helper method to create profiling context manager."""

        return StrategyInitializationHelper.create_profiling_context(self, name, **metadata)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute GAN losses for logging and monitoring (NO BACKWARD).

        In the GAN strategy, the actual optimization happens in `train_step` which
        updates the Discriminator and Generator separately. This method is primarily
        used to aggregate losses for the final metrics report.

        Args:
            input_batch: Low-resolution input tensor batch (or other model input).
            target_batch: High-resolution ground truth tensor batch.
            epoch: The current epoch index.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing:
            - `g_total_loss`: The total generator loss (for reporting).
            - `adversarial`: Adversarial component of generator loss.
            - `l1`: Reconstruction component (if configured).
            - Other configured loss components.
        """

        from spectramr.infrastructure.training.utils.training_utils import clamp_to_range

        # Ensure generator is a Module
        generator = self.env.generator
        if not isinstance(generator, nn.Module):
            # Fallback or error if not callable/module
            # Mypy needs assurance. If it's not a Module, we can't call it easily.
            # Assuming it's a Module for now as per BaseTrainingStrategy assumptions.
            pass

        # Forward pass
        hr_fakes = self.generator_model(input_batch)
        if hr_fakes.device != target_batch.device:
            hr_fakes = hr_fakes.to(target_batch.device)

        if getattr(self.config.training, "enforce_output_range", False):
            hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

        components = {}
        total_loss = None

        env_losses: dict[str, Any] = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}
        elif hasattr(self, "context") and self.context and hasattr(self.context, "loss_fn"):
            env_losses = self.env.losses if self.env else {}

        # Add adversarial loss if discriminator is available
        if self.discriminator_model:
            # Reconstruct generator inputs (for loss computer)
            # Forward pass happens in compute if pred not supplied, or we just pass pred
            pass

        loss_output = self.loss_computer.compute(
            pred=hr_fakes,
            target=target_batch,
            epoch=epoch,
            discriminator=self.discriminator_model,
            losses_dict=env_losses,
        )

        total_loss = loss_output.total
        components = loss_output.components

        if total_loss is None:
            total_loss = torch.tensor(0.0, device=target_batch.device, requires_grad=True)

        return {"g_total_loss": total_loss, **components}

    def _train_discriminator_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, Any] | None = None,
    ) -> Any:
        """Execute one discriminator update step.

        Responsibility: Single discriminator optimization with AMP context,
        loss computation (with fallback logic), backward pass, gradient clipping,
        and optimizer step.

        Args:
            input_batch: Input conditioning tensor.
            target_batch: Target (real) image tensor.
            discriminator: Discriminator module (casted).
            epoch: Current epoch for loss scheduling.
            iteration: Current iteration for loss scheduling.
            losses_dict: Optional losses dictionary.

        Returns:
            A closure function that returns the computed `d_total_loss`.
        """

        def d_closure() -> torch.Tensor:
            # Generate fake images (detached from graph)
            """d_closure.

            Returns:
                torch.Tensor: Description.
            """
            # The discriminator trains on its own step: re-enable grad on its
            # parameters (the generator step below freezes them). Toggling at
            # closure entry — not after — keeps the state correct through the
            # trainer's out-of-closure backward.
            if isinstance(discriminator, nn.Module):
                discriminator.requires_grad_(True)

            with torch.no_grad():
                hr_fakes = self.env.generator(input_batch)

            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            # [FIX] Complex->Real guard for generated outputs before passing to D
            if torch.is_complex(hr_fakes):
                hr_fakes = torch.cat([hr_fakes.real, hr_fakes.imag], dim=1)

            # Called unguarded: ``setup_adversarial`` installs
            # ``UnifiedGANLossComputer`` unconditionally, so the deleted
            # ``hasattr``/``is None`` fallback could only route an unexpected
            # computer down a second critic feed that disagreed with the other
            # copy. A computer lacking the method must fail loudly (NN3).
            d_loss_output = self.loss_computer.compute_discriminator_loss(
                real=target_batch,
                fake=hr_fakes,
                discriminator=discriminator,
                epoch=epoch,
                iteration=iteration,
            )

            d_total = d_loss_output.total if hasattr(d_loss_output, "total") else d_loss_output

            # Store detached metrics as TENSORS (no host transfer). float() here
            # fired a GPU sync per key every step; get_last_metrics() converts to
            # float later, once and throttled.
            with torch.no_grad():
                if d_total is not None:
                    self._last_step_metrics["d_total_loss"] = d_total.detach()
                if hasattr(d_loss_output, "components"):
                    for k, v in d_loss_output.components.items():
                        self._last_step_metrics[f"d_{k}"] = v.detach()

            return d_total

        return d_closure

    def _train_generator_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, Any] | None = None,
    ) -> Any:
        """Execute one generator update step.

        Responsibility: Single generator optimization with AMP context,
        optional output range enforcement, loss computation (with fallback logic),
        backward pass, gradient clipping, and optimizer step.

        Args:
            input_batch: Input conditioning tensor.
            target_batch: Target (real) image tensor.
            discriminator: Discriminator module (casted).
            epoch: Current epoch for loss scheduling.
            iteration: Current iteration for loss scheduling.
            losses_dict: Optional losses dictionary.

        Returns:
            A closure function that returns the computed `g_total_loss`.
        """

        def g_closure() -> torch.Tensor:
            # Generate fake images (attached to graph for gradients)
            """g_closure.

            Returns:
                torch.Tensor: Description.
            """
            # Freeze the discriminator for the generator step. Gradients still
            # flow to the generator THROUGH D's activations (that needs only the
            # input to require grad, not D's params), but D.grad is left
            # untouched — preventing the "G trains D" leak that, under gradient
            # accumulation, survives to D's optimizer.step(). Also saves a full
            # D backward every step. The D step re-enables grad at its entry.
            if isinstance(discriminator, nn.Module):
                discriminator.requires_grad_(False)

            hr_fakes = self.env.generator(input_batch)

            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            if getattr(self.config.training, "enforce_output_range", False):
                hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

            # Called unguarded: ``setup_adversarial`` installs
            # ``UnifiedGANLossComputer`` unconditionally, so the deleted
            # ``hasattr``/``is None`` fallback could only route an unexpected
            # computer down a second critic feed -- realified here, raw in
            # ``AdversarialMixin`` -- or, with no critic, silently swap the whole
            # adversarial objective for plain L1. Both must now fail loudly (NN3).
            g_loss_output = self.loss_computer.compute_generator_loss(
                pred=hr_fakes,
                target=target_batch,
                discriminator=discriminator,
                epoch=epoch,
                iteration=iteration,
            )
            g_total = g_loss_output.total if hasattr(g_loss_output, "total") else g_loss_output

            # Store detached metrics as TENSORS (no host transfer). float() here
            # fired a GPU sync per key every step; get_last_metrics() converts to
            # float later, once and throttled.
            with torch.no_grad():
                if g_total is not None:
                    self._last_step_metrics["g_total_loss"] = g_total.detach()
                if hasattr(g_loss_output, "components"):
                    for k, v in g_loss_output.components.items():
                        self._last_step_metrics[
                            f"g_{k}" if not str(k).startswith("loss_") else k
                        ] = v.detach()

                if hasattr(self, "_compute_training_metrics"):
                    # Live iteration (loop_state seam), not the frozen
                    # ``self.env.step`` (=0) — restores the train-metric
                    # interval throttle (pitfall #16).
                    current_step = resolve_loop_iteration(self)
                    train_metrics = self._compute_training_metrics(
                        pred=hr_fakes,
                        target=target_batch,
                        config=self.config,
                        current_step=current_step,
                    )
                    self._last_step_metrics.update(
                        {
                            k: float(v) if isinstance(v, torch.Tensor) else v
                            for k, v in train_metrics.items()
                        }
                    )

            return g_total

        return g_closure

    @accepts_step_io
    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        iteration: int = 0,
        batch_data: Any = None,
        scaler: Any = None,  # [AMP] Accept scaler
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute a full GAN training step (Discriminator + Generator).

        Clean orchestrator implementing alternating optimization schedule.
        Delegates to focused helpers for discriminator and generator training.

        Key operations:
        1. **Batch unpacking** → validate inputs
        2. **Discriminator loop** → Run n_critic updates via _train_discriminator_step
        3. **Generator step** → Run single update via _train_generator_step
        4. **Metrics aggregation** → Collect and return loss scalars

        Args:
            batch: Raw batch data from dataloader.
            epoch: Current epoch index.
            input_batch: Optional pre-unpacked input tensor.
            target_batch: Optional pre-unpacked target tensor.
            **kwargs: Additional context (iteration, step).

        Returns:
            Dictionary of aggregated metrics (d_total_loss, g_total_loss, training metrics).
        """
        self._step_counter += 1

        # Step 1: Unpack batch if not provided
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        if input_batch is not None:
            input_batch = self._to_device(input_batch)
        if target_batch is not None:
            target_batch = self._to_device(target_batch)

        # [FIX] Complex→Real guard: GAN train_step bypasses base training_step,
        # so we must convert complex tensors here to prevent CUDA kernel errors
        # (c10::complex<float> vs float bias type mismatch in nn.Conv2d).
        if input_batch is not None and torch.is_complex(input_batch):
            input_batch = torch.cat([input_batch.real, input_batch.imag], dim=1)
        if target_batch is not None and torch.is_complex(target_batch):
            target_batch = torch.cat([target_batch.real, target_batch.imag], dim=1)

        # Live global iteration from the loop_state seam — NOT a kwargs lookup.
        # The training loop passes ``iteration=`` as a keyword, which binds to
        # this method's explicit ``iteration`` parameter (above), leaving
        # ``kwargs`` empty; the old ``kwargs.get("iteration", ...)`` therefore
        # clobbered the real value back to 0, silently degrading the R1-reg
        # cadence (``_should_apply_r1``) from every-r1_interval-iterations to
        # every step of every r1_interval-th epoch (review 2026-07-01). This is
        # the same seam already used for ``current_step`` below.
        iteration = resolve_loop_iteration(self)
        config = self.env.config if self.env else self.config
        num_d_updates = _resolve_disc_updates(config)

        # Validate discriminator availability
        discriminator = self.env.discriminator
        if discriminator is None:
            raise RuntimeError("Discriminator is required for GANTrainingStrategy")
        discriminator = cast(nn.Module, discriminator)

        # State block used for metric reporting
        self._last_step_metrics = {}

        # Steps 2 + 3: N critic updates, then one generator update. The assembly
        # lives in ``AdversarialMixin`` so this cadence has ONE owner -- it was
        # written out here AND in ``train_step_adversarial``, through different
        # accessors, and only this copy runs (non-negotiable 17).
        step_configs = assemble_adversarial_step_configs(
            num_d_updates=num_d_updates,
            d_closure_factory=lambda: self._train_discriminator_step(
                input_batch,
                target_batch,
                discriminator,
                epoch,
                iteration,
                self.env.losses,
            ),
            g_closure=self._train_generator_step(
                input_batch, target_batch, discriminator, epoch, iteration, self.env.losses
            ),
            discriminator=discriminator,
            generator=self.env.generator,
            opt_d=self.env.opt_d,
            opt_g=self.env.opt_g,
        )

        return step_configs

    def get_last_metrics(self) -> dict[str, Any]:
        """Return the detached component metrics, ON-DEVICE (#707).

        See :meth:`BaseTrainingStrategy.get_last_metrics`. The comments at
        ``:324`` and ``:435`` in this file say the metrics are kept on-device to
        avoid a per-step sync and name this method as where conversion happens --
        but `training_loop` calls it every iteration, so converting here defeated
        exactly the optimization those comments describe.
        """
        return dict(getattr(self, "_last_step_metrics", {}))

    def _backward_and_step(
        self,
        losses: dict[str, torch.Tensor],
        epoch: int,
        step: int = 0,
    ) -> None:
        """
        Override to skip - custom train_step handles backward.
        """
        pass

    def _report_metrics(self, losses: dict[str, torch.Tensor], epoch: int) -> None:
        """Report metrics asynchronously."""
        try:
            metrics_dict = {
                key: val.detach() if isinstance(val, torch.Tensor) else val
                for key, val in losses.items()
            }
            if hasattr(self, "async_metrics_reporter"):
                self.async_metrics_reporter.report_async(metrics_dict)
        except Exception as e:
            if self.logging_service:
                self.logging_service.log_warning(
                    f"Failed to report async metrics: {e!s}",
                    model_type=self.config.model.model_type,
                )

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Perform a single validation step for GANs.

        Generates fake images, computes reconstruction metrics (PSNR, SSIM, etc.),
        and optionally computes discriminator scores to track convergence.

        Args:
            batch: The raw batch data.
            input_batch: Optional pre-unpacked input.
            target_batch: Optional pre-unpacked target.
            **kwargs: Additional context.

        Returns:
            A dictionary of validation metrics (e.g., `val_psnr`, `val_real_score`).
        """
        batch = (input_batch, target_batch)
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)
            if input_batch is None or target_batch is None:
                return {}

        # Fail gracefully if unpacking failed
        if input_batch is None or target_batch is None:
            return {}

        if input_batch is None or target_batch is None:
            return {}

        # [FIX] Complex→Real guard for validation
        if torch.is_complex(input_batch):
            input_batch = torch.cat([input_batch.real, input_batch.imag], dim=1)
        if torch.is_complex(target_batch):
            target_batch = torch.cat([target_batch.real, target_batch.imag], dim=1)

        with torch.no_grad():
            self.generator_model.eval()

            # Single inference pass
            hr_fakes = self.generator_model(input_batch)
            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            # Handle models that return tuples (e.g., DisentangledMRI)
            if isinstance(hr_fakes, tuple):
                hr_fakes = hr_fakes[0]

            if getattr(self.config.training, "enforce_output_range", False):
                hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

            # Reconstruction metrics
            val_config = self.config.validation if hasattr(self.config, "validation") else None
            compute_img_metrics = val_config.scoring.enable_image_metrics if val_config else True

            metrics = {}
            if compute_img_metrics:
                try:
                    output_transformed, target_transformed = self._apply_metric_transforms(
                        hr_fakes, target_batch, val_config
                    )

                    computed = self.validation_metrics_computer.compute(
                        output_transformed, target_transformed
                    )
                    metrics.update(computed)
                except Exception as e:
                    if hasattr(self, "logging_service") and self.logging_service:
                        self.logging_service.log_warning(f"Validation metrics failed: {e!s}")

            # Compute discriminator scores if available
            disc_scores = {}
            if self.env.discriminator:
                # Cast for callability
                disc = cast(nn.Module, self.env.discriminator)
                real_scores = disc(target_batch)
                fake_scores = disc(hr_fakes)
                disc_scores = {
                    "real_score": real_scores.mean().detach().item(),
                    "fake_score": fake_scores.mean().detach().item(),
                }

            # Combine metrics
            validation_results = {**metrics, **disc_scores}

            # Log images to TensorBoard (if configured)
            self._log_validation_images_to_tensorboard(
                predictions=hr_fakes,
                targets=target_batch,
                inputs=input_batch,
                metrics=validation_results,
            )

            return validation_results

    def _should_apply_r1_regularization(self, epoch: int | None, iteration: int = 0) -> bool:
        """Determine if R1 regularization should be applied this step.

        Uses iteration-based interval for iteration-based training.

        Args:
            epoch: Current epoch index (legacy, fallback)
            iteration: Current iteration (preferred for iteration-based training)

        Returns:
            Whether to apply R1 regularization
        """
        # Use iteration-based interval for iteration-based training
        try:
            if self.config.losses and self.config.losses.gan:
                interval = self.config.losses.gan.r1_interval
            else:
                interval = 16
        except AttributeError:
            # v6.0 default if not configured
            interval = 16

        if interval > 0:
            # Prefer iteration if available
            if iteration > 0:
                return (iteration % interval) == 0
            # Fallback to epoch if no iteration
            if epoch is not None:
                return (epoch % interval) == 0
            return False

        # Fallback to expected-value scaling based on probability
        try:
            if (
                self.config.losses
                and self.config.losses.gan
                and hasattr(self.config.losses.gan, "r1_probability")
            ):
                probability = self.config.losses.gan.r1_probability
            else:
                probability = 1.0
        except AttributeError:
            # v6.0 default if not configured
            probability = 1.0
        # [PERF FIX] Avoid GPU→CPU sync: torch.rand generates scalar without .item()
        should_apply = torch.rand(()) < probability  # Empty tuple creates 0-D tensor

        return should_apply
