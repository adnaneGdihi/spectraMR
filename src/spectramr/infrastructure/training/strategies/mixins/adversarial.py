"""
Adversarial Mixin Module

This module contains the AdversarialMixin for GAN-based training logic.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.optimization_utils import (
    AsyncMetricsReporter,
    OptimizationMetrics,
)
from spectramr.infrastructure.training.strategy_helpers import (
    StrategyInitializationHelper,
)
from spectramr.infrastructure.training.utils.training_utils import clamp_to_range
from spectramr.models.losses.computers import UnifiedGANLossComputer

if TYPE_CHECKING:
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


def _resolve_disc_updates(config: Any) -> int:
    """Return the discriminator-updates-per-G-step count.

    v6.0 SSOT path: ``config.losses.gan.disc_updates``. Reading a
    top-level ``config.disc_updates`` is the legacy flat-config pattern
    (CLAUDE.md pitfall #1) and silently falls back to 1 on a v6.0
    config because ``extra="forbid"`` rejects the unknown top-level
    field — see ``TODO/audit/05_strategies_core_mixins_builders.md`` F10.

    Raises:
        ValueError: if ``config.losses.gan`` is not configured. Any
            adversarial-mixin caller has a GAN strategy in MRO and must
            ship a populated ``losses.gan`` block.
    """
    _losses = getattr(config, "losses", None)
    gan_losses = getattr(_losses, "gan", None)
    if gan_losses is None:
        raise ValueError(
            "AdversarialMixin requires `config.losses.gan` to be set. "
            "Add a `losses.gan:` block to the YAML (the canonical home "
            "for `disc_updates`, `gan_loss_type`, R1 schedule, etc.)."
        )
    return int(gan_losses.disc_updates)


def assemble_adversarial_step_configs(
    *,
    num_d_updates: int,
    d_closure_factory: "Callable[[], Callable[[], torch.Tensor]]",
    g_closure: "Callable[[], torch.Tensor]",
    discriminator: "nn.Module",
    generator: "nn.Module",
    opt_d: Any,
    opt_g: Any,
) -> list[dict[str, Any]]:
    """N discriminator step-configs, then exactly one generator step-config.

    THE cadence, in one place. Two implementations of this assembly existed --
    ``AdversarialMixin.train_step_adversarial`` and
    ``GANTrainingStrategy.train_step`` -- reachable through different accessors
    (``self.state.opt_d`` vs ``self.env.opt_d``) and therefore easy to change in
    one and not the other. Only the GAN one runs in production, so a divergence
    would have surfaced as a wrong *training schedule*, not an error: N critic
    updates per generator update is the definition of the paradigm, and nothing
    downstream re-checks it (non-negotiable 17).

    ``d_closure_factory`` is called once per discriminator update rather than
    reused, because each step needs its own closure over a fresh forward pass.
    """
    step_configs: list[dict[str, Any]] = [
        {
            "optimizer": opt_d,
            "closure": d_closure_factory(),
            "model": discriminator,
            "name": "discriminator",
        }
        for _ in range(num_d_updates)
    ]
    step_configs.append(
        {
            "optimizer": opt_g,
            "closure": g_closure,
            "model": generator,
            "name": "generator",
        }
    )
    return step_configs


class AdversarialMixin:
    """Mixin for adversarial training (GAN) logic."""

    def setup_adversarial(
        self: "BaseTrainingStrategy", expected_modes: tuple[str, ...] = ("gan",)
    ) -> None:
        """Initialize adversarial components."""
        if hasattr(self, "_verify_strategy_config"):
            self._verify_strategy_config(expected_modes=expected_modes)

        if hasattr(self, "_log_config_features") and hasattr(self, "logging_service"):
            self._log_config_features(self.logging_service)

        config = self.env.config if hasattr(self, "env") and self.env else self.state.config
        device = self.device

        self.loss_computer = UnifiedGANLossComputer(
            config=config,
            device=device,
        )

        # Announce the objective from the loss-weight SSOT, not the ``enable_*`` /
        # ``lambda_*`` pairs (#1918, #1919). The prefix names the concrete class
        # because four strategies share this setup and their banners are otherwise
        # indistinguishable in one log.
        self._log_loss_objective(f"[{type(self).__name__}]")

        StrategyInitializationHelper.initialize_profiling_service(self, fallback_enabled=False)

        self.async_metrics_reporter = AsyncMetricsReporter(batch_size=10)
        self.optimization_metrics = OptimizationMetrics(name="gan_training")

        def metrics_callback(metrics: dict) -> None:
            """Async callback for reporting aggregated metrics."""
            # Use getattr to safely access logging_service
            logging_service = self.logging_service
            if logging_service is not None and logger.isEnabledFor(logging.DEBUG):
                for key, value in metrics.items():
                    logging_service.log_debug(
                        "Aggregated metric: %s=%.6f",
                        key,
                        value,
                        model_type=self.state.model_type,
                    )

        self.async_metrics_reporter.set_callback(metrics_callback)

        self._step_counter = 0

    def _check_discriminator_features(self: "BaseTrainingStrategy") -> bool:
        """Check if discriminator supports feature extraction."""
        try:
            disc_forward = self.discriminator_model.forward
            if disc_forward and hasattr(disc_forward, "__code__"):
                return "return_features" in disc_forward.__code__.co_varnames
        except (AttributeError, TypeError) as _exc:
            logger.debug("Suppressed exception: %s", _exc)
        return False

    def _should_apply_r1_regularization(
        self: "BaseTrainingStrategy", epoch: int | None, iteration: int = 0
    ) -> bool:
        """Determine if R1 regularization should be applied this step."""
        try:
            if self.config.losses and self.config.losses.gan:
                interval = self.config.losses.gan.r1_interval
            else:
                interval = 16
        except AttributeError:
            interval = 16

        if interval > 0:
            if iteration > 0:
                return (iteration % interval) == 0
            if epoch is not None:
                return (epoch % interval) == 0
            return False

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
            probability = 1.0

        should_apply = torch.rand(()) < probability
        return should_apply

    def train_step_adversarial(
        self: "BaseTrainingStrategy",
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Adversarial training step returning closures for the Trainer.
        """
        self._step_counter += 1

        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        if input_batch is not None:
            input_batch = self._to_device(input_batch)
        if target_batch is not None:
            target_batch = self._to_device(target_batch)

        iteration = kwargs.get("iteration", kwargs.get("step", 0))
        config = self.env.config if hasattr(self, "env") and self.env else self.state.config
        num_d_updates = _resolve_disc_updates(config)

        # State block used for metric reporting
        self._last_step_metrics = {}

        def make_d_closure():
            """make_d_closure.

            Returns:
                Any: Description.
            """

            def d_closure() -> torch.Tensor:
                """d_closure.

                Returns:
                    torch.Tensor: Description.
                """
                with torch.no_grad():
                    hr_fakes = self.generator_model(input_batch)

                if hr_fakes.device != target_batch.device:
                    hr_fakes = hr_fakes.to(target_batch.device)

                # Called unguarded: ``setup_adversarial`` installs
                # ``UnifiedGANLossComputer`` unconditionally, so the deleted
                # ``hasattr``/``is None`` fallback could only route an unexpected
                # computer down a second critic feed that disagreed with
                # ``gan.py``'s. A computer lacking the method must fail loudly (NN3).
                d_loss_output = self.loss_computer.compute_discriminator_loss(
                    real=target_batch,
                    fake=hr_fakes,
                    discriminator=self.discriminator_model,
                    epoch=epoch,
                    iteration=iteration,
                )

                d_total = d_loss_output.total if hasattr(d_loss_output, "total") else d_loss_output

                # Store detached metrics. Keep them on-device — `get_last_metrics`
                # resolves to Python floats at a coarser cadence, so no per-step
                # D2H sync happens inside the closure (NN#9).
                with torch.no_grad():
                    self._last_step_metrics["d_total_loss"] = d_total.detach()
                    if hasattr(d_loss_output, "components"):
                        for k, v in d_loss_output.components.items():
                            self._last_step_metrics[f"d_{k}"] = v.detach()

                return d_total

            return d_closure

        def g_closure() -> torch.Tensor:
            """g_closure.

            Returns:
                torch.Tensor: Description.
            """
            hr_fakes = self.generator_model(input_batch)

            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            if hasattr(self.config, "training") and getattr(
                self.config.training, "enforce_output_range", False
            ):
                hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

            # Called unguarded: ``setup_adversarial`` installs
            # ``UnifiedGANLossComputer`` unconditionally, so the deleted
            # ``hasattr``/``is None`` fallback could only route an unexpected
            # computer down a second critic feed -- raw here, realified in
            # ``gan.py`` -- or, with no critic, silently swap the whole
            # adversarial objective for plain L1. Both must now fail loudly (NN3).
            g_loss_output = self.loss_computer.compute_generator_loss(
                pred=hr_fakes,
                target=target_batch,
                discriminator=self.discriminator_model,
                epoch=epoch,
                iteration=iteration,
            )
            g_total = g_loss_output.total if hasattr(g_loss_output, "total") else g_loss_output

            # Store detached metrics. Keep them on-device — `get_last_metrics`
            # resolves to Python floats at a coarser cadence, so no per-step
            # D2H sync happens inside the closure (NN#9).
            with torch.no_grad():
                if g_total is not None:
                    self._last_step_metrics["g_total_loss"] = g_total.detach()
                if hasattr(g_loss_output, "components"):
                    for k, v in g_loss_output.components.items():
                        key = f"g_{k}" if not str(k).startswith("loss_") else k
                        self._last_step_metrics[key] = v.detach()

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
                            k: v.detach() if isinstance(v, torch.Tensor) else v
                            for k, v in train_metrics.items()
                        }
                    )

            return g_total

        return assemble_adversarial_step_configs(
            num_d_updates=num_d_updates,
            d_closure_factory=make_d_closure,
            g_closure=g_closure,
            discriminator=self.discriminator_model,
            generator=self.generator_model,
            opt_d=self.state.opt_d,
            opt_g=self.state.opt_g,
        )

    def get_last_metrics(self) -> dict[str, Any]:
        """Return the detached component metrics, ON-DEVICE (#707).

        The explicit ``.detach().item()`` here was the clearest case: the closures
        at ``:227`` and ``:309`` store these on-device with a comment saying
        ``get_last_metrics`` converts them, and `training_loop` then called it on
        every iteration. The sync count per step never dropped -- it only moved.
        Values are already detached at the store sites, so this is a plain copy.
        """
        return dict(getattr(self, "_last_step_metrics", {}))
