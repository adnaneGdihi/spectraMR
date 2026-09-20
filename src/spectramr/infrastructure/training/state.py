#!/usr/bin/env python
"""Infrastructure Training State

.. deprecated:: Phase 2 Migration
   Use :class:`spectramr.infrastructure.training.builders.environment.TrainingEnvironment` instead.
   TrainingState is mutable and will be removed in future releases.
   TrainingEnvironment is the new immutable SSOT for all training components.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from spectramr.core.module_utils import unwrap_model

from .interfaces import ITrainingState

# from spectramr.shared.utils.ema import ExponentialMovingAverage


@dataclass
class TrainingState(ITrainingState):
    """Concrete implementation of training state following SOLID principles.

    .. deprecated:: Phase 2 Migration
       This class is deprecated. Use TrainingEnvironment from
       spectramr.infrastructure.training.builders.environment instead.

       TrainingEnvironment provides:
       - Immutable frozen dataclass (thread-safe)
       - Dict-based component storage (extensible)
       - Built via TrainingEnvironmentDirector (validated)
       - Type-safe (no Union types needed)

       Migration path:

       OLD::

           state = TrainingState(...70+ args...)
           strategy = MyStrategy(state=state)

       NEW::

           env = TrainingEnvironmentDirector(config).build_environment()
           strategy = MyStrategy(env=env)
    """

    # Configuration
    config: Any  # TrainingConfig
    model_type: str
    model_type_for_saving: str
    model_info: dict[str, Any]
    is_diffusion: bool
    device: torch.device

    # Models and Optimizers
    generator: nn.Module
    discriminator: nn.Module | None
    opt_g: Optimizer
    opt_d: Optimizer | None
    sched_g: Any
    sched_d: Any | None

    # Loss Functions and Stability
    loss_function: Any  # GANLossSystem or dict
    criterion_l1: nn.Module | None
    criterion_mse: nn.Module | None
    criterion_perceptual: nn.Module | None
    stability_manager: Any  # spectramr.models.stability.StabilityManager
    grad_scaler: Any  # SafeGradScaler

    # Logging and Tracking
    gradient_logger: Any  # ExtensiveGradientLogger
    metrics_tracker: Any | None  # MetricsTracker
    profiling_state: Any  # ProfilingState

    # Data
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_samples: torch.Tensor

    # Paths and Identifiers
    run_name: str
    run_output_dir: str
    model_save_path: str
    loss_log_path: str
    epoch_summary_file: str
    csv_log_file: str
    image_gen_path: str
    profiler_log_path: str

    # Training Strategy
    training_strategy: Any | None = None  # TrainingStrategyBase

    # Checkpoint Manager
    checkpoint_manager: Any | None = None  # CheckpointManager

    # EMA for Generator (optional)
    ema: Any | None = None
    # Runtime-attached services (optional, set post-init)
    logging_service: Any | None = None
    metrics: dict[str, Any] | None = None
    environment: Any | None = None

    # Dynamic State (initialized with defaults)
    epoch_loss_tracker: dict[str, float] = field(default_factory=dict)
    recent_g_losses: list[float] = field(default_factory=list)
    recent_d_losses: list[float] = field(default_factory=list)
    recent_real_scores: list[float] = field(default_factory=list)
    recent_fake_scores: list[float] = field(default_factory=list)
    balance_window: int = 50
    epoch_data_list: list[dict[str, Any]] = field(default_factory=list)
    validation_history: list[dict[str, Any]] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_fid: float = float("inf")

    # Abstract method implementations
    _current_epoch: int = 0
    _global_step: int = 0

    @property
    def epoch(self) -> int:
        """Current training epoch."""
        return self._current_epoch

    @property
    def global_step(self) -> int:
        """Global training step."""
        return self._global_step

    @property
    def optimizer(self) -> Optimizer:
        """Alias for generator optimizer (for generic training compatibility)."""
        return self.opt_g

    def update_epoch(self, epoch: int) -> None:
        """Updates the current epoch."""
        self._current_epoch = epoch

    def increment_step(self) -> None:
        """Increments the global step."""
        self._global_step += 1

    # Early Stopping State
    early_stopping_counter: int = 0
    early_stopping_best_metric: float = float("inf")
    early_stopping_triggered: bool = False

    def __post_init__(self):
        """Post-initialization setup."""
        # Emit deprecation warning
        warnings.warn(
            "TrainingState is deprecated and will be removed in a future release. "
            "Use TrainingEnvironment from spectramr.infrastructure.training.builders.environment instead. "
            "See migration guide in docs/MIGRATION_PHASE2.md",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize empty collections if not provided
        if not self.epoch_loss_tracker:
            self.epoch_loss_tracker = {}
        if not self.recent_g_losses:
            self.recent_g_losses = []
        if not self.recent_d_losses:
            self.recent_d_losses = []
        if not self.recent_real_scores:
            self.recent_real_scores = []
        if not self.recent_fake_scores:
            self.recent_fake_scores = []
        if not self.epoch_data_list:
            self.epoch_data_list = []
        if not self.validation_history:
            self.validation_history = []

        # Initialize EMA if enabled in config
        ema_enabled = (
            self.config.ema.enabled if hasattr(self.config, "ema") and self.config.ema else False
        )
        if ema_enabled and self.ema is None:
            self.initialize_ema(self.config)

    def update_epoch_losses(self, losses: dict[str, float]) -> None:
        """Update epoch loss tracker with new losses."""
        self.epoch_loss_tracker.update(losses)

    def add_recent_losses(self, g_loss: float, d_loss: float | None = None) -> None:
        """Add recent losses to tracking lists."""
        self.recent_g_losses.append(g_loss)
        if d_loss is not None:
            self.recent_d_losses.append(d_loss)

        # Maintain window size
        if len(self.recent_g_losses) > self.balance_window:
            self.recent_g_losses.pop(0)
        if len(self.recent_d_losses) > self.balance_window:
            self.recent_d_losses.pop(0)

    def add_scores(self, real_score: float, fake_score: float) -> None:
        """Add discriminator scores to tracking."""
        self.recent_real_scores.append(real_score)
        self.recent_fake_scores.append(fake_score)

        # Maintain window size
        if len(self.recent_real_scores) > self.balance_window:
            self.recent_real_scores.pop(0)
        if len(self.recent_fake_scores) > self.balance_window:
            self.recent_fake_scores.pop(0)

    def get_recent_stats(self) -> dict[str, float]:
        """Get recent training statistics."""
        return {
            "avg_g_loss": (
                sum(self.recent_g_losses) / len(self.recent_g_losses)
                if self.recent_g_losses
                else 0.0
            ),
            "avg_d_loss": (
                sum(self.recent_d_losses) / len(self.recent_d_losses)
                if self.recent_d_losses
                else 0.0
            ),
            "avg_real_score": (
                sum(self.recent_real_scores) / len(self.recent_real_scores)
                if self.recent_real_scores
                else 0.5
            ),
            "avg_fake_score": (
                sum(self.recent_fake_scores) / len(self.recent_fake_scores)
                if self.recent_fake_scores
                else 0.5
            ),
        }

    def should_early_stop(self, current_metric: float) -> bool:
        """Check if early stopping should be triggered."""
        if current_metric < self.early_stopping_best_metric:
            self.early_stopping_best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.config.early_stopping.patience:
            self.early_stopping_triggered = True
            return True
        return False

    def reset_early_stopping(self) -> None:
        """Reset early stopping state."""
        self.early_stopping_counter = 0
        self.early_stopping_best_metric = float("inf")
        self.early_stopping_triggered = False

    def get_training_summary(self) -> dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            "run_name": self.run_name,
            "model_type": self.model_type,
            "current_epoch_losses": self.epoch_loss_tracker,
            "recent_stats": self.get_recent_stats(),
            "validation_history": self.validation_history,
            "best_val_loss": self.best_val_loss,
            "best_fid": self.best_fid,
            "early_stopping_triggered": self.early_stopping_triggered,
            "early_stopping_counter": self.early_stopping_counter,
            "ema_enabled": self.ema is not None,
        }

    def initialize_ema(self, config) -> None:
        """Initialize EMA for the generator.

        This used to branch on ``config.ema.enable_adaptive_ema`` into
        ``spectramr.models.utils.adaptive_ema``, a module deleted in ff0efff9f —
        so the branch could only ever raise ``ImportError`` and fall through to
        standard EMA, silently discarding the whole adaptive declaration
        (#1294). The adaptive schedule now lives in :class:`ModelEma` itself,
        so there is one construction path and the knobs are read where they
        are honoured.
        """
        if self.ema is not None:
            return

        from spectramr.infrastructure.optimization.ema import ModelEma

        # No ``except ImportError`` here: ModelEma is an in-tree module, so a
        # failure to import it is a packaging bug, not a missing optional
        # dependency. Swallowing it would disable EMA for the whole run and
        # let validation grade live weights while the log claimed EMA was on
        # (non-negotiable 3).
        ema_cfg = getattr(config, "ema", None)
        if ema_cfg is None:
            self.ema = ModelEma(self.generator, decay=0.999, device=self.device)
            return

        self.ema = ModelEma(
            self.generator,
            decay=ema_cfg.decay,
            device=self.device,
            # num_updates warmup ramp (config.ema.warmup, default True) so
            # validation isn't grading a near-random shadow on short runs.
            warmup=getattr(ema_cfg, "warmup", True),
            adaptive=getattr(ema_cfg, "enable_adaptive_ema", False),
            warmup_steps=getattr(ema_cfg, "warmup_steps", 0),
            initial_decay=getattr(ema_cfg, "initial_decay", 0.0),
            final_decay=getattr(ema_cfg, "final_decay", 0.999),
        )

    def update_ema(self) -> None:
        """Update EMA parameters.

        The former ``loss_value`` / ``gradient_norm`` arguments fed
        ``update_stability_score`` on the deleted adaptive-EMA class. That
        signal is not part of the restored schedule (see
        :meth:`~spectramr.config.schemas.ema.EMAConfigSchema.validate_adaptive_ema_is_fully_wired`),
        so accepting them here would advertise an unread knob
        (non-negotiable 8). The decay ramp is driven entirely by
        ``ModelEma.num_updates``.
        """
        if self.ema is not None:
            # Unwrap for the same reason training_loop does: the shadow carries
            # bare keys, and a DDP/FSDP/DeepSpeed/compile wrapper prefixes every
            # key on the live model, which makes the key-matched blend a no-op
            # (#2172).
            self.ema.update(unwrap_model(self.generator))

    def apply_ema_for_inference(self) -> nn.Module:
        """Apply EMA parameters for inference."""
        if self.ema is not None:
            # Return the EMA module for inference
            if hasattr(self.ema, "module"):
                return self.ema.module
            return self.ema
        return self.generator

    def get_ema_state_dict(self) -> dict[str, Any] | None:
        """Get EMA state for saving.

        Delegates to ``ModelEma.shadow_state_dict`` so this spelling and the
        checkpoint director's produce the same bytes — bare weight keys plus the
        warmup counter. It previously called ``state_dict()``, which emits
        ``module.``-prefixed keys, a third and divergent on-disk shape.
        """
        if self.ema is not None:
            return self.ema.shadow_state_dict()
        return None

    def load_ema_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load EMA state from saved state."""
        if self.ema is not None and state_dict:
            self.ema.load_state_dict(state_dict)
