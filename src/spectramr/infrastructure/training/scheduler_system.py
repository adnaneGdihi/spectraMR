"""Enhanced Learning Rate Scheduler and Warmup System
==================================================

This module provides comprehensive learning rate scheduling with warmup capabilities
for all model types in the GAN training system.

Features:
- Multiple scheduler types: cosine_annealing, step_lr, exponential, linear, constant
- Linear warmup with smooth transition to main scheduler
- Model-specific scheduler recommendations
- Integration with existing training pipeline

Author: spectraMR Research Project
Date: August 2025
"""

import logging
from collections.abc import Callable
from typing import Any

import torch

from spectramr.infrastructure.training.schedulers.monotone_cosine import MonotoneCosineLR

logger = logging.getLogger(__name__)
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    StepLR,
)

# Registry for scheduler creation functions
SCHEDULER_REGISTRY: dict[
    str,
    Callable[[torch.optim.Optimizer, dict[str, Any]], torch.optim.lr_scheduler._LRScheduler],
] = {}


def register_scheduler(name: str):
    """Decorator to register a scheduler creation function"""

    def decorator(
        func: Callable[
            [torch.optim.Optimizer, dict[str, Any]],
            torch.optim.lr_scheduler._LRScheduler,
        ],
    ):
        """decorator.

        Args:
            func (Callable[[torch.optim.Optimizer, dict[str, Any]], torch.optim.lr_scheduler._LRScheduler]): Description.
        Returns:
            Any: Description.
        """
        SCHEDULER_REGISTRY[name] = func
        return func

    return decorator


# Scheduler creation functions
@register_scheduler("cosine_annealing")
def _create_cosine_annealing_scheduler(
    optimizer: torch.optim.Optimizer, params: dict[str, Any]
) -> MonotoneCosineLR:
    """Cosine decay held at ``eta_min`` past ``T_max``; never restarts on resume."""
    return MonotoneCosineLR(
        optimizer,
        T_max=params.get("T_max", 100),
        eta_min=params.get("eta_min", 1e-6),
    )


@register_scheduler("cosine_annealing_warm_restarts")
def _create_cosine_annealing_warm_restarts_scheduler(
    optimizer: torch.optim.Optimizer, params: dict[str, Any]
) -> CosineAnnealingWarmRestarts:
    """Create CosineAnnealingWarmRestarts scheduler"""
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=params.get("T_0", 10),
        T_mult=params.get("T_mult", 2),
        eta_min=params.get("eta_min", 1e-6),
    )


@register_scheduler("step_lr")
def _create_step_lr_scheduler(optimizer: torch.optim.Optimizer, params: dict[str, Any]) -> StepLR:
    """Create StepLR scheduler"""
    return StepLR(
        optimizer,
        step_size=params.get("step_size", 30),
        gamma=params.get("gamma", 0.1),
    )


@register_scheduler("exponential")
def _create_exponential_scheduler(
    optimizer: torch.optim.Optimizer, params: dict[str, Any]
) -> ExponentialLR:
    """Create ExponentialLR scheduler"""
    return ExponentialLR(
        optimizer,
        gamma=params.get("gamma", 0.95),
    )


@register_scheduler("constant")
def _create_constant_scheduler(
    optimizer: torch.optim.Optimizer, params: dict[str, Any]
) -> ConstantLR:
    """Create ConstantLR — an explicit hold-the-LR scheduler.

    ``lr_scheduler_strategy: constant`` appears in the corpus and resolved to no
    registered factory. Defaults (factor 1.0, total_iters 0) make it a genuine
    no-op rather than torch's 1/3-for-5-epochs default, so "constant" means
    constant.
    """
    return ConstantLR(
        optimizer,
        factor=params.get("factor", 1.0),
        total_iters=params.get("total_iters", 0),
    )


@register_scheduler("linear")
def _create_linear_scheduler(optimizer: torch.optim.Optimizer, params: dict[str, Any]) -> LinearLR:
    """Create LinearLR scheduler.

    Registered because ``lr_scheduler_strategy: linear_decay`` appears in the
    corpus; before it was wired, that name resolved to nothing and the arm
    silently got cosine annealing.
    """
    return LinearLR(
        optimizer,
        start_factor=params.get("start_factor", 1.0),
        end_factor=params.get("end_factor", 0.01),
        total_iters=params.get("total_iters", 100),
    )


class WarmupScheduler:
    """Learning rate scheduler with linear warmup followed by a main scheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        main_scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_steps: int,
        warmup_start_lr: float | None = None,
        warmup_end_lr: float | None = None,
    ):
        """__init__.

        Args:
            optimizer (torch.optim.Optimizer): Description.
            main_scheduler (torch.optim.lr_scheduler._LRScheduler): Description.
            warmup_steps (int): Description.
            warmup_start_lr (Optional[float]): Description.
            warmup_end_lr (Optional[float]): Description.
        """
        self.optimizer = optimizer
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Set warmup learning rates
        if warmup_start_lr is None:
            self.warmup_start_lr = [lr * 0.01 for lr in self.base_lrs]
        else:
            self.warmup_start_lr = [warmup_start_lr] * len(self.base_lrs)

        if warmup_end_lr is None:
            self.warmup_end_lr = self.base_lrs
        else:
            self.warmup_end_lr = [warmup_end_lr] * len(self.base_lrs)

        # Apply the warmup start LR immediately. ``step()`` only runs AFTER the
        # first optimizer step, so without this the very first update — the one
        # warmup exists to protect — ran at the full base LR (issue #533).
        if self.warmup_steps > 0:
            for group, lr in zip(optimizer.param_groups, self.warmup_start_lr, strict=False):
                group["lr"] = lr

    def step(self) -> None:
        """Step the scheduler"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup phase
            for i, param_group in enumerate(self.optimizer.param_groups):
                warmup_progress = self.current_step / self.warmup_steps
                lr = self.warmup_start_lr[i] + warmup_progress * (
                    self.warmup_end_lr[i] - self.warmup_start_lr[i]
                )
                param_group["lr"] = lr
        else:
            self.main_scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Get the last learning rate"""
        if self.current_step <= self.warmup_steps:
            # During warmup, calculate current LR
            lrs = []
            for i in range(len(self.base_lrs)):
                warmup_progress = self.current_step / self.warmup_steps
                lr = self.warmup_start_lr[i] + warmup_progress * (
                    self.warmup_end_lr[i] - self.warmup_start_lr[i]
                )
                lrs.append(lr)
            return lrs
        return self.main_scheduler.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state"""
        return {
            "current_step": self.current_step,
            "main_scheduler": self.main_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load scheduler state, refusing one saved by a different decay family."""
        check_scheduler_state_compatible(
            self.main_scheduler, state_dict["main_scheduler"], "main scheduler"
        )
        self.current_step = state_dict["current_step"]
        self.main_scheduler.load_state_dict(state_dict["main_scheduler"])


class SchedulerStateMismatchError(ValueError):
    """A saved scheduler state belongs to a different scheduler family than the one built."""


#: Keys every torch LR scheduler carries; the family-specific remainder identifies the family.
_COMMON_SCHEDULER_KEYS = frozenset(
    {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_get_lr_called_within_step",
        "_last_lr",
        "_is_initial",
        "verbose",
    }
)


def check_scheduler_state_compatible(scheduler: Any, state: dict[str, Any], what: str) -> None:
    """Raise when ``state`` was saved by a scheduler family other than ``scheduler``'s.

    ``load_state_dict`` copies whatever keys it is given, so a warm-restart state loaded into
    a plain cosine schedule keeps its ``last_epoch`` past the new ``T_max`` and the LR climbs
    back up instead of annealing. The family-specific keys (``T_0``/``T_cur`` against
    ``T_max``) tell the two apart.
    """
    built = set(scheduler.state_dict()) - _COMMON_SCHEDULER_KEYS
    saved = set(state) - _COMMON_SCHEDULER_KEYS
    if built != saved:
        raise SchedulerStateMismatchError(
            f"The checkpoint's {what} state was saved by a different scheduler family "
            f"({sorted(saved)}) than this run builds ({sorted(built)}), e.g. a warm-restart "
            "schedule resumed after T_0/T_mult were dropped. Resume with the scheduler it "
            "was saved with, or start the schedule fresh with "
            "checkpoint.load_scheduler_state: false."
        )


def load_scheduler_state(scheduler: Any, state: dict[str, Any], what: str = "LR scheduler") -> None:
    """``scheduler.load_state_dict(state)`` after the family check; the one resume entry point."""
    # A WarmupScheduler checks its own main scheduler, which is also the DeepSpeed resume path.
    if not isinstance(scheduler, WarmupScheduler):
        check_scheduler_state_compatible(scheduler, state, what)
    scheduler.load_state_dict(state)


class SchedulerFactory:
    """Factory for creating learning rate schedulers with warmup"""

    @staticmethod
    def get_model_specific_scheduler_recommendations() -> dict[str, dict[str, Any]]:
        """Get recommended scheduler configurations for different model types"""
        return {
            # Standard models
            "unet": {
                "scheduler": "cosine_annealing",
                "params": {"T_max": 50, "eta_min": 1e-6},
                "warmup_steps": 100,
            },
            "vit": {
                "scheduler": "cosine_annealing_warm_restarts",
                "params": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6},
                "warmup_steps": 500,
            },
            "swin": {
                "scheduler": "cosine_annealing_warm_restarts",
                "params": {"T_0": 15, "T_mult": 2, "eta_min": 1e-6},
                "warmup_steps": 300,
            },
            "stylegan": {
                "scheduler": "step_lr",
                "params": {"step_size": 20, "gamma": 0.5},
                "warmup_steps": 200,
            },
            "diffusion": {
                "scheduler": "cosine_annealing",
                "params": {"T_max": 100, "eta_min": 1e-7},
                "warmup_steps": 1000,
            },
            "standard": {
                "scheduler": "step_lr",
                "params": {"step_size": 30, "gamma": 0.7},
                "warmup_steps": 100,
            },
            # KAN models
            "vit_kan": {
                "scheduler": "cosine_annealing",
                "params": {"T_max": 30, "eta_min": 1e-7},
                "warmup_steps": 400,
            },
            "swin_transformer_kan": {
                "scheduler": "cosine_annealing",
                "params": {"T_max": 25, "eta_min": 1e-7},
                "warmup_steps": 300,
            },
            "stylegan_kan": {
                "scheduler": "step_lr",
                "params": {"step_size": 15, "gamma": 0.6},
                "warmup_steps": 250,
            },
            # Diffusion variants
            "cold_diffusion": {
                "scheduler": "cosine_annealing",
                "params": {"T_max": 80, "eta_min": 1e-7},
                "warmup_steps": 800,
            },
            "score_based_diffusion": {
                "scheduler": "exponential",
                "params": {"gamma": 0.99},
                "warmup_steps": 500,
            },
            "chi_square_diffusion": {
                "scheduler": "cosine_annealing",
                "params": {"T_max": 60, "eta_min": 1e-7},
                "warmup_steps": 600,
            },
            "stable_diffusion": {
                "scheduler": "cosine_annealing_warm_restarts",
                "params": {"T_0": 20, "T_mult": 1, "eta_min": 1e-7},
                "warmup_steps": 1000,
            },
        }

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        scheduler_params: dict[str, Any] | None = None,
        warmup_steps: int = 0,
        warmup_start_lr: float | None = None,
        warmup_end_lr: float | None = None,
        total_epochs: int = 100,
        model_type: str | None = None,
    ) -> torch.optim.lr_scheduler._LRScheduler | WarmupScheduler:
        """Create a learning rate scheduler with optional warmup

        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler
            scheduler_params: Parameters for the scheduler
            warmup_steps: Number of warmup steps (0 for no warmup)
            warmup_start_lr: Starting learning rate for warmup
            warmup_end_lr: Ending learning rate for warmup
            total_epochs: Total number of training epochs
            model_type: Model type for automatic recommendations

        Returns:
            Scheduler (with or without warmup)

        """
        # Use model-specific recommendations if available
        if model_type and not scheduler_params:
            recommendations = SchedulerFactory.get_model_specific_scheduler_recommendations()
            if model_type in recommendations:
                rec = recommendations[model_type]
                scheduler_type = rec["scheduler"]
                scheduler_params = rec["params"]
                if warmup_steps == 0:
                    warmup_steps = rec["warmup_steps"]

        # Default parameters
        if scheduler_params is None:
            scheduler_params = {}

        # Create main scheduler using registry
        if scheduler_type not in SCHEDULER_REGISTRY:
            # Fallback for cosine -> cosine_annealing
            if scheduler_type == "cosine":
                scheduler_type = "cosine_annealing"
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        main_scheduler = SCHEDULER_REGISTRY[scheduler_type](optimizer, scheduler_params)

        # Add warmup if requested
        if warmup_steps > 0:
            scheduler = WarmupScheduler(
                optimizer=optimizer,
                main_scheduler=main_scheduler,
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                warmup_end_lr=warmup_end_lr,
            )
        else:
            scheduler = main_scheduler

        return scheduler

    @staticmethod
    def create_dual_schedulers(
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer | None,
        scheduler_type: str,
        scheduler_params: dict[str, Any] | None = None,
        warmup_steps: int = 0,
        total_epochs: int = 100,
        model_type: str | None = None,
        different_d_scheduler: bool = False,
    ) -> tuple[
        torch.optim.lr_scheduler._LRScheduler | WarmupScheduler,
        torch.optim.lr_scheduler._LRScheduler | WarmupScheduler | None,
    ]:
        """Create schedulers for both generator and discriminator

        Args:
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer (optional)
            scheduler_type: Type of scheduler
            scheduler_params: Parameters for the scheduler
            warmup_steps: Number of warmup steps
            total_epochs: Total number of training epochs
            model_type: Model type for recommendations
            different_d_scheduler: Use different scheduler for discriminator

        Returns:
            Tuple of (generator_scheduler, discriminator_scheduler)

        """
        # Create generator scheduler
        scheduler_g = SchedulerFactory.create_scheduler(
            optimizer=optimizer_g,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
            warmup_steps=warmup_steps,
            total_epochs=total_epochs,
            model_type=model_type,
        )

        # Create discriminator scheduler
        scheduler_d = None
        if optimizer_d is not None:
            if different_d_scheduler:
                # Use different scheduler for discriminator (typically more
                # conservative)
                d_scheduler_type = "step_lr" if scheduler_type != "step_lr" else "exponential"
                d_scheduler_params = {
                    "step_size": 40 if d_scheduler_type == "step_lr" else None,
                    "gamma": 0.8 if d_scheduler_type == "step_lr" else 0.98,
                }
                d_warmup_steps = max(
                    1,
                    warmup_steps // 2,
                )  # Shorter warmup for discriminator
            else:
                # Use same scheduler as generator
                d_scheduler_type = scheduler_type
                d_scheduler_params = scheduler_params
                d_warmup_steps = warmup_steps

            scheduler_d = SchedulerFactory.create_scheduler(
                optimizer=optimizer_d,
                scheduler_type=d_scheduler_type,
                scheduler_params=d_scheduler_params,
                warmup_steps=d_warmup_steps,
                total_epochs=total_epochs,
                model_type=model_type,
            )

        return scheduler_g, scheduler_d


class AdaptiveWarmupScheduler(WarmupScheduler):
    """Adaptive warmup scheduler that adjusts warmup based on training progress"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        main_scheduler: torch.optim.lr_scheduler._LRScheduler,
        initial_warmup_steps: int,
        loss_history_window: int = 10,
        stability_threshold: float = 0.1,
        max_warmup_extension: int = 500,
    ):
        """__init__.

        Args:
            optimizer (torch.optim.Optimizer): Description.
            main_scheduler (torch.optim.lr_scheduler._LRScheduler): Description.
            initial_warmup_steps (int): Description.
            loss_history_window (int): Description.
            stability_threshold (float): Description.
            max_warmup_extension (int): Description.
        """
        super().__init__(optimizer, main_scheduler, initial_warmup_steps)
        self.initial_warmup_steps = initial_warmup_steps
        self.loss_history = []
        self.loss_history_window = loss_history_window
        self.stability_threshold = stability_threshold
        self.max_warmup_extension = max_warmup_extension
        self.warmup_extended = False

    def step(self, loss: float | None = None) -> None:
        """Step with adaptive warmup extension"""
        # Track loss for stability assessment
        if loss is not None:
            self.loss_history.append(loss)
            if len(self.loss_history) > self.loss_history_window:
                self.loss_history.pop(0)

        # Check if we need to extend warmup
        if (
            self.current_step == self.warmup_steps
            and not self.warmup_extended
            and len(self.loss_history) >= self.loss_history_window
        ):
            # Calculate loss stability
            recent_losses = self.loss_history[-self.loss_history_window :]
            loss_std = torch.std(torch.tensor(recent_losses)).item()
            loss_mean = torch.mean(torch.tensor(recent_losses)).item()

            # If loss is unstable, extend warmup
            if loss_std / (loss_mean + 1e-8) > self.stability_threshold:
                extension = min(self.max_warmup_extension, self.warmup_steps // 2)
                self.warmup_steps += extension
                self.warmup_extended = True
                logger.warning(
                    f"⚠️  Training instability detected. Extending warmup by {extension} steps.",
                )

        # Call parent step
        super().step()


def create_enhanced_scheduler_system(
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    scheduler_type: str = "cosine_annealing",
    warmup_steps: int = 0,
    total_epochs: int = 100,
    model_type: str | None = None,
    adaptive_warmup: bool = False,
    scheduler_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an enhanced scheduler system with comprehensive features

    Args:
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        scheduler_type: Type of scheduler to use
        warmup_steps: Number of warmup steps
        total_epochs: Total training epochs
        model_type: Model type for recommendations
        adaptive_warmup: Use adaptive warmup based on training stability
        scheduler_params: Custom scheduler parameters

    Returns:
        Dictionary containing schedulers and configuration

    """
    if adaptive_warmup and warmup_steps > 0:
        # Create adaptive warmup schedulers
        main_scheduler_g = SchedulerFactory.create_scheduler(
            optimizer_g,
            scheduler_type,
            scheduler_params,
            warmup_steps=0,
            total_epochs=total_epochs,
            model_type=model_type,
        )

        scheduler_g = AdaptiveWarmupScheduler(
            optimizer_g,
            main_scheduler_g,
            warmup_steps,
        )

        scheduler_d = None
        if optimizer_d is not None:
            main_scheduler_d = SchedulerFactory.create_scheduler(
                optimizer_d,
                scheduler_type,
                scheduler_params,
                warmup_steps=0,
                total_epochs=total_epochs,
                model_type=model_type,
            )
            scheduler_d = AdaptiveWarmupScheduler(
                optimizer_d,
                main_scheduler_d,
                warmup_steps // 2,
            )

    else:
        # Create standard schedulers
        scheduler_g, scheduler_d = SchedulerFactory.create_dual_schedulers(
            optimizer_g,
            optimizer_d,
            scheduler_type,
            scheduler_params,
            warmup_steps,
            total_epochs,
            model_type,
        )

    return {
        "scheduler_g": scheduler_g,
        "scheduler_d": scheduler_d,
        "scheduler_type": scheduler_type,
        "warmup_steps": warmup_steps,
        "adaptive_warmup": adaptive_warmup,
        "total_epochs": total_epochs,
        "model_type": model_type,
    }


def get_scheduler_info(
    scheduler: torch.optim.lr_scheduler._LRScheduler | WarmupScheduler,
) -> dict[str, Any]:
    """Get information about a scheduler"""
    info = {
        "type": type(scheduler).__name__,
        "current_lr": (scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else "N/A"),
    }

    if isinstance(scheduler, WarmupScheduler):
        info.update(
            {
                "warmup_steps": scheduler.warmup_steps,
                "current_step": scheduler.current_step,
                "in_warmup": scheduler.current_step <= scheduler.warmup_steps,
                "main_scheduler_type": type(scheduler.main_scheduler).__name__,
            },
        )

    if isinstance(scheduler, AdaptiveWarmupScheduler):
        info.update(
            {
                "warmup_extended": scheduler.warmup_extended,
                "loss_history_length": len(scheduler.loss_history),
            },
        )

    return info


if __name__ == "__main__":
    # Example usage
    from torch import nn

    # Create a dummy model and optimizer
    model = nn.Linear(10, 1)
    from spectramr.infrastructure.training.builders import OptimizationBuilder

    optimizer = OptimizationBuilder.create_single_optimizer(model.parameters(), learning_rate=0.001)

    # Test different scheduler types
    scheduler_types = [
        "cosine_annealing",
        "step_lr",
        "exponential",
    ]

    for scheduler_type in scheduler_types:
        logger.debug(f"\nTesting {scheduler_type} scheduler:")

        scheduler = SchedulerFactory.create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=10,
            total_epochs=100,
            model_type="unet",
        )

        logger.debug(f"Scheduler created: {type(scheduler).__name__}")
        logger.debug(f"Initial LR: {scheduler.get_last_lr()}")

        # Test a few steps
        for step in range(15):
            scheduler.step()

            if step in [0, 5, 10, 14]:
                logger.debug(f"Step {step}: LR = {scheduler.get_last_lr()}")
