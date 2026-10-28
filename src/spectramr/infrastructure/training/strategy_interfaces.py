"""Training Strategy Interfaces

This module defines the interfaces for training strategy collaborators
that handle specific concerns in the training process.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

# Canonical IOptimizerStepper lives in interfaces.py — re-exported here for
# backwards-compatibility so existing imports keep working.
from spectramr.infrastructure.training.interfaces import IOptimizerStepper  # noqa: F401
from spectramr.infrastructure.training.nonfinite_monitor import (
    NonFiniteMonitor,
    guard_nonfinite_loss,
)


class ILossComputer(ABC):
    """Interface for computing losses in training strategies."""

    @abstractmethod
    def compute_loss(
        self,
        lr_reals: torch.Tensor,
        hr_reals: torch.Tensor,
        hr_fakes: torch.Tensor,
        discriminator_outputs: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for a training step.

        Args:
            lr_reals: Low-resolution real images
            hr_reals: High-resolution real images
            hr_fakes: High-resolution fake images
            discriminator_outputs: Optional discriminator outputs
            **kwargs: Additional arguments for specific loss types

        Returns:
            Dictionary of loss components

        """
        raise NotImplementedError(
            "The `compute_loss` method must be implemented by subclasses of "
            "ILossComputer. This method is responsible for calculating the "
            "total loss for a given training step.",
        )


class IStepPolicy(ABC):
    """Who owns the backward pass, the optimizer step, and their bookkeeping.

    ``backward_and_step`` has been the de-facto seam of the training loop for a
    long time -- ``StepExecutor.execute_step`` calls it exactly once and nothing
    else drives an optimizer -- but it was never declared anywhere. It was an
    undeclared interface with one implementation, which is fine right up until a
    second thing needs to own the step.

    Two such things do. A **sharded/engine-owned** backend (DeepSpeed) performs
    its own loss scaling, gradient accumulation and ``zero_grad`` inside
    ``engine.backward``/``engine.step``; running the executor's versions as well
    gives 1/N^2 loss scaling and one real step per N^2 micro-batches. **SAM**
    needs two forward+backward passes, so it needs the closure rather than a
    computed loss. Declaring the seam is what lets those arrive as injected
    objects instead of ``if backend == ...`` branches in the loop.

    The two capability flags exist so the executor can ask "do you already do
    this?" rather than test a backend name:

    ``owns_gradient_accumulation``
        The policy divides the loss and decides accumulation boundaries itself.
        The executor must then NOT divide, and must treat every micro-batch as a
        step.
    ``owns_zero_grad``
        The policy zeroes gradients inside its own step. The executor must not.
    ``owns_lr_scheduler``
        The backend steps the LR scheduler it was handed inside its own step
        (DeepSpeed does, at every accumulation boundary). The loop must not step
        it again, or the schedule decays at twice the configured rate.

    Defaults are ``False``, so an existing policy that declares neither behaves
    exactly as before.
    """

    #: See the class docstring. Read once by ``StepExecutor.__init__``.
    owns_gradient_accumulation: bool = False
    owns_zero_grad: bool = False
    owns_lr_scheduler: bool = False

    def guard_loss(
        self,
        loss: "torch.Tensor",
        *,
        name: str,
        global_step: int,
        scaler: Any | None,
        monitor: NonFiniteMonitor | None = None,
    ) -> None:
        """Reject a loss that must not be allowed to reach ``backward()``.

        Default: ``guard_nonfinite_loss`` -- defer to a ``GradScaler``, record
        into the training loop's ``monitor`` (no sync; the loop stops on the
        verdict), or, with no monitor, raise synchronously on a non-finite loss.

        Override to no-op when the backend has its own overflow handling --
        stopping here would kill runs it is designed to survive.
        """
        guard_nonfinite_loss(
            loss, name=name, global_step=global_step, scaler=scaler, monitor=monitor
        )

    @abstractmethod
    def clip_gradients(self, model: "torch.nn.Module", max_norm: float) -> None:
        """Clip gradients for ``model`` in place.

        A method rather than a free function because the correct call is not the
        same for every backend: ``clip_grad_norm_(model.parameters(), ...)``
        computes a *per-shard* norm under FSDP and therefore clips inconsistently
        across ranks -- silently, and it presents as training instability rather
        than as a bug. FSDP requires ``model.clip_grad_norm_()``.
        """
        raise NotImplementedError("IStepPolicy.clip_gradients must be implemented")


class IAMPPolicy(IStepPolicy):
    """Interface for automatic mixed precision policy."""

    @abstractmethod
    def should_use_amp(
        self,
        model_type: str,
        device: torch.device,
        config: Any,
    ) -> bool:
        """Determine if AMP should be used for given model and device.

        Args:
            model_type: Type of model (gan, reconstruction, diffusion, etc.)
            device: Device being used
            config: Training configuration

        Returns:
            Whether to use automatic mixed precision

        """
        # TODO: Subclasses must implement this method
        raise NotImplementedError("IAMPPolicy.should_use_amp must be implemented")

    @abstractmethod
    def get_autocast_device(self, device: torch.device) -> str:
        """Get device type for autocast.

        Args:
            device: PyTorch device

        Returns:
            Device type string for autocast

        """
        # TODO: Subclasses must implement this method
        raise NotImplementedError("IAMPPolicy.get_autocast_device must be implemented")


class IMetricsReporter(ABC):
    """Interface for reporting training metrics and handling validation."""

    @abstractmethod
    def report_losses(
        self,
        losses: dict[str, float | torch.Tensor],
        epoch: int,
        step_type: str = "train",
    ) -> dict[str, float]:
        """Report loss metrics, converting tensors to scalars.

        Args:
            losses: Dictionary of loss tensors/values
            epoch: Current epoch
            step_type: Type of step (train/validation)

        Returns:
            Dictionary of scalar loss values

        """
        # TODO: Subclasses must implement this method
        raise NotImplementedError("IMetricsReporter.report_losses must be implemented")

    @abstractmethod
    def detect_anomalies(
        self,
        losses: dict[str, float | torch.Tensor],
    ) -> tuple[bool, bool]:
        """Detect NaN/Inf values in losses.

        Args:
            losses: Dictionary of loss tensors/values

        Returns:
            Tuple of (nan_detected, inf_detected)

        """
        # TODO: Subclasses must implement this method
        raise NotImplementedError(
            "IMetricsReporter.detect_anomalies must be implemented",
        )

    @abstractmethod
    def compute_validation_metrics(
        self,
        hr_fakes: torch.Tensor,
        hr_reals: torch.Tensor,
        device: str,
    ) -> dict[str, float]:
        """Compute validation metrics.

        Args:
            hr_fakes: Generated fake images
            hr_reals: Real images
            device: Device string

        Returns:
            Dictionary of validation metrics

        """
        # TODO: Subclasses must implement this method
        raise NotImplementedError(
            "IMetricsReporter.compute_validation_metrics must be implemented",
        )
