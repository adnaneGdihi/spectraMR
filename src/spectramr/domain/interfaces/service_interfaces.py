"""Infrastructure Service Interfaces
================================

This module defines interfaces for infrastructure services following SOLID principles.
Each interface represents a single responsibility.
"""

import logging as stdlib_logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from spectramr.config.settings import TrainingSettings

import torch
from torch import nn

from spectramr.domain.entities.data.types import MetricsDict, TensorDict

# Import ICheckpointService from its dedicated module
from spectramr.domain.interfaces.checkpoint_service_interface import ICheckpointService

# Type alias for configuration objects (TrainingSettings or dict)
ConfigType = Union["TrainingSettings", dict[str, Any]]
SessionType = dict[str, Any]  # Training session state

__all__ = [
    "ConfigType",
    "ICheckpointService",
    "IDeviceService",
    "ILoggingService",
    "IMetricsService",
    "SessionType",
]


class ILoggingService(ABC):
    """Interface for logging operations.

    ### Logging Interface
    ```mermaid
    classDiagram
        class ILoggingService {
            <<interface>>
            +setup(log_dir, log_level)
            +get_logger(name)
            +log_info(message)
            +log_warning(message)
            +log_error(message)
            +log_critical(message)
            +log_debug(message)
        }
    ```
    """

    @abstractmethod
    def setup(
        self,
        log_dir: str,
        log_level: str = "INFO",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Set up the logger."""

    @abstractmethod
    def get_logger(self, name: str) -> stdlib_logging.Logger:
        """Get a logger instance."""

    @abstractmethod
    def log_info(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log an info message."""

    @abstractmethod
    def log_warning(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a warning message."""

    @abstractmethod
    def log_error(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log an error message."""

    @abstractmethod
    def log_critical(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a critical message: a condition that stops the run.

        Abstract rather than a default shim because an implementation that silently
        lacked this rung is exactly how the training loop's divergence tripwire spent
        its life raising ``AttributeError`` instead of halting (#2254).
        """

    @abstractmethod
    def log_debug(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a debug message."""

    @abstractmethod
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        **kwargs: Any,
    ) -> None:
        """Log a single batch of images with a tag."""

    @abstractmethod
    def log_images_batch(
        self,
        images: torch.Tensor | dict[str, Any],
        step: int,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a batch of images."""

    @abstractmethod
    def set_step(self, step: int) -> None:
        """Set the current global step."""

    @abstractmethod
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""

    @property
    @abstractmethod
    def step(self) -> int:
        """Return the current global step."""

    @property
    @abstractmethod
    def epoch(self) -> int:
        """Return the current epoch."""


class IConfigurationService(ABC):
    """Interface for configuration management.

    ### Configuration Interface
    ```mermaid
    classDiagram
        class IConfigurationService {
            <<interface>>
            +load_config(path)
            +validate_config(config)
            +update_config(config, updates)
        }
    ```
    """

    @abstractmethod
    def load_config(self, path: str) -> ConfigType:
        """Load configuration from path."""

    @abstractmethod
    def validate_config(self, config: ConfigType) -> bool:
        """Validate configuration."""

    @abstractmethod
    def update_config(self, config: ConfigType, updates: dict[str, Any]) -> ConfigType:
        """Update configuration with new values."""


class ITrainingOrchestrator(ABC):
    """Interface for training orchestration."""

    @abstractmethod
    def orchestrate_training(self, config: ConfigType) -> MetricsDict:
        """Orchestrate the training process."""

    @abstractmethod
    def finalize_training(self, session: SessionType) -> None:
        """Finalize training session."""

    @abstractmethod
    def run_full_training_pipeline(self, config: ConfigType) -> MetricsDict:
        """Run the complete training pipeline."""

    @abstractmethod
    def setup_training(self, config: ConfigType) -> None:
        """Setup training environment."""

    @abstractmethod
    def cleanup_training(self) -> None:
        """Cleanup after training."""


class IDeviceManager(ABC):
    """Interface for device management.

    ### Device Management
    ```mermaid
    classDiagram
        class IDeviceManager {
            <<interface>>
            +get_device() torch.device
            +move_to_device(data)
            +is_cuda_available() bool
        }
    ```
    """

    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the current device."""

    @abstractmethod
    def move_to_device(self, data: torch.Tensor | TensorDict) -> torch.Tensor | TensorDict:
        """Move data to the current device."""

    @abstractmethod
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""


# Alias for backward compatibility
IDeviceService = IDeviceManager


class ICheckpointManager(ABC):
    """Interface for model checkpoint management."""

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        **kwargs,
    ) -> None:
        """Save a model checkpoint."""

    @abstractmethod
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Load a model checkpoint."""

    @abstractmethod
    def list_checkpoints(self) -> list:
        """list available checkpoints."""


class IMetricsService(ABC):
    """Interface for metrics collection and reporting."""

    @abstractmethod
    def update_metrics(self, metrics: dict[str, float]) -> None:
        """Update current metrics."""

    @abstractmethod
    def get_current_metrics(self) -> dict[str, float]:
        """Get current metrics."""

    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset metrics."""

    @abstractmethod
    def log_metrics(self, step: int) -> None:
        """Log metrics at a specific step."""


class IMemoryOptimizationService(ABC):
    """Interface for memory optimization."""

    @abstractmethod
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear memory cache."""

    @abstractmethod
    def get_memory_stats(self) -> dict[str, float]:
        """Get memory statistics."""


class I3DGenerator(ABC):
    """Interface for 3D asset generation services (TRELLIS)."""

    @abstractmethod
    def generate_3d_asset(
        self,
        input_data: torch.Tensor | str,
        output_format: str = "gaussian",
        **kwargs,
    ) -> TensorDict:
        """Generate 3D assets from input data."""

    @abstractmethod
    def generate_from_text(
        self,
        prompt: str,
        output_format: str = "gaussian",
        **kwargs,
    ) -> TensorDict:
        """Generate 3D assets from a text prompt.

        Args:
            prompt: The text prompt driving generation (the primary input;
                previously this was undeclared and silently absorbed by
                ``**kwargs``).
        """

    @abstractmethod
    def generate_from_image(
        self,
        image: torch.Tensor,
        output_format: str = "gaussian",
        **kwargs,
    ) -> TensorDict:
        """Generate 3D assets from 2D images."""

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported output formats."""


class IMedical3DGenerator(ABC):
    """Interface for medical 3D generation services (X-Diffusion)."""

    @abstractmethod
    def generate_mri_volume(
        self,
        condition_image: torch.Tensor,
        target_resolution: tuple[int, int, int],
        conditioning_axis: int = 2,
        **kwargs,
    ) -> torch.Tensor:
        """Generate 3D MRI volumes from 2D conditioning images."""

    @abstractmethod
    def reconstruct_3d_from_2d(
        self,
        input_2d: torch.Tensor,
        reconstruction_params: dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Reconstruct 3D volumes from 2D inputs."""

    @abstractmethod
    def get_supported_datasets(self) -> list[str]:
        """Get list of supported medical datasets."""


class IManifestService(ABC):
    """Interface for manifest management and tracking."""

    @abstractmethod
    def create_manifest(
        self,
        dataset_name: str,
        processing_tasks: list[str],
        artifacts: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a manifest for a dataset processing operation."""

    @abstractmethod
    def save_manifest(
        self,
        manifest: dict[str, Any],
        output_path: str,
    ) -> str:
        """Save a manifest to file."""

    @abstractmethod
    def load_manifest(self, manifest_path: str) -> dict[str, Any]:
        """Load a manifest from file."""

    @abstractmethod
    def list_manifests(self, dataset_name: str | None = None) -> list[str]:
        """List available manifests, optionally filtered by dataset."""

    @abstractmethod
    def validate_manifest(self, manifest: dict[str, Any]) -> bool:
        """Validate manifest structure and content."""
