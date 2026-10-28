__all__ = [
    "ABC",
    "Any",
    "Callable",
    "IConfigurationService",
    "IDeviceService",
    "IErrorHandlingService",
    "ILoggingService",
    "IMetricsService",
    "IModelCompilationService",
    "IModelFactory",
    "IModelManagementService",
    "IPerformanceMonitor",
    "IProfilingService",
    "IServiceRegistry",
    "LoggingService",
    "Optional",
    "Path",
    "Union",
    "abstractmethod",
    "torch",
]

"""Core Service Interfaces and Implementations
=========================================

This module defines core interfaces and exports concrete implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union

import torch

# Import interfaces from canonical location to avoid duplicate definition
# This fixes Sphinx cross-reference warning [ref.python]
from spectramr.domain.interfaces.service_interfaces import ILoggingService, IMetricsService

from .logging_service import LoggingService

# Add others as needed or rely on direct imports.


class IDeviceService(ABC):
    """Interface for device management."""

    @abstractmethod
    def get_device(self) -> torch.device:
        """Gets the current device."""

    @abstractmethod
    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves tensor to device."""

    @abstractmethod
    def is_cuda_available(self) -> bool:
        """Checks if CUDA is available."""


class IPerformanceMonitor(ABC):
    """Interface for performance monitoring."""

    @abstractmethod
    def start_monitoring(self) -> None:
        """Starts performance monitoring."""

    @abstractmethod
    def stop_monitoring(self) -> dict[str, Any]:
        """Stops performance monitoring and returns metrics."""

    @abstractmethod
    def get_current_metrics(self) -> dict[str, Any]:
        """Gets current performance metrics."""


class IServiceRegistry(ABC):
    """Interface for service registry."""

    @abstractmethod
    def register_service(self, name: str, service: Any) -> None:
        """Registers a service."""

    @abstractmethod
    def get_service(self, name: str) -> Any:
        """Gets a registered service."""

    @abstractmethod
    def has_service(self, name: str) -> bool:
        """Checks if service is registered."""


class IModelManagementService(ABC):
    """Interface for comprehensive model management services."""

    @abstractmethod
    def register_model(
        self,
        model: torch.nn.Module,
        model_type: str,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a model with metadata and versioning."""

    @abstractmethod
    def get_model(self, model_id: str) -> torch.nn.Module | None:
        """Retrieve a model by ID."""

    @abstractmethod
    def list_models(
        self,
        model_type: str | None = None,
        version: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """list models with optional filtering."""

    @abstractmethod
    def update_model_metadata(self, model_id: str, metadata: dict[str, Any]) -> bool:
        """Update model metadata."""

    @abstractmethod
    def export_model(
        self,
        model_id: str,
        export_format: str,
        export_path: str | None = None,
    ) -> str:
        """Export model in specified format."""

    @abstractmethod
    def create_model_card(
        self,
        model_id: str,
        training_details: dict[str, Any],
        performance_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Create comprehensive model card."""

    @abstractmethod
    def get_model_card(self, model_id: str) -> Any | None:
        """Retrieve model card."""

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete model and all associated artifacts."""


class IErrorHandlingService(ABC):
    """Interface for comprehensive error handling and recovery services."""

    @abstractmethod
    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "medium",
    ) -> dict[str, Any]:
        """Handle an error with appropriate recovery actions."""

    @abstractmethod
    def register_error_handler(
        self,
        error_type: type,
        handler: Callable[[Exception, dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a custom error handler for specific error types."""

    @abstractmethod
    def get_error_statistics(self) -> dict[str, Any]:
        """Get statistics about handled errors."""

    @abstractmethod
    def create_error_context(
        self,
        operation: str,
        component: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized error context."""

    @abstractmethod
    def attempt_recovery(
        self,
        error: Exception,
        _recovery_strategy: str,
        context: dict[str, Any],
    ) -> bool:
        """Attempt to recover from an error using specified strategy."""


class IConfigurationService(ABC):
    """Interface for configuration management services."""

    @abstractmethod
    def load_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Load configuration from dictionary."""

    @abstractmethod
    def validate_all(self) -> list[str]:
        """Validate all configuration sections."""

    @abstractmethod
    def get_all_defaults(self) -> dict[str, dict[str, Any]]:
        """Get default values for all sections."""

    @abstractmethod
    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert configuration to dictionary."""

    @abstractmethod
    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file."""

    @abstractmethod
    def to_flat_dict(self) -> dict[str, Any]:
        """Convert configuration to flat dictionary (legacy format)."""

    @classmethod
    @abstractmethod
    def from_flat_dict(cls, _flat_dict: dict[str, Any]) -> "IConfigurationService":
        """Load configuration from flat dictionary (legacy format)."""


class IModelFactory(ABC):
    """Interface for model factory services."""

    @abstractmethod
    def create_model(
        self,
        model_type: str,
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs,
    ) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
        """Creates generator and discriminator models for given type."""

    @abstractmethod
    def list_available_models(self) -> list[str]:
        """Lists available model types."""


class IProfilingService(ABC):
    """Interface for a profiling service."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether profiling is enabled."""

    @abstractmethod
    def enable(self) -> None:
        """Enable profiling."""

    @abstractmethod
    def disable(self) -> None:
        """Disable profiling."""

    @abstractmethod
    def reset(self) -> None:
        """Reset profiling data."""

    @abstractmethod
    def profile(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Context manager for profiling a code block.

        Args:
            name: Name of the profiling event
            metadata: Optional metadata for the event

        Yields:
            None
        """

    @abstractmethod
    def record_event(
        self,
        name: str,
        *,
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a profiling event manually.

        Args:
            name: Name of the event
            duration_s: Duration in seconds
            metadata: Optional metadata
        """

    @abstractmethod
    def snapshot(self) -> list[Any]:
        """Get a snapshot of profiling events.

        Returns:
            List of profiling events
        """

    @abstractmethod
    def flush_metrics(self, *, prefix: str = "profile") -> dict[str, float]:
        """Flush profiling metrics.

        Args:
            prefix: Prefix for metric keys

        Returns:
            Dictionary of profiling metrics
        """

    @abstractmethod
    def extend_metrics(
        self,
        target: dict[str, float],
        *,
        prefix: str = "profile",
    ) -> dict[str, float]:
        """Extend target dictionary with profiling metrics.

        Args:
            target: Dictionary to extend
            prefix: Prefix for metric keys

        Returns:
            Extended dictionary
        """

    @abstractmethod
    def start_profiling(self) -> None:
        """Start profiling session."""

    @abstractmethod
    def end_profiling(self, name: str) -> dict[str, Any]:
        """End profiling session and return metrics.

        Args:
            name: Name of the profiling session

        Returns:
            Dictionary with profiling metrics
        """


class IModelCompilationService(ABC):
    """Interface for model compilation optimization service.

    Provides PyTorch 2.0+ torch.compile support for training optimization.
    Handles compilation configuration, error recovery, and fallback to eager mode.
    """

    @abstractmethod
    def compile_model(
        self,
        model: torch.nn.Module,
        mode: str = "default",
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> torch.nn.Module:
        """Compile a model using torch.compile with configuration.

        Args:
            model: The model to compile
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            backend: Compilation backend ('inductor', 'cudagraph', etc.)
            fullgraph: Whether to compile the full graph (requires no dynamic shapes)
            dynamic: Allow dynamic shapes in the compiled graph

        Returns:
            Compiled model or original model if compilation fails/disabled
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if torch.compile is available in current environment."""

    @abstractmethod
    def get_compilation_stats(self) -> dict[str, Any]:
        """Get statistics about compiled models."""

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset compilation statistics."""
