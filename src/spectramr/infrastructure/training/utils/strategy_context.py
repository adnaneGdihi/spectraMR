"""Strategy Context Module

This module provides the StrategyContext class that encapsulates
shared dependencies and configuration for training strategies,
following SOLID principles and dependency injection patterns.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch

# Import unified physics operator interface
from spectramr.infrastructure.physics.interfaces import IPhysicsOperator

# Use direct alias to protocol to avoid stub implementation
PhysicsOperator = IPhysicsOperator


def _coerce_device(device: Any) -> torch.device:
    """Convert various device representations to ``torch.device``.

    Strategy constructors frequently receive mocked training states in tests
    where ``state.device`` is a lightweight stub with a ``type`` attribute.
    This helper normalizes those inputs while keeping strict validation for
    unsupported types.
    """
    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        return torch.device(device)

    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return torch.device(device_type)

    raise ValueError("device must be a torch.device instance or string")


@dataclass
class StrategyContext:
    """Context object containing shared dependencies for training strategies.

    This class encapsulates device, configuration, and physics operations
    that are commonly needed across different training strategies,
    enabling dependency injection and better separation of concerns.
    """

    # Core dependencies
    device: torch.device
    config: Any  # TrainingConfig or similar

    # Utility services
    mask_generator: Any | None = None  # KSpaceMaskGenerator
    reconstruction_utils: Any | None = None  # ReconstructionUtilities

    # Performance settings
    use_amp: bool = False
    gradient_clip_norm: float | None = None

    # Additional utility services
    fft_transformer: Any | None = None  # FFTTransformer
    complex_handler: Any | None = None  # ComplexTensorHandler
    loss_fn: Any | None = None  # Loss function

    # Physics operations
    fft_ops: Any | None = None  # FFT operations utility
    physics_ops: dict[str, PhysicsOperator] | None = None

    # Logging and monitoring
    enable_logging: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate context after initialization."""
        self._validate_context()

    def _validate_context(self):
        """Validate that required dependencies are present."""
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device instance")

        if self.config is None:
            raise ValueError("config cannot be None")

        # Validate physics operations if provided
        if self.physics_ops is not None:
            for name, op in self.physics_ops.items():
                if not hasattr(op, "forward") or not hasattr(op, "adjoint"):
                    raise ValueError(
                        f"Physics operator '{name}' must implement forward and adjoint methods",
                    )

    def get_physics_operator(self, name: str) -> PhysicsOperator | None:
        """Get a physics operator by name.

        Args:
            name: Name of the physics operator

        Returns:
            Physics operator instance or None if not found

        """
        if self.physics_ops is None:
            return None
        return self.physics_ops.get(name)

    def has_fft_ops(self) -> bool:
        """Check if FFT operations are available."""
        return self.fft_ops is not None

    def has_mask_generator(self) -> bool:
        """Check if mask generator is available."""
        return self.mask_generator is not None

    def has_reconstruction_utils(self) -> bool:
        """Check if reconstruction utilities are available."""
        return self.reconstruction_utils is not None

    def get_device_memory_info(self) -> dict[str, Any]:
        """Get memory information for the current device.

        Returns:
            Dictionary with memory statistics

        """
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device),
                "total": torch.cuda.get_device_properties(
                    self.device,
                ).total_memory,
            }
        return {"allocated": 0, "reserved": 0, "total": 0}

    def log_context_info(self):
        """Log information about the current context."""
        if not self.enable_logging:
            return

        logger = logging.getLogger(__name__)
        memory_info = self.get_device_memory_info()

        logger.info("StrategyContext initialized")
        logger.debug(f"  Device: {self.device}")
        logger.debug(f"  Use AMP: {self.use_amp}")
        logger.debug(f"  Gradient clipping: {self.gradient_clip_norm}")
        logger.debug(f"  FFT ops available: {self.has_fft_ops()}")
        logger.debug(f"  Mask generator available: {self.has_mask_generator()}")
        logger.debug(f"  Reconstruction utils available: {self.has_reconstruction_utils()}")
        physics_ops = list(self.physics_ops.keys()) if self.physics_ops else "None"
        logger.debug(f"  Physics operators: {physics_ops}")

        if self.device.type == "cuda":
            allocated_gb = memory_info["allocated"] / 1024**3
            logger.debug(f"  GPU Memory - Allocated: {allocated_gb:.2f} GB")
            reserved_gb = memory_info["reserved"] / 1024**3
            logger.debug(f"  GPU Memory - Reserved: {reserved_gb:.2f} GB")
            total_gb = memory_info["total"] / 1024**3
            logger.debug(f"  GPU Memory - Total: {total_gb:.2f} GB")


def create_strategy_context(
    device: torch.device | str | Any,
    config: Any,
    fft_ops: Any | None = None,
    physics_ops: dict[str, PhysicsOperator] | None = None,
    mask_generator: Any | None = None,
    reconstruction_utils: Any | None = None,
    fft_transformer: Any | None = None,
    complex_handler: Any | None = None,
    loss_fn: Any | None = None,
    **kwargs,
) -> StrategyContext:
    """Factory function for creating StrategyContext instances.

    Args:
        device: PyTorch device
        config: Training configuration
        fft_ops: FFT operations utility
        physics_ops: Dictionary of physics operators
        mask_generator: K-space mask generator
        reconstruction_utils: Reconstruction utilities
        **kwargs: Additional context parameters

    Returns:
        Configured StrategyContext instance

    """
    coerced_device = _coerce_device(device)

    return StrategyContext(
        device=coerced_device,
        config=config,
        fft_ops=fft_ops,
        physics_ops=physics_ops,
        mask_generator=mask_generator,
        reconstruction_utils=reconstruction_utils,
        fft_transformer=fft_transformer,
        complex_handler=complex_handler,
        loss_fn=loss_fn,
        **kwargs,
    )
