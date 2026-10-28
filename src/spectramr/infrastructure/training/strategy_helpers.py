"""Strategy Helpers Module

This module contains helper classes for initializing and managing training strategies.
It serves as a shared utility to reduce code duplication across different strategy implementations.
"""

import contextlib
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model


class StrategyInitializationHelper:
    """Helper class for strategy initialization and common tasks."""

    @staticmethod
    def initialize_loss_computer(strategy: Any, loss_computer_cls: type) -> None:
        """Initialize loss computer for the strategy.

        SSOT Principle: Ensure context provides loss_function dict before initializing.
        The losses dict may be provided via:
        - TrainingEnvironment.loss_function property (aliases losses)
        - TrainingState.loss_function attribute
        - Direct hasattr check for backward compatibility

        Args:
            strategy: The training strategy instance
            loss_computer_cls: The class of the loss computer to instantiate
        """
        context = getattr(strategy, "env", None) or getattr(strategy, "state", None)
        if context is None:
            raise AttributeError(
                "Strategy must have 'env' or 'state' attribute to initialize loss computer"
            )

        # Verify context has loss_function available (via property or attribute)
        if not hasattr(context, "loss_function"):
            raise AttributeError(
                f"Context must have 'loss_function' accessible. "
                f"Got context type: {type(context).__name__}, "
                f"Available attributes: {[a for a in dir(context) if not a.startswith('_')]}"
            )

        # Instantiate loss computer with strategy context
        strategy.loss_computer = loss_computer_cls(context)

        # Log initialization if logging service available
        if hasattr(strategy, "logging_service"):
            strategy.logging_service.log_info(
                f"Initialized {loss_computer_cls.__name__}",
                model_type=getattr(context, "model_type", "unknown"),
            )

    @staticmethod
    def initialize_profiling_service(strategy: Any, fallback_enabled: bool = False) -> None:
        """Initialize profiling service interactions.

        Args:
            strategy: The training strategy instance
            fallback_enabled: Whether to enable fallback mechanisms
        """
        # Wire to CUDA NVTX profiling if available for nsight/nvprof visibility
        strategy._profiling_enabled = False
        if torch.cuda.is_available():
            strategy._profiling_enabled = True
        if hasattr(strategy, "logging_service"):
            strategy.logging_service.log_info(
                f"Profiling service initialized (NVTX={'enabled' if getattr(strategy, '_profiling_enabled', False) else 'disabled'}, "
                f"fallback={fallback_enabled})",
                model_type=getattr(
                    getattr(strategy, "env", getattr(strategy, "state", None)),
                    "model_type",
                    "unknown",
                ),
            )

    @staticmethod
    @contextlib.contextmanager
    def create_profiling_context(strategy: Any, name: str, **metadata: Any):
        """Create a profiling context for a specific operation.

        Args:
            strategy: The training strategy instance
            name: Name of the operation to profile
            **metadata: Additional metadata for the profile
        """
        # Use CUDA NVTX ranges for GPU profiler integration (nsight, nvprof)
        use_nvtx = torch.cuda.is_available() and getattr(strategy, "_profiling_enabled", False)

        if use_nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()

    @staticmethod
    def initialize_data_consistency(
        strategy: Any,
        use_dc: bool = True,
        dc_weight: float = 1.0,
        dc_method: str = "simple",
    ) -> None:
        """Initialize data consistency layer for physics-aware reconstruction.

        [PHASE 2] This method now prioritizes model-integrated DC layers to
        prevent double application and support learnable parameters (e.g. SoftDC).

        Args:
            strategy: The training strategy instance
            use_dc: Whether to enable data consistency
            dc_weight: Weight for data consistency term (legacy/fallback)
            dc_method: Method identifier (legacy/fallback)
        """
        if not use_dc:
            strategy.dc_layer = None
            strategy.use_dc = False
            return

        # 1. Prioritize model-integrated DC
        # unwrap_model, not one ``.module`` hop: a compiled-then-DDP model nests
        # as DDP(OptimizedModule), and the DC layer sits on the innermost module.
        gen = unwrap_model(getattr(strategy, "generator_model", None))

        if hasattr(gen, "dc_layer") and gen.dc_layer is not None:
            strategy.dc_layer = gen.dc_layer
            strategy.use_dc = True
            if hasattr(strategy, "logging_service"):
                strategy.logging_service.log_info(
                    f"🧲 Strategy DC: Reusing model-integrated {type(strategy.dc_layer).__name__}"
                )
            return

        # 2. Skip strategy-side instantiation if not found in model (Architectural SSOT)
        strategy.dc_layer = None
        strategy.use_dc = False

        if hasattr(strategy, "logging_service"):
            strategy.logging_service.log_warning(
                "🧲 Strategy DC: Enabled in config but NOT found in generator model. "
                "Skipping strategy-side instantiation to maintain SSOT."
            )

    @staticmethod
    def initialize_metrics_adapter(strategy: Any, adapter_cls: type) -> None:
        """Initialize a metrics adapter for the strategy.

        Args:
            strategy: The training strategy instance
            adapter_cls: The class of the metrics adapter
        """
        try:
            strategy.metrics_adapter = adapter_cls()
        except Exception:
            strategy.metrics_adapter = None
