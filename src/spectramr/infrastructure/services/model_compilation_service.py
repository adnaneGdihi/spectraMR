"""Model Compilation Service

.. deprecated:: 2026-07-01
    **DORMANT — not wired into the training path.** The
    ``IModelCompilationService`` DI registration was deliberately removed
    (see ``bootstrap.py``: "register back if a consumer appears") and no
    training consumer exists (review 2026-07-01). The LIVE ``torch.compile``
    path is ``apply_compile``
    (``infrastructure/training/builders/compile_apply.py``), invoked by the
    director at the point ``builders/compile_placement`` names for the arm's
    parallel strategy and gated by the ``optimization.compile.enabled`` YAML
    knobs. Constructing ``ModelCompilationService`` emits a
    ``DeprecationWarning``.

Provides PyTorch 2.0+ torch.compile support for training optimization.
Implements IModelCompilationService interface with error handling and statistics.

Author: spectraMR Research Team
"""

import logging
import warnings
from typing import Any

import torch

from spectramr.infrastructure.di.di_container import resolve_service
from spectramr.infrastructure.services import ILoggingService, IModelCompilationService

logger = logging.getLogger(__name__)


class ModelCompilationService(IModelCompilationService):
    """Service for compiling models using torch.compile.

    Provides safe compilation with error recovery, fallback to eager mode,
    and comprehensive statistics tracking.
    """

    def __init__(self, logging_service: ILoggingService | None = None):
        """Initialize the model compilation service.

        Args:
            logging_service: Optional logging service for diagnostics
        """
        warnings.warn(
            "ModelCompilationService is dormant (removed from DI; no training-"
            "path consumer). The live path is builders/compile_apply.apply_compile, "
            "placed per parallel strategy by builders/compile_placement and driven by "
            "the optimization.compile.enabled YAML knob.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.logging_service = logging_service or resolve_service(ILoggingService)
        self._compilation_stats: dict[str, Any] = {
            "total_compiled": 0,
            "compilation_failures": 0,
            "compilation_warnings": 0,
            "compiled_models": [],
        }
        self._compiled_models: dict[int, dict[str, Any]] = {}

    def is_available(self) -> bool:
        """Check if torch.compile is available.

        Returns:
            True if torch.compile is available, False otherwise
        """
        try:
            # torch.compile was introduced in PyTorch 2.0
            return hasattr(torch, "compile") and callable(torch.compile)
        except (AttributeError, Exception):
            return False

    def compile_model(
        self,
        model: torch.nn.Module,
        mode: str = "default",
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> torch.nn.Module:
        """Compile a model using torch.compile with error recovery.

        Args:
            model: The model to compile
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            backend: Compilation backend ('inductor', 'cudagraph', etc.)
            fullgraph: Whether to compile the full graph
            dynamic: Allow dynamic shapes in compiled graph

        Returns:
            Compiled model on success, or original model on failure
        """
        if not self.is_available():
            self.logging_service.log_warning(
                "torch.compile not available (requires PyTorch 2.0+). Returning original model."
            )
            return model

        model_id = id(model)

        try:
            # Enable error suppression to fall back to eager mode on compilation failures
            # This prevents crashes due to backend incompatibilities (e.g., Triton version issues)
            torch._dynamo.config.suppress_errors = True

            # Validate compilation mode
            if mode not in ("default", "reduce-overhead", "max-autotune"):
                self.logging_service.log_warning(
                    f"Invalid compilation mode '{mode}'. Using 'default'."
                )
                mode = "default"

            # Log compilation intent
            self.logging_service.log_info(
                f"Compiling model {model.__class__.__name__} "
                f"(mode={mode}, backend={backend}, dynamic={dynamic})"
            )

            # Prepare compilation kwargs
            compile_kwargs: dict[str, Any] = {
                "mode": mode,
                "backend": backend,
                "fullgraph": fullgraph,
                "dynamic": dynamic,
            }

            # Attempt compilation
            compiled_model = torch.compile(model, **compile_kwargs)

            # Record success
            self._compilation_stats["total_compiled"] += 1
            self._compiled_models[model_id] = {
                "model_class": model.__class__.__name__,
                "mode": mode,
                "backend": backend,
                "fullgraph": fullgraph,
                "dynamic": dynamic,
                "status": "success",
            }
            self._compilation_stats["compiled_models"].append(model.__class__.__name__)

            self.logging_service.log_info(f"Successfully compiled {model.__class__.__name__}")

            return compiled_model

        except RuntimeError as e:
            # Handle runtime errors (incompatible operations, etc.)
            self._handle_compilation_error(
                model,
                e,
                "RuntimeError",
                model_id,
            )
            return model

        except TypeError as e:
            # Handle type errors (invalid arguments, etc.)
            self._handle_compilation_error(
                model,
                e,
                "TypeError",
                model_id,
            )
            return model

        except Exception as e:
            # Handle unexpected errors
            self._handle_compilation_error(
                model,
                e,
                type(e).__name__,
                model_id,
            )
            return model

    def _handle_compilation_error(
        self,
        model: torch.nn.Module,
        error: Exception,
        error_type: str,
        model_id: int,
    ) -> None:
        """Handle compilation errors gracefully.

        Args:
            model: The model that failed compilation
            error: The exception that occurred
            error_type: String name of the exception type
            model_id: ID of the model
        """
        self._compilation_stats["compilation_failures"] += 1
        self._compiled_models[model_id] = {
            "model_class": model.__class__.__name__,
            "status": "failed",
            "error_type": error_type,
            "error_message": str(error),
        }

        # Log the error with full context
        error_message = (
            f"torch.compile failed for {model.__class__.__name__}: "
            f"{error_type}: {error}. "
            "Model will run in eager mode."
        )
        self.logging_service.log_warning(error_message)

    def get_compilation_stats(self) -> dict[str, Any]:
        """Get compilation statistics.

        Returns:
            Dictionary containing compilation statistics
        """
        return {
            "total_compiled": self._compilation_stats["total_compiled"],
            "compilation_failures": self._compilation_stats["compilation_failures"],
            "success_rate": (
                self._compilation_stats["total_compiled"]
                / max(
                    1,
                    self._compilation_stats["total_compiled"]
                    + self._compilation_stats["compilation_failures"],
                )
            ),
            "compiled_model_types": list(set(self._compilation_stats["compiled_models"])),
            "compiled_models_count": len(self._compiled_models),
            "detailed_models": self._compiled_models,
        }

    def reset_stats(self) -> None:
        """Reset compilation statistics."""
        self._compilation_stats = {
            "total_compiled": 0,
            "compilation_failures": 0,
            "compilation_warnings": 0,
            "compiled_models": [],
        }
        self._compiled_models.clear()


class NoOpModelCompilationService(IModelCompilationService):
    """No-op implementation of model compilation service.

    Used when compilation is disabled or unavailable.
    """

    def __init__(self, logging_service: ILoggingService | None = None):
        """Initialize the no-op compilation service.

        Args:
            logging_service: Optional logging service
        """
        self.logging_service = logging_service or resolve_service(ILoggingService)

    def compile_model(
        self,
        model: torch.nn.Module,
        mode: str = "default",
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> torch.nn.Module:
        """Return model unchanged (no-op).

        Args:
            model: The model to "compile"
            mode: Ignored
            backend: Ignored
            fullgraph: Ignored
            dynamic: Ignored

        Returns:
            Original model unchanged
        """
        return model

    def is_available(self) -> bool:
        """Always return False (disabled).

        Returns:
            False
        """
        return False

    def get_compilation_stats(self) -> dict[str, Any]:
        """Return empty statistics.

        Returns:
            Empty statistics dictionary
        """
        return {
            "total_compiled": 0,
            "compilation_failures": 0,
            "success_rate": 0.0,
            "compiled_model_types": [],
            "compiled_models_count": 0,
            "detailed_models": {},
        }

    def reset_stats(self) -> None:
        """No-op reset."""
        pass


if __name__ == "__main__":
    # Example usage
    from spectramr.models.factories.model_factory import get_model_factory

    service = ModelCompilationService()

    if service.is_available():
        logger.debug("torch.compile is available")

        # Create a simple model
        factory = get_model_factory()
        model = factory.create_generator("standard_unet", in_channels=1, out_channels=1)

        # Compile it
        compiled = service.compile_model(model, mode="reduce-overhead")
        logger.debug(f"Compiled model: {compiled}")

        # Print stats
        stats = service.get_compilation_stats()
        logger.debug(f"Stats: {stats}")
    else:
        logger.debug("torch.compile is not available (requires PyTorch 2.0+)")
