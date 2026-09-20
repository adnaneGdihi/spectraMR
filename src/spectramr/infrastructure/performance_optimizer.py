"""Performance Optimization Module

.. deprecated:: 2026-07-01
    **DORMANT — not wired into the training path.** Nothing in
    ``bootstrap.py`` / the DI container / ``pipelines/`` consumes this
    module (review 2026-07-01: consumers are its own tests only). The LIVE
    ``torch.compile`` path is ``ModelBuilder.compile()``
    (``infrastructure/training/builders/model_builder.py``), gated by the
    ``optimization.compile.enabled`` / ``compile_mode`` / ``compile_backend``
    / ``compile_fullgraph`` / ``compile_dynamic`` YAML knobs. Do NOT adopt
    this module as-is: its ``optimize_model_for_inference`` falls back to
    eager on compile failure *silently* (pitfall #9).
    ``get_performance_optimizer()`` emits a ``DeprecationWarning``.

This module provides comprehensive performance optimizations
for MRI reconstruction:
- Automatic Mixed Precision (AMP) training
- Gradient checkpointing for memory efficiency
- Memory monitoring and optimization
- Auto batch size tuning
- Performance profiling and reporting

Author: spectraMR Research Team
"""

import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
import psutil
import torch
from torch import amp, nn

from spectramr.core.metrics.performance import PerformanceMetrics

# from spectramr.domain.entities.path_setup import setup_python_path
from spectramr.infrastructure.di.di_container import resolve_service
from spectramr.infrastructure.services import ILoggingService

logger = logging.getLogger(__name__)


# Ensure project paths are set up
# setup_python_path(add_project_root=True, add_src=True)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    system_memory_percent: float
    gpu_memory_percent: float


class PerformanceOptimizer:
    """Main performance optimization class."""

    def __init__(self, config: dict[str, Any] | None = None):
        """__init__.

        Args:
            config (Optional[dict[str, Any]]): Description.
        """
        self.config = config or {}
        self.logging_service = resolve_service(ILoggingService)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pass an explicit device to GradScaler so it does not emit the
        # "CUDA not available, disabling" UserWarning on CPU-only test runs
        # where pyproject.toml promotes warnings to errors.
        self.scaler = amp.GradScaler(self.device.type) if self._amp_enabled() else None

        # Performance tracking
        self.metrics_history = []
        self.memory_history = []

    def _amp_enabled(self) -> bool:
        """Check if AMP is enabled in config."""
        return self.config.get("amp", {}).get("enabled", True)

    def _gradient_checkpointing_enabled(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return self.config.get("gradient_checkpointing", {}).get("enabled", False)

    def _memory_optimization_enabled(self) -> bool:
        """Check if memory optimization is enabled."""
        return self.config.get("memory_optimization", {}).get("enabled", True)

    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.scaler is not None:
            with torch.amp.autocast(self.device.type):
                yield
        else:
            yield

    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage."""
        if not self._gradient_checkpointing_enabled():
            return model

        # Apply to specific layer types
        def apply_checkpointing(module):
            """apply_checkpointing.

            Args:
                module (Any): Description.
            Returns:
                Any: Description.
            """
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                for child in module.children():
                    apply_checkpointing(child)
            elif hasattr(module, "apply_checkpointing"):
                module.apply_checkpointing()

        apply_checkpointing(model)
        self.logging_service.log_info("Applied gradient checkpointing to model")
        return model

    def optimize_memory_usage(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to the model."""
        if not self._memory_optimization_enabled():
            return model

        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Set memory format for Conv2d layers
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.to(memory_format=torch.channels_last)

        self.logging_service.log_info("Applied memory optimizations")
        return model

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

            # GPU utilization
            if GPUTIL_AVAILABLE:
                try:
                    gpu = GPUtil.getGPUs()[0]
                    gpu_percent = gpu.memoryUtil * 100
                except (IndexError, Exception):
                    gpu_percent = 0.0
            else:
                gpu_percent = 0.0
        else:
            allocated = reserved = peak = gpu_percent = 0.0

        # System memory
        system_memory = psutil.virtual_memory()
        system_percent = system_memory.percent

        stats = MemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            peak_mb=peak,
            system_memory_percent=system_percent,
            gpu_memory_percent=gpu_percent,
        )

        self.memory_history.append(stats)
        return stats

    def profile_model(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_runs: int = 10,
    ) -> PerformanceMetrics:
        """Profile model performance."""
        model = model.to(self.device).eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Warmup
        dummy_input = torch.randn(input_shape).to(self.device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)

        # Profile forward pass
        forward_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()

                _ = model(dummy_input)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                forward_times.append(time.time() - start)

        # Profile forward + backward
        backward_times = []
        for _ in range(num_runs):
            input_tensor = torch.randn(input_shape).to(self.device)
            input_tensor.requires_grad_(True)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            output = model(input_tensor)
            loss = output.mean()
            loss.backward()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            backward_times.append(time.time() - start)

        # Memory stats
        memory_stats = self.get_memory_stats()

        # GPU utilization
        if GPUTIL_AVAILABLE:
            try:
                gpu = GPUtil.getGPUs()[0]
                gpu_util = gpu.load * 100
            except (IndexError, Exception):
                gpu_util = 0.0
        else:
            gpu_util = 0.0

        # Throughput calculation
        batch_size = input_shape[0]
        avg_forward_time = sum(forward_times) / len(forward_times)
        throughput = batch_size / avg_forward_time

        metrics = PerformanceMetrics(
            forward_time=sum(forward_times) / len(forward_times),
            backward_time=sum(backward_times) / len(backward_times),
            memory_peak=memory_stats.peak_mb,
            memory_allocated=memory_stats.allocated_mb,
            gpu_utilization=gpu_util,
            throughput=throughput,
        )

        self.metrics_history.append(metrics)
        return metrics

    def auto_tune_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple,
        max_batch_size: int = 64,
        target_memory_percent: float = 80.0,
    ) -> int:
        """Automatically find optimal batch size based on memory
        constraints.
        """
        if not torch.cuda.is_available():
            return max_batch_size

        model = model.to(self.device).eval()
        optimal_batch_size = 1

        for batch_size in range(1, max_batch_size + 1, 2):
            try:
                # Test with current batch size
                test_shape = (batch_size,) + input_shape[1:]
                dummy_input = torch.randn(test_shape).to(self.device)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                with torch.no_grad():
                    _ = model(dummy_input)

                # Check memory usage
                memory_stats = self.get_memory_stats()
                if memory_stats.gpu_memory_percent > target_memory_percent:
                    break

                optimal_batch_size = batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                raise e

        self.logging_service.log_info(f"Auto-tuned batch size: {optimal_batch_size}")
        return optimal_batch_size

    # ``create_optimized_data_loader`` was deleted here (data-layer audit D22,
    # 2026-08-05). It had ZERO callers anywhere -- src, tests, scripts, runners.
    # The one apparent caller, ``scripts/hpo/performance_optimization_experiment
    # .py``, defines its OWN method of the same name and does not inherit this
    # class, so ``self.create_optimized_data_loader`` there never resolved here.
    #
    # It was deleted rather than rerouted through the leaf ``DataLoaderBuilder``
    # because it was a DIVERGENT sibling, not an unfinished capability: it read a
    # plain ``dict`` config instead of the frozen SSOT, hardcoded
    # ``shuffle=True`` (wrong for any val/inference use), and supplied neither a
    # collate strategy nor ``worker_init_fn`` seeding. Wiring it would have
    # forked loader construction a fourth way; see the module deprecation note
    # above for why nothing here should be adopted as-is.

    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply inference-specific optimizations."""
        model = model.eval()

        # Fuse operations where possible
        if hasattr(torch, "compile"):
            try:
                # Enable error suppression to fall back to eager mode on compilation failures
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model)
                self.logging_service.log_info("Applied torch.compile optimization")
            except (RuntimeError, TypeError) as e:
                self.logging_service.log_warning(f"torch.compile failed: {e}")

        # Convert to half precision if AMP is enabled
        if self._amp_enabled() and torch.cuda.is_available():
            model = model.half()

        return model

    def create_performance_report(self, output_dir: Path) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate metrics
        if self.metrics_history:
            avg_metrics = {
                "forward_time": (
                    sum(m.forward_time for m in self.metrics_history) / len(self.metrics_history)
                ),
                "backward_time": (
                    sum(m.backward_time for m in self.metrics_history) / len(self.metrics_history)
                ),
                "memory_peak": max(m.memory_peak for m in self.metrics_history),
                "throughput": (
                    sum(m.throughput for m in self.metrics_history) / len(self.metrics_history)
                ),
            }
        else:
            avg_metrics = {}

        # Memory analysis
        if self.memory_history:
            memory_analysis = {
                "peak_memory_mb": max(m.peak_mb for m in self.memory_history),
                "avg_allocated_mb": (
                    sum(m.allocated_mb for m in self.memory_history) / len(self.memory_history)
                ),
                "system_memory_percent": max(m.system_memory_percent for m in self.memory_history),
                "gpu_memory_percent": max(m.gpu_memory_percent for m in self.memory_history),
            }
        else:
            memory_analysis = {}

        report = {
            "timestamp": time.time(),
            "config": self.config,
            "performance_metrics": avg_metrics,
            "memory_analysis": memory_analysis,
            "optimizations_applied": {
                "amp": self._amp_enabled(),
                "gradient_checkpointing": (self._gradient_checkpointing_enabled()),
                "memory_optimization": self._memory_optimization_enabled(),
            },
            "device_info": {
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": (torch.cuda.get_device_name() if torch.cuda.is_available() else None),
            },
        }

        # Save report
        report_path = output_dir / "performance_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logging_service.log_info(f"Performance report saved to {report_path}")
        return report


class MemoryEfficientTrainer:
    """Memory-efficient training wrapper."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        performance_optimizer: PerformanceOptimizer,
    ):
        """__init__.

        Args:
            optimizer (torch.optim.Optimizer): Description.
            performance_optimizer (PerformanceOptimizer): Description.
        """
        self.optimizer = optimizer
        self.perf_optimizer = performance_optimizer

    def training_step(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Perform optimized training step."""
        self.optimizer.zero_grad(set_to_none=True)

        with self.perf_optimizer.autocast_context():
            output = model(batch["lr"])
            loss = loss_fn(output, batch["hr"])

        if self.perf_optimizer.scaler is not None:
            # AMP backward pass
            self.perf_optimizer.scaler.scale(loss).backward()
            self.perf_optimizer.scaler.step(self.optimizer)
            self.perf_optimizer.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}


def performance_monitor(func: Callable) -> Callable:
    """Decorator for monitoring function performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper.

        Returns:
            Any: Description.
        """
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            logging.info(
                f"{func.__name__}: {execution_time:.4f}s, "
                f"Memory delta: {memory_delta / 1024**2:.2f}MB",
            )

    return wrapper


# Global performance optimizer instance
_performance_optimizer = None


def get_performance_optimizer(
    config: dict[str, Any] | None = None,
) -> PerformanceOptimizer:
    """Get global performance optimizer instance.

    .. deprecated:: 2026-07-01
        Dormant module — see the module docstring. The live torch.compile
        path is ``builders/compile_apply.apply_compile``
        (``optimization.compile.enabled``).
    """
    import warnings

    warnings.warn(
        "spectramr.infrastructure.performance_optimizer is dormant (no training-"
        "path consumer). The live path is builders/compile_apply.apply_compile, "
        "placed per parallel strategy by builders/compile_placement and driven by "
        "the optimization.compile.enabled YAML knob.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(config)
    return _performance_optimizer


if __name__ == "__main__":
    # Example usage
    config = {
        "amp": {"enabled": True},
        "gradient_checkpointing": {"enabled": True},
        "memory_optimization": {"enabled": True},
    }

    optimizer = get_performance_optimizer(config)

    # Example model profiling
    from spectramr.models.factories.model_factory import get_model_factory

    factory = get_model_factory()
    model = factory.create_generator("standard_unet", in_channels=1, out_channels=1)

    # Cast to nn.Module for profiling (generators inherit from nn.Module)
    model_to_profile = model  # type: ignore[arg-type]

    metrics = optimizer.profile_model(model_to_profile, (1, 1, 256, 256))
    logger.debug("Model profiling results:")
    logger.debug(f"Forward time: {metrics.forward_time:.4f}s")
    logger.debug(f"Peak memory: {metrics.memory_peak:.2f}MB")
    logger.debug(f"Throughput: {metrics.throughput:.2f} samples/s")
