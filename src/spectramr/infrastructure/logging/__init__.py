"""Logging Infrastructure Package.

Canonical import path for all logging functionality. Re-exports from
their current source locations to provide a unified public API:

- ``LoggingService`` — structured console + file logging
- ``MetricsTracker`` — CSV metrics, validation image saving (``IMetricsService``)
- ``sinks`` — the run's sink registry (file, TensorBoard, metric and loss CSVs)
- ``LoggingServiceFactory`` — factory for config-driven logger creation

Usage::

    from spectramr.infrastructure.logging import LoggingService, MetricsTracker

Note: Uses lazy ``__getattr__`` to avoid circular imports with
``spectramr.infrastructure.services.__init__`` which also re-exports LoggingService.
"""

from .banners import banner, phase  # re-exported via __all__
from .progress import make_progress  # re-exported via __all__
from .provenance import (
    collect_run_provenance,
    count_parameters,
    cpu_resources,
    format_provenance_lines,
    git_provenance,
    gpu_resources,
    log_provenance,
    memory_resources,
    node_resources,
    runtime_environment,
    torch_runtime,
)


def __getattr__(name: str):
    """Lazy imports to break circular dependency with services.__init__."""
    if name in ("LoggingService", "LoggingServiceFactory"):
        from spectramr.infrastructure.services.logging_service import (
            LoggingService,
            LoggingServiceFactory,
        )

        return {"LoggingService": LoggingService, "LoggingServiceFactory": LoggingServiceFactory}[name]

    if name == "MetricsTracker":
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        return MetricsTracker

    if name == "ColoredConsoleFormatter":
        from spectramr.infrastructure.services.logging_service import (
            ColoredConsoleFormatter,
        )

        return ColoredConsoleFormatter

    if name == "bootstrap_console_logging":
        try:
            from spectramr.infrastructure.services.logging_service import (
                bootstrap_console_logging,
            )
        except ImportError as exc:
            # A failing transitive import (e.g. a torch version mismatch) would
            # otherwise surface as "cannot import name bootstrap_console_logging".
            raise ImportError(
                f"bootstrap_console_logging unavailable: logging_service failed to import ({exc})"
            ) from exc

        return bootstrap_console_logging

    raise AttributeError(f"module 'spectramr.infrastructure.logging' has no attribute {name!r}")


# A SINGLE ``__all__`` — the union of the eagerly imported re-exports (banners /
# progress / provenance) and the lazily exposed services (resolved through
# ``__getattr__`` above). There used to be TWO ``__all__`` assignments; the
# second silently clobbered the first, dropping every eager name from
# ``from ... import *`` (#6 of the 2026-06-19 logging-duplication audit).
__all__ = [
    "ColoredConsoleFormatter",
    "LoggingService",
    "LoggingServiceFactory",
    "MetricsTracker",
    "banner",
    "bootstrap_console_logging",
    "collect_run_provenance",
    "count_parameters",
    "cpu_resources",
    "format_provenance_lines",
    "git_provenance",
    "gpu_resources",
    "log_provenance",
    "make_progress",
    "memory_resources",
    "node_resources",
    "phase",
    "runtime_environment",
    "torch_runtime",
]
