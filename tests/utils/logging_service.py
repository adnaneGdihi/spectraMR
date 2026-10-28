"""One way for tests to build the logging service the container would hold."""

from __future__ import annotations

from spectramr.infrastructure.services.logging_service import LoggingService


def make_logging_service(log_dir: str) -> LoggingService:
    """A ``GANTraining`` service set up on ``log_dir``, as ``bootstrap`` builds one."""
    service = LoggingService(logger_name="GANTraining")
    service.setup(log_dir, "INFO")
    return service
