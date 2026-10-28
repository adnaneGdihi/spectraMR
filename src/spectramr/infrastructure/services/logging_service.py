#!/usr/bin/env python3
"""Logging Service for spectraMR project.

This module provides SOLID-compliant logging services including issue tracking,
CSV logging, and model-specific logging with proper dependency injection.

.. mermaid::

    sequenceDiagram
        participant Client
        participant Logger
        participant Throttle
        participant Handlers
        participant Fallback

        Client->>Logger: log(level, msg)
        Logger->>Throttle: check_limit()
        alt Throttled
            Throttle-->>Client: return
        else Allowed
            Logger->>Logger: unify_metadata()
            try
                Logger->>Handlers: emit(record)
            catch Error
                Logger->>Fallback: record_failure()
            end
        end
"""

import logging as std_logging
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from spectramr.config.schemas.logging import LoggingConfigSchema

import json

import torch

from spectramr.domain.interfaces.service_interfaces import ILoggingService
from spectramr.infrastructure.logging.sinks import (
    ArtifactSink,
    FileSink,
    SinkRegistry,
    TensorBoardSink,
    resolve_sinks_dir,
)
from spectramr.infrastructure.logging.console import (
    ColoredConsoleFormatter,  # noqa: F401  (re-exported)
    install_or_upgrade_colored_console as _install_or_upgrade_colored_console,
)
from spectramr.infrastructure.logging.log_policy import (  # noqa: F401  (re-exported)
    adopt_logging_policy,
    bootstrap_console_logging,
    console_stream,
    effective_level,
    rank_log_path,
)


class JSONFormatter(std_logging.Formatter):
    """JSON log formatter."""

    def format(self, record: std_logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if they exist in the record
        # Note: 'extra' dict passed to logger is merged into record.__dict__
        # We need to filter out standard LogRecord attributes
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_record[key] = value

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


# ---------------------------------------------------------------------------
#  File-log format constant (console rendering lives in logging/console.py)
# ---------------------------------------------------------------------------

# Plain-text format for FILE handlers (the colored formatter is console-only).
# A single constant so the try/except branches of ``LoggingService.setup`` —
# which used to carry byte-identical copies — can't drift apart (#3 of the
# 2026-06-19 logging-duplication audit).


class LoggingService(ILoggingService):
    """Concrete implementation of logging service."""

    _SUPPORTED_LEVELS = {"debug", "info", "warning", "error", "critical"}

    def __init__(self, logger_name: str = "GANTraining"):
        """Initialize the logging service.

        Args:
            logger_name: Name of the logger to use.

        """
        self._logger = self._initialize_logger(logger_name)
        self._fallback_logger = std_logging.getLogger(f"{logger_name}.fallback")
        if not self._fallback_logger.handlers:
            self._fallback_logger.addHandler(std_logging.NullHandler())

        self.sinks = SinkRegistry()
        self.resolved_log_path: str | None = None
        self.log_dir_relocated_from: str | None = None
        self.sinks_dir_source: str = "declared"

        # [FORENSIC FIX]: Bounded deque prevents memory leak
        self._fallback_records: deque[dict[str, Any]] = deque(maxlen=1000)

        # [FORENSIC FIX]: Bounded throttle counts
        self._throttle_counts: dict[str, int] = {}
        self._throttle_limit = 3

    def _initialize_logger(self, logger_name: str) -> std_logging.Logger:
        """_initialize_logger.

        Args:
            logger_name (str): Description.
        Returns:
            std_logging.Logger: Description.
        """
        logger = std_logging.getLogger(logger_name)
        if not logger.handlers:
            logger.addHandler(std_logging.NullHandler())
        return logger

    def close(self) -> None:
        """Close every sink this service opened."""
        self.sinks.close_all()

    def setup(
        self,
        log_dir: str,
        log_level: str = "INFO",
        config: dict[str, Any] | None = None,
        logging_config: Optional["LoggingConfigSchema"] = None,
    ) -> None:
        """Set up the logger."""

        # Determine logging destinations from config
        log_to_file = True
        log_to_console = True
        silent = False

        if logging_config:
            log_to_file = logging_config.sinks.to_file
            log_to_console = logging_config.sinks.to_console
            # If silent=True, suppress console but keep file logging
            silent = logging_config.sinks.silent
            if silent:
                log_to_console = False
            adopt_logging_policy(logging_config)

        # ``log_policy`` owns the level: a ``--log-level`` / ``SPECTRAMR_LOG_LEVEL`` given
        # at the CLI beats ``log_level`` here, and a secondary rank is floored once, up
        # front, because every write below would otherwise undo the clamp.
        resolved_level = effective_level(log_level)

        # Loggers inherit the root level; nothing sets a level per logger.
        root_logger = std_logging.root
        root_logger.setLevel(resolved_level)

        # Set the level on our specific logger
        self._logger.setLevel(resolved_level)

        # Remove any existing handlers that strictly conflict (e.g. debugging handlers in production)
        # But be careful not to remove NOTSET (0) handlers which delegate to logger level.
        for handler in root_logger.handlers[:]:
            if (
                hasattr(handler, "level")
                and handler.level != std_logging.NOTSET
                and handler.level < resolved_level
            ):
                root_logger.removeHandler(handler)

        # Colored console handler on root via the shared SSOT helper (was
        # duplicated here and in bootstrap_console_logging — #1/#2 of the
        # 2026-06-19 logging-duplication audit). ``install_if_missing`` is gated
        # on log_to_console so silent mode never adds a console sink, while any
        # pre-existing plain handler is still upgraded in place.
        _install_or_upgrade_colored_console(
            root_logger, resolved_level, install_if_missing=log_to_console, stream=console_stream()
        )

        # Enforce the resolved level on every existing handler (level policy is a
        # separate concern from console install). NOTSET handlers intentionally
        # delegate to their logger's level, so leave them untouched.
        for handler in root_logger.handlers:
            if handler.level != std_logging.NOTSET:
                handler.setLevel(resolved_level)

        # Also ensure our specific logger's handlers have the correct level
        for handler in self._logger.handlers[:]:
            handler.setLevel(resolved_level)

        # Only rank 0 writes the text log unless ``logging.distributed.file_ranks`` is
        # ``all``, in which case each rank gets its own ``<name>.rank<N>.log``. A second
        # service on the same named logger reuses the first one's handler.
        _log_name = rank_log_path(f"{self._logger.name}.log")
        if (
            log_dir
            and log_to_file
            and _log_name is not None
            and "file" not in self.sinks
            and not any(isinstance(h, std_logging.FileHandler) for h in self._logger.handlers)
        ):
            sink = self.sinks.add(FileSink(self._logger, log_dir, _log_name))
            self.resolved_log_path = sink.resolved_path
            self.log_dir_relocated_from = sink.relocated_from

        # Prevent propagation to parent loggers to avoid duplicate messages
        self._logger.propagate = False

        # Console handler on the service's own logger. ``propagate = False`` was
        # set above, so this logger needs its OWN sink (root's handler won't see
        # it). Same shared SSOT helper as root — upgrades-or-installs exactly one
        # colored handler, never double-adds (#2 of the 2026-06-19 audit).
        if log_to_console:
            _install_or_upgrade_colored_console(
                self._logger, resolved_level, stream=console_stream()
            )


    def open_tensorboard(
        self,
        run_dir: Any,
        logging_config: "LoggingConfigSchema",
        *,
        is_rank_zero: bool = True,
        start_iteration: int = 0,
    ) -> Any:
        """Build the run's TensorBoard writer and register it as the ``tensorboard`` sink."""
        from spectramr.infrastructure.services.tensorboard_writer import TensorBoardWriter

        writer = TensorBoardWriter(
            run_dir, logging_config, is_rank_zero=is_rank_zero, start_iteration=start_iteration
        )
        self.sinks.add(TensorBoardSink(writer), replace=True)
        return writer

    def attach_artifact_sink(self, name: str, path: Any) -> None:
        """Register a destination another component writes (``path`` is a zero-arg callable)."""
        self.sinks.add(ArtifactSink(name, path), replace=True)

    def sink_records(self) -> list[dict[str, Any]]:
        """Every registered sink as ``{name, enabled, path}``."""
        return [asdict(r) for r in self.sinks.records()]

    def get_logger(self, name: str) -> Any:
        """Get a logger instance."""

        return std_logging.getLogger(name)

    def log_info(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log an info message."""

        self.log(
            "info",
            message,
            extra=extra,
            model_type=model_type,
            epoch=epoch,
        )

    def log_warning(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a warning message."""

        self.log(
            "warning",
            message,
            extra=extra,
            model_type=model_type,
            epoch=epoch,
        )

    def log_error(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log an error message."""

        self.log(
            "error",
            message,
            extra=extra,
            model_type=model_type,
            epoch=epoch,
        )

    def log_critical(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a critical message.

        The fifth rung of a ladder whose other four already delegate to :meth:`log`;
        ``"critical"`` was in ``_SUPPORTED_LEVELS`` and ``_level_map`` from the start
        and only this wrapper was missing, so callers reached for a method that was
        never defined (#2254).
        """

        self.log(
            "critical",
            message,
            extra=extra,
            model_type=model_type,
            epoch=epoch,
        )

    def log_debug(
        self,
        message: str,
        model_type: str = "",
        epoch: int = -1,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a debug message."""

        self.log(
            "debug",
            message,
            extra=extra,
            model_type=model_type,
            epoch=epoch,
        )

    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        **kwargs: Any,
    ) -> None:
        """Log a single batch of images with a tag."""
        self.log_images_batch({tag: images}, step, **kwargs)

    def set_step(self, step: int) -> None:
        """Set the current global step."""
        self._current_step = step

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self._current_epoch = epoch

    @property
    def step(self) -> int:
        """Return the current global step."""
        return getattr(self, "_current_step", 0)

    @property
    def epoch(self) -> int:
        """Return the current epoch."""
        return getattr(self, "_current_epoch", 0)

    def log_exception(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log an exception."""
        self._logger.exception(message, extra=extra)

    @property
    def logger(self) -> std_logging.Logger:
        """Public accessor for the underlying logger."""

        return self._logger

    def log(
        self,
        level: str,
        message: str,
        *,
        extra: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> None:
        """Log a message at the requested level with unified metadata."""

        # [FORENSIC FIX]: Throttle Memory Protection — but NEVER silence
        # high-severity levels. The cap is a lifetime cap (it never decays), so
        # throttling warnings/errors would hide a recurring NaN-loss / OOM /
        # gradient-overflow message after 3 occurrences (the repo "warnings are
        # not OK" anti-pattern). Only INFO/DEBUG spam is rate-limited.
        normalized = level.lower()
        if normalized in ("info", "debug"):
            key = f"{level}:{message}"
            count = self._throttle_counts.get(key, 0)
            self._throttle_counts[key] = count + 1

            # Periodic cleanup of throttle counts if it grows too large
            if len(self._throttle_counts) > 10000:
                self._throttle_counts.clear()

            if count >= self._throttle_limit:
                return  # Skip logging if throttled

        unified = self._unify_metadata(extra=extra, **metadata)

        try:
            logger_method = self._resolve_logger_method(level)
        except ValueError as error:
            self._record_fallback(level, message, unified, error)
            raise

        try:
            logger_method(message, extra=unified)
            # Flush only WARNING+ for crash-durability. Handlers already flush on
            # emit, so flushing every INFO line was a redundant I/O stall on the
            # per-iteration logging hot path.
            if normalized in ("warning", "error", "critical"):
                for handler in self._logger.handlers:
                    handler.flush()
        except Exception as error:  # pragma: no cover - defensive guard
            self._record_fallback(level, message, unified, error)

    @property
    def fallback_records(self) -> list[dict[str, Any]]:
        """Return recorded fallback events for diagnostics."""

        return list(self._fallback_records)

    def _resolve_logger_method(self, level: str):
        """_resolve_logger_method.

        Args:
            level (str): Description.
        Returns:
            Any: Description.
        """
        normalized_level = level.lower()
        if normalized_level not in self._SUPPORTED_LEVELS:
            raise ValueError(f"Unsupported log level '{level}'.")
        method = getattr(self._logger, normalized_level, None)
        if method is None or not callable(method):
            raise ValueError(f"Unsupported log level '{level}'.")
        return method

    def _unify_metadata(
        self,
        *,
        extra: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        """Combine provided metadata with optional extra dict."""

        unified: dict[str, Any] = {}
        if extra:
            unified.update(extra)
        for key, value in metadata.items():
            if value is not None:
                unified[key] = value
        return unified

    def _record_fallback(
        self,
        level: str,
        message: str,
        metadata: dict[str, Any],
        error: Exception,
    ) -> None:
        """Record a fallback event when the primary logger fails."""

        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": level.lower(),
            "message": message,
            "metadata": metadata,
            "error": repr(error),
        }
        self._fallback_records.append(record)
        try:
            self._fallback_logger.warning(
                "Logging fallback activated for level '%s': %s",
                level,
                message,
            )
        except Exception:  # pragma: no cover - fallback must not raise
            std_logging.getLogger("GANTraining.fallback_safety").debug(
                "Failed to emit fallback log for level '%s'", level
            )

    def log_images_batch(
        self,
        images: Any,
        step: int,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a batch of images to TensorBoard.

        Args:
            images: Dict of {tag: tensor} OR single Tensor (requires prefix)
            step: Global step
            prefix: Tag prefix (optional for dict, required for Tensor)
        """
        # LoggingService base doesn't have TensorBoard writer by default
        # But we implement it to satisfy interface.
        # If subclasses (Comprehensive) have writer, they can override or we can check for attribution.
        # Check if we have a writer (monkey-patched or mixed-in)
        writer = getattr(self, "_writer", None)
        enable_tb = getattr(self, "_enable_tensorboard", False)

        if not enable_tb or not writer:
            return

        try:
            if isinstance(images, dict):
                for tag, img_tensor in images.items():
                    full_tag = f"{prefix}/{tag}" if prefix else tag
                    writer.add_images(full_tag, img_tensor, global_step=step, **kwargs)
            else:
                if not prefix:
                    self.log_warning("log_images_batch called with Tensor but no prefix.")
                    return
                writer.add_images(prefix, images, global_step=step, **kwargs)
        except Exception as e:
            self.log_warning(f"Failed to log images batch: {e}")


class LoggingServiceFactory:
    """Factory for creating LoggingService instances."""

    @staticmethod
    def create(
        config: "LoggingConfigSchema",
        model_type: str = "Training",
        run_dir: Any = None,
    ) -> LoggingService:
        """Create a LoggingService; an undeclared ``sinks.dir`` follows ``run_dir``."""
        service = LoggingService(logger_name=model_type)
        log_dir, service.sinks_dir_source = (
            (str(config.sinks.dir), "declared")
            if run_dir is None
            else resolve_sinks_dir(config, run_dir)
        )
        service.setup(
            log_dir=log_dir,
            log_level=config.sinks.level,
            logging_config=config,
        )
        return service

    @staticmethod
    def create_unconfigured(model_type: str = "hpo") -> LoggingService:
        """A service with no sinks, for a parent process that owns no run directory."""
        return LoggingService(logger_name=model_type)
