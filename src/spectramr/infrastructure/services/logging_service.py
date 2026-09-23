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
import os
import sys
import tempfile
import warnings
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from spectramr.config.schemas.logging import LoggingConfigSchema

import json

import torch

from spectramr.domain.interfaces.service_interfaces import ILoggingService
from spectramr.infrastructure.logging.rank_console import rank_floor
from spectramr.infrastructure.services.iteration_counter_service import (
    IterationCounterService,
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
#  ANSI Color Console Formatter
# ---------------------------------------------------------------------------


class ColoredConsoleFormatter(std_logging.Formatter):
    """Rich ANSI-colored formatter for console output.

    Produces clean, scannable log lines with:
    - Dimmed timestamps (no date — only HH:MM:SS)
    - Colored level badges with glyphs
    - Cyan logger name (abbreviated)
    - Clean white message body
    - Bold red for errors/critical with full module path

    Falls back to plain text when stdout is not a terminal (piped/redirected).
    """

    # ANSI escape sequences
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _ITALIC = "\033[3m"

    # Foreground colors (256-color mode for richer palette)
    _WHITE = "\033[97m"
    _GREY = "\033[90m"
    _CYAN = "\033[36m"
    _GREEN = "\033[38;5;114m"  # Soft sage green
    _BLUE = "\033[38;5;75m"  # Sky blue
    _YELLOW = "\033[38;5;221m"  # Warm amber
    _RED = "\033[38;5;203m"  # Coral red
    _MAGENTA = "\033[38;5;204m"  # Hot pink

    # Background accents for badges
    _BG_GREEN = "\033[48;5;22m"
    _BG_BLUE = "\033[48;5;24m"
    _BG_YELLOW = "\033[48;5;94m"
    _BG_RED = "\033[48;5;52m"
    _BG_MAGENTA = "\033[48;5;53m"

    _LEVEL_STYLES = {
        std_logging.DEBUG: ("DBG", "🔍", _DIM + _BLUE, _BG_BLUE),
        std_logging.INFO: ("INF", "✦", _GREEN, _BG_GREEN),
        std_logging.WARNING: ("WRN", "⚠", _BOLD + _YELLOW, _BG_YELLOW),
        std_logging.ERROR: ("ERR", "✖", _BOLD + _RED, _BG_RED),
        std_logging.CRITICAL: ("CRT", "💀", _BOLD + _MAGENTA, _BG_MAGENTA),
    }

    def __init__(self, use_color: bool | None = None):
        """Initialize formatter.

        Args:
            use_color: Force color on/off. None = auto-detect from terminal.
        """
        super().__init__()
        if use_color is None:
            # Respect standard environment variables
            if os.environ.get("FORCE_COLOR", ""):
                self._use_color = True
            elif os.environ.get("NO_COLOR", ""):
                self._use_color = False
            else:
                # Check both stdout and stderr (StreamHandler defaults to stderr)
                self._use_color = (hasattr(sys.stderr, "isatty") and sys.stderr.isatty()) or (
                    hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
                )
        else:
            self._use_color = use_color

    def _abbreviate_name(self, name: str, max_len: int = 28) -> str:
        """Shorten logger names like 'spectramr.infrastructure.training.strategies.diffusion' → 's.i.t.s.diffusion'."""
        parts = name.split(".")
        if len(".".join(parts)) <= max_len:
            return name
        # Keep last part full, abbreviate the rest
        abbreviated = [p[0] for p in parts[:-1]] + [parts[-1]]
        return ".".join(abbreviated)

    def format(self, record: std_logging.LogRecord) -> str:
        """Format a log record with colors and clean layout."""
        if not self._use_color:
            # Plain fallback for non-TTY (pipes, files, CI)
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            return f"{ts} [{record.levelname:<8s}] {record.name}: {record.getMessage()}"

        R = self._RESET
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Level badge
        tag, glyph, fg, bg = self._LEVEL_STYLES.get(
            record.levelno,
            ("???", "•", self._WHITE, ""),
        )
        badge = f"{bg}{fg} {glyph} {tag} {R}"

        # Dimmed timestamp
        time_str = f"{self._DIM}{ts}{R}"

        # Cyan abbreviated logger name
        short_name = self._abbreviate_name(record.name)
        name_str = f"{self._CYAN}{short_name}{R}"

        # Message color varies by level
        msg = record.getMessage()
        if record.levelno >= std_logging.ERROR:
            msg_str = f"{self._BOLD}{self._RED}{msg}{R}"
        elif record.levelno >= std_logging.WARNING:
            msg_str = f"{self._YELLOW}{msg}{R}"
        else:
            msg_str = f"{self._WHITE}{msg}{R}"

        line = f"{time_str} {badge} {name_str} {self._DIM}│{R} {msg_str}"

        # Append exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            line += f"\n{self._RED}{self.formatException(record.exc_info)}{R}"

        return line


# ---------------------------------------------------------------------------
#  Shared console-handler routine + format constants (SSOT)
# ---------------------------------------------------------------------------

# Plain-text format for FILE handlers (the colored formatter is console-only).
# A single constant so the try/except branches of ``LoggingService.setup`` —
# which used to carry byte-identical copies — can't drift apart (#3 of the
# 2026-06-19 logging-duplication audit).
_FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _install_or_upgrade_colored_console(
    target: std_logging.Logger,
    level: int,
    *,
    install_if_missing: bool = True,
) -> bool:
    """Ensure ``target`` renders to the console via :class:`ColoredConsoleFormatter`.

    The single source of truth for "put a colored console handler on this
    logger", shared by :func:`bootstrap_console_logging` and
    :meth:`LoggingService.setup`. Before the 2026-06-19 logging-duplication
    audit this routine was copy-pasted in three places (audit items #1/#2) —
    which is why the bootstrap comment used to reference a non-existent
    ``LoggingService._upgrade_existing_handlers`` method.

    Behaviour:

    * Any existing plain (non-colored, non-file) ``StreamHandler`` is UPGRADED
      in place — its formatter is swapped to the colored one and the handler
      identity is preserved, so a caller's custom stream / level / filters
      survive. This is the F-LOG-COLOR round-12.1 fix: upgrade rather than
      stack, so a log line never emits twice (once plain, once colored).
    * A fresh colored ``StreamHandler`` is installed ONLY when ``target`` has no
      console handler at all AND ``install_if_missing`` is True. ``setup`` passes
      ``install_if_missing=log_to_console`` so silent mode never adds a sink,
      while a pre-existing plain handler is still upgraded.

    Returns True if a handler was upgraded or installed.
    """
    colored_fmt = ColoredConsoleFormatter()
    changed = False
    has_console = False
    for h in target.handlers:
        if isinstance(h, std_logging.StreamHandler) and not isinstance(h, std_logging.FileHandler):
            has_console = True
            if not isinstance(h.formatter, ColoredConsoleFormatter):
                h.setFormatter(colored_fmt)
                if h.level == std_logging.NOTSET or h.level > level:
                    h.setLevel(level)
                changed = True
    if not has_console and install_if_missing:
        handler = std_logging.StreamHandler()
        handler.setFormatter(colored_fmt)
        handler.setLevel(level)
        target.addHandler(handler)
        changed = True
    return changed


def bootstrap_console_logging(
    level: int | str = std_logging.INFO,
    force: bool = False,
) -> None:
    """Install :class:`ColoredConsoleFormatter` on the root logger.

    F-LOG-COLOR (2026-05-17 round 12): CLI entry points (``train``,
    ``audit``, ``predict``, ``sanity_check``, ``campaign``, ``report``,
    ...) historically did NOT instantiate :class:`LoggingService` —
    only ``hpo`` did. So ``logger.info(...)`` calls anywhere in those
    flows fell back to Python's ``lastResort`` handler (plain ``stderr``,
    no colors, no badges, default level=WARNING).

    The user-visible symptom: pretty colored output in HPO runs, plain
    "LEVEL:name:message" or nothing at all in train / audit / predict
    runs, even though ``isatty()`` / ``TERM`` / ``FORCE_COLOR`` were all
    correctly set.

    Call this exactly once at the top of each CLI entry point. The
    function is **idempotent**: re-invocations either no-op (when a
    colored handler is already present) or reset the level if the
    caller asked for a more permissive one.

    Args:
        level: Root logger level. Defaults to ``INFO`` so the
            informational lines that ``logger.info(...)`` emits in
            pipelines actually appear. ``LoggingService.__init__``
            defaults to ``WARNING`` because it has a separate level-
            controller; this helper is for plain CLI runs that never
            touch the service.
        force: When True, remove every existing root-logger handler
            before installing the colored one. Use ONLY when you
            need to recover from a misconfigured handler set; the
            default ``False`` is safe and preserves any
            already-installed colored formatter.
    """
    root = std_logging.getLogger()
    resolved_level: int
    if isinstance(level, str):
        resolved_level = std_logging.getLevelName(level.upper())
        if not isinstance(resolved_level, int):
            resolved_level = std_logging.INFO  # invalid string fallback
    else:
        resolved_level = int(level)

    # Same secondary-rank clamp as ``LoggingService.setup``, for the same
    # reason: the ``has_colored`` branch below deliberately lowers an existing
    # root level ("never SILENCE existing handlers"), which would re-verbose a
    # rank that ``quiet_secondary_ranks()`` had already quietened. Clamping the
    # resolved level makes the policy independent of the order these two run in,
    # rather than relying on ``cli/app.py`` happening to bootstrap first.
    resolved_level = rank_floor(resolved_level)

    has_colored = any(
        isinstance(h, std_logging.StreamHandler)
        and not isinstance(h, std_logging.FileHandler)
        and isinstance(h.formatter, ColoredConsoleFormatter)
        for h in root.handlers
    )

    if has_colored and not force:
        # Already configured. Only bump the level downward if the
        # caller wants more verbose output — never SILENCE existing
        # handlers from this helper.
        if root.level == std_logging.NOTSET or root.level > resolved_level:
            root.setLevel(resolved_level)
        return

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # pragma: no cover - close() is best-effort
                pass

    # Upgrade any pre-existing plain StreamHandler to the colored formatter in
    # place (a transitive ``logging.basicConfig(...)`` may have left one), else
    # install a fresh colored handler. Delegates to the shared SSOT helper that
    # ``LoggingService.setup`` also uses — collapsing the copy that used to live
    # here (#1/#2 of the 2026-06-19 logging-duplication audit). Upgrading in
    # place rather than stacking is the F-LOG-COLOR round-12.1 fix: a log line
    # must never emit twice (once plain, once colored).
    _install_or_upgrade_colored_console(root, resolved_level)

    root.setLevel(resolved_level)


# The pristine stdlib ``logging.getLogger``, captured once at import before any
# LoggingService patches it. Every patched wrapper delegates to THIS, so repeated
# LoggingService instantiations (tests, HPO workers, nested pipelines) never chain
# wrappers — a chain slowed every getLogger call and never restored the original.
_TRUE_GETLOGGER = std_logging.getLogger


class _GlobalLevelFilter(std_logging.Filter):
    """Global filter to enforce logging level across all loggers."""

    def __init__(self, level: int):
        """__init__.

        Args:
            level (int): Description.
        """
        super().__init__()
        self._level = level

    def filter(self, record: std_logging.LogRecord) -> bool:
        """Filter out log records below the configured level."""
        return record.levelno >= self._level

    def set_level(self, level: int) -> None:
        """Update the filter level."""
        self._level = level


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

        # [FORENSIC FIX]: Bounded deque prevents memory leak
        self._fallback_records: deque[dict[str, Any]] = deque(maxlen=1000)

        # [FORENSIC FIX]: Bounded throttle counts
        self._throttle_counts: dict[str, int] = {}
        self._throttle_limit = 3

        # Set up level mapping for log level resolution
        self._level_map = {
            "debug": std_logging.DEBUG,
            "info": std_logging.INFO,
            "warning": std_logging.WARNING,
            "error": std_logging.ERROR,
            "critical": std_logging.CRITICAL,
        }

        # Set a default level on the root logger to prevent other modules from setting INFO
        root_logger = std_logging.getLogger()
        if not root_logger.handlers:
            # If no handlers exist, set up a colored console handler as default
            _default_handler = std_logging.StreamHandler()
            _default_handler.setFormatter(ColoredConsoleFormatter())
            _default_handler.setLevel(std_logging.WARNING)
            root_logger.addHandler(_default_handler)
            root_logger.setLevel(std_logging.WARNING)

        # Install a global filter to enforce logging levels — idempotently.
        # Drop any _GlobalLevelFilter a prior LoggingService left on the root so
        # they don't accumulate (each added one, none removed it).
        root_logger.filters = [
            f for f in root_logger.filters if not isinstance(f, _GlobalLevelFilter)
        ]
        self._global_filter = _GlobalLevelFilter(std_logging.WARNING)
        root_logger.addFilter(self._global_filter)

        # Monkey-patch getLogger to auto-set levels on new loggers. Always
        # delegate to the PRISTINE stdlib getLogger (captured at import), never to
        # whatever is currently installed — otherwise a second instantiation
        # captures the first instance's patch as "original" and wrappers chain,
        # slowing every getLogger call and making restoration impossible.
        self._original_getLogger = _TRUE_GETLOGGER
        self._current_level = std_logging.WARNING
        # Store ONE bound-method object: accessing ``self._patched_getLogger``
        # each time creates a fresh bound method that is not ``is``-identical, so
        # close() must compare against this stored reference to know whether its
        # own patch is the installed one.
        self._patched_getLogger_ref = self._patched_getLogger
        std_logging.getLogger = self._patched_getLogger_ref

    def _resolve_log_level(self, level: str) -> int:
        """_resolve_log_level.

        Args:
            level (str): Description.
        Returns:
            int: Description.
        """
        try:
            return self._level_map[level.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported log level: {level}") from exc

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

    def _patched_getLogger(self, name: str = "") -> std_logging.Logger:
        """Patched getLogger that automatically sets the level on new loggers."""
        logger = self._original_getLogger(name)
        # Only set level if it hasn't been explicitly set to a different level
        if logger.level == std_logging.NOTSET or logger.level < self._current_level:
            logger.setLevel(self._current_level)
        return logger

    def close(self) -> None:
        """Undo the process-global logging patches this instance installed.

        Restores the pristine ``logging.getLogger`` (only if THIS instance's
        patch is the one currently installed, so a later service's patch is left
        intact) and removes this instance's ``_GlobalLevelFilter`` from the root
        logger, so the state does not leak past the service's lifetime.
        """
        if getattr(std_logging, "getLogger", None) is getattr(self, "_patched_getLogger_ref", None):
            std_logging.getLogger = _TRUE_GETLOGGER

        if getattr(self, "_global_filter", None) is not None:
            try:
                _TRUE_GETLOGGER().removeFilter(self._global_filter)
            except Exception as _exc:  # pragma: no cover - defensive
                _TRUE_GETLOGGER(__name__).debug("Suppressed exception removing filter: %s", _exc)

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

        # Resolve the log level.
        #
        # Clamped HERE, once, rather than after each of the six places below
        # that write it: this method pushes ``resolved_level`` onto the global
        # filter, the root logger, every logger in ``loggerDict``, this
        # service's own logger, the console handler and every existing handler.
        # An earlier ``quiet_secondary_ranks()`` clamp is undone by the first of
        # those, so on a 4-rank launch every INFO logged from training onwards
        # would go back to appearing four times. Rank 0 and single-process runs
        # resolve unchanged.
        resolved_level = rank_floor(self._resolve_log_level(log_level))

        # Update the global filter and current level
        self._global_filter.set_level(resolved_level)
        self._current_level = resolved_level

        # Set the root logger level (basicConfig may have already been called in __init__)
        root_logger = std_logging.getLogger()
        root_logger.setLevel(resolved_level)

        # Also set the level on all existing loggers to ensure they respect the level
        for name in std_logging.root.manager.loggerDict:
            logger = std_logging.getLogger(name)
            if hasattr(logger, "setLevel"):
                logger.setLevel(resolved_level)

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
            root_logger, resolved_level, install_if_missing=log_to_console
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

        # Where the log ACTUALLY went. `self._log_dir` holds the INTENDED
        # directory and survives relocation unchanged, so it cannot answer "where
        # is this run's log": `logging.sinks.dir` is authoritative over the run
        # directory (correct per non-negotiable 3b -- a declared value must not be
        # replaced by a caller default), which means the log routinely lands
        # somewhere other than beside the artifacts it describes, and nothing
        # recorded which path won. Set unconditionally so a consumer never has to
        # distinguish "no attribute" from "no file log".
        self.resolved_log_path: str | None = None
        self.log_dir_relocated_from: str | None = None

        if log_dir and log_to_file:
            # Check if file handler already exists
            has_file_handler = any(
                isinstance(h, std_logging.FileHandler) for h in self._logger.handlers
            )

            if not has_file_handler:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, f"{self._logger.name}.log")
                    file_handler = std_logging.FileHandler(log_path)
                    file_handler.setFormatter(std_logging.Formatter(_FILE_LOG_FORMAT))
                    self._logger.addHandler(file_handler)
                    self.resolved_log_path = log_path
                except (PermissionError, OSError) as exc:
                    # Fallback to a temporary directory when permissions fail.
                    #
                    # This branch used to be entirely silent, and it could not be
                    # otherwise: it touched neither `self._logger` nor `warnings`,
                    # so the one event that relocates the whole run log announced
                    # itself nowhere. On a compute node that temp directory is
                    # wiped at job teardown, so the run's log did not merely move
                    # -- it ceased to exist, while the run reported success. That
                    # is how a run wrote provenance, a resolved config, a
                    # TensorBoard event file, eight debug snapshots and sixteen
                    # PNGs, and not one log line (pitfall #9 / non-negotiable 3).
                    #
                    # `propagate` is still True here (it is set False further
                    # down), so this reaches the root handlers configured above.
                    # `warnings.warn` is emitted as well because a log warning
                    # about the log sink failing is exactly the message most
                    # likely to be lost.
                    fallback_dir = tempfile.mkdtemp(prefix="spectramr_logs_")
                    log_path = os.path.join(fallback_dir, f"{self._logger.name}.log")
                    file_handler = std_logging.FileHandler(log_path)
                    file_handler.setFormatter(std_logging.Formatter(_FILE_LOG_FORMAT))
                    self._logger.addHandler(file_handler)
                    self.resolved_log_path = log_path
                    self.log_dir_relocated_from = str(log_dir)
                    relocation_notice = (
                        f"log directory {log_dir!r} is not writable "
                        f"({type(exc).__name__}: {exc}); this run's ENTIRE log has "
                        f"been relocated to {log_path!r}. On a compute node that "
                        "temporary directory is wiped at job teardown, so copy the "
                        "log out before the job ends or set logging.sinks.dir to a "
                        "writable path. The relocation is recorded in "
                        "provenance.json under `logging`."
                    )
                    self._logger.warning(relocation_notice)
                    warnings.warn(relocation_notice, RuntimeWarning, stacklevel=2)

        # Prevent propagation to parent loggers to avoid duplicate messages
        self._logger.propagate = False

        # Console handler on the service's own logger. ``propagate = False`` was
        # set above, so this logger needs its OWN sink (root's handler won't see
        # it). Same shared SSOT helper as root — upgrades-or-installs exactly one
        # colored handler, never double-adds (#2 of the 2026-06-19 audit).
        if log_to_console:
            _install_or_upgrade_colored_console(self._logger, resolved_level)

        # Apply the level to all loggers again after setup (in case new ones were created)
        root_logger.setLevel(resolved_level)
        for name in std_logging.root.manager.loggerDict:
            logger = std_logging.getLogger(name)
            if hasattr(logger, "setLevel"):
                logger.setLevel(resolved_level)

        # Explicitly disable logging for problematic modules that bypass the service
        problematic_modules = [
            "spectramr.application.orchestration.pipeline_orchestrator",
            "spectramr.data.factories.dataloader_factory",
            "spectramr.data.loaders.async_data_loader",
        ]

        for module_name in problematic_modules:
            module_logger = std_logging.getLogger(module_name)
            module_logger.setLevel(std_logging.WARNING)
            # Remove any handlers that might log INFO
            for handler in module_logger.handlers[:]:
                if hasattr(handler, "level") and handler.level < std_logging.WARNING:
                    module_logger.removeHandler(handler)
            # Ensure no INFO messages get through
            module_logger.addHandler(std_logging.NullHandler())

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
    def create(config: "LoggingConfigSchema", model_type: str = "Training") -> LoggingService:
        """Create a LoggingService instance."""
        service = LoggingService(logger_name=model_type)
        # Use config.level (canonical name in LoggingConfigSchema)
        service.setup(
            log_dir=config.sinks.dir,
            log_level=config.sinks.level,
            logging_config=config,
        )
        return service


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class ComprehensiveLoggingService(LoggingService):
    """Combines all logging functionality.

    Now includes TensorBoard support via torch.utils.tensorboard.
    """

    _DEFAULT_METADATA = {
        "model_type": "unknown",
        "epoch": -1,
        "phase": "unspecified",
        "stage": "global",
        "run_id": "untracked",
    }

    def __init__(
        self,
        log_dir: str = "./logs",
        logger_name: str = "GANTraining",
        log_level: str = "INFO",
        logging_config: Optional["LoggingConfigSchema"] = None,
    ) -> None:
        """Initialize ComprehensiveLoggingService.

        Args:
            log_dir: Base directory for log files.
            logger_name: Logger name for structured output.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            logging_config: Optional logging configuration schema.
        """
        super().__init__(logger_name=logger_name)

        # Store the logging config for later use
        self._logging_config = logging_config

        # Use config values if provided, otherwise fall back to parameters
        if logging_config:
            actual_log_dir = logging_config.sinks.dir or log_dir
            actual_log_level = (
                logging_config.sinks.level.value
                if hasattr(logging_config.sinks.level, "value")
                else str(logging_config.sinks.level)
            )
            actual_log_level = actual_log_level.upper() if actual_log_level else log_level
        else:
            actual_log_dir = log_dir
            actual_log_level = log_level

        self.setup(actual_log_dir, actual_log_level, logging_config=logging_config)

        self._log_dir = actual_log_dir
        # Throttling state is initialised once by ``LoggingService.__init__`` (the
        # single throttle authority) — no re-init here. See the ``log`` override
        # and #4 of the 2026-06-19 logging-duplication audit.

        # No TensorBoard writer here. This class used to build a SECOND one from
        # `logging.tracking.tensorboard_dir`, in a different directory from the
        # writer `pipelines/train.py` actually uses -- and it never ran, because
        # `bootstrap.py` resolves `ILoggingService` through
        # `LoggingServiceFactory.create`, which returns a base `LoggingService`.
        # So the knob the config documented steered a writer nobody had (#928).
        #
        # `TensorBoardWriter` (infrastructure/services/tensorboard_writer.py) is
        # now the single writer and `pipelines/train.py` owns it. The `_writer`
        # attributes stay `None`/`False` so the TB branches in the methods below
        # remain the no-ops they have always been in practice, rather than this
        # class quietly acquiring a third writer.
        self._writer = None
        self._enable_tensorboard = False

    def _unify_metadata(self, **metadata: object) -> dict[str, object]:
        """_unify_metadata.

        Returns:
            dict[str, object]: Description.
        """
        if not metadata:
            return {}
        merged = {**self._DEFAULT_METADATA}
        merged.update({k: v for k, v in metadata.items() if v is not None})
        return merged

    def log(
        self,
        level: str,
        message: str,
        **metadata: object,
    ) -> None:
        """Inject default metadata, then delegate to the base logger.

        Throttling lives SOLELY in :meth:`LoggingService.log` (the base class).
        This override used to re-throttle on the SAME ``_throttle_counts`` dict
        and the SAME ``f"{level}:{message}"`` key before calling ``super().log``,
        so every info/debug line was counted twice — a message meant to print
        three times printed once (#4 of the 2026-06-19 logging-duplication
        audit). Worse, the local throttle here applied to WARNING/ERROR/CRITICAL
        too, violating the "warnings are not OK" rule that the base ``log`` is
        careful to honour (it rate-limits only INFO/DEBUG). Delegating
        unconditionally fixes both — the base class is the single throttle
        authority.
        """
        unified = self._unify_metadata(**metadata)
        if unified:
            super().log(level, message, **unified)
        else:
            super().log(level, message)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        prefix: str | None = None,
        level: str = "info",
    ) -> None:
        """log_metrics.

        Args:
            metrics (Mapping[str, float]): Description.
            step (int | None): Description.
            prefix (str | None): Description.
            level (str): Description.
        """
        metadata: dict[str, object] = {"metrics": dict(metrics)}
        if step is not None:
            metadata["step"] = step
        if prefix:
            metadata["prefix"] = prefix
        unified = self._unify_metadata(**metadata)
        super().log(level, "Metrics update", **unified)

        # Log to TensorBoard
        if self._enable_tensorboard and self._writer and (step is not None):
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self._writer.add_scalar(tag, value, global_step=step)

    def should_log_step(self, counter: IterationCounterService) -> bool:
        """Check if metrics should be logged based on interval.

        Queries the IterationCounterService to determine if the logging interval
        has been reached and we haven't exceeded max_iterations.

        Args:
            counter: IterationCounterService instance (SSOT for step tracking)

        Returns:
            True if logging interval reached and within limits, False otherwise
        """
        if not self._logging_config:
            return True  # Log every step if no config

        return counter.should_log(self._logging_config.intervals.log)

    def log_metrics_if_needed(
        self,
        counter: IterationCounterService,
        metrics: Mapping[str, float],
        *,
        prefix: str | None = None,
        level: str = "info",
    ) -> None:
        """Log metrics only if logging interval reached (SSOT pattern).

        This method queries the IterationCounterService to determine if the logging
        interval has been reached. If so, logs the metrics with step information.

        Args:
            counter: IterationCounterService instance (SSOT for step tracking)
            metrics: Dictionary of metric names to values
            prefix: Optional prefix for metric names
            level: Logging level (default "info")

        Example:
            >>> logging_service.log_metrics_if_needed(
            ...     counter, {"loss": 0.5, "accuracy": 0.95}
            ... )
        """
        if not self.should_log_step(counter):
            return

        # Add step and epoch context
        metadata_dict = {
            "step": counter.current_step,
            "epoch": counter.current_epoch,
            "metrics": dict(metrics),
        }
        if prefix:
            metadata_dict["prefix"] = prefix

        unified = self._unify_metadata(**metadata_dict)
        super().log(level, "Training metrics", **unified)

        # Log to TensorBoard
        # Note: log_metrics above already handles TB if we passed step, but here we do it explicitly
        # to ensure it respects the 'if_needed' condition (already checked above)
        if self._enable_tensorboard and self._writer:
            step = counter.current_step
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self._writer.add_scalar(tag, value, global_step=step)

    def log_image(
        self,
        tag: str,
        image_tensor: Any,
        global_step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image to TensorBoard.

        Args:
            tag: Data identifier
            image_tensor: Image data (torch.Tensor, numpy.array)
            global_step: Global step value
            dataformats: Image data format specification (e.g. 'CHW', 'HWC', 'HW')
        """
        if self._enable_tensorboard and self._writer:
            try:
                self._writer.add_image(tag, image_tensor, global_step, dataformats=dataformats)
            except Exception as e:
                # Don't crash training on viz failure
                self.log_warning(f"Failed to log image '{tag}' to TensorBoard: {e}")

    def log_images_batch(
        self,
        images: torch.Tensor | dict[str, Any],
        step: int,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a batch of multiple images organized by category.

        Args:
            images: Dict mapping category names to image tensors or a single tensor
            step: Global step value for logging
            prefix: Optional prefix for tags
            kwargs: Additional arguments (e.g. max_images)
        """
        if not self._enable_tensorboard or not self._writer:
            return

        import torch

        images_dict = images if isinstance(images, dict) else {prefix or "images": images}
        max_images = kwargs.get("max_images", 4)

        for category_name, img_tensor in images_dict.items():
            if img_tensor is None:
                continue

            try:
                # Handle single image
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                # Limit to max_images
                num_to_log = min(max_images, img_tensor.size(0))
                images_to_log = img_tensor[:num_to_log]

                # Convert to numpy if needed for visualization
                if isinstance(images_to_log, torch.Tensor):
                    images_to_log = images_to_log.detach().cpu()

                # Normalize to [0, 1] for visualization if not already
                if images_to_log.dtype == torch.float32 or images_to_log.dtype == torch.float64:
                    min_val = images_to_log.min()
                    max_val = images_to_log.max()
                    if max_val > min_val:
                        images_to_log = (images_to_log - min_val) / (max_val - min_val)
                    else:
                        images_to_log = torch.zeros_like(images_to_log)

                # Log as grid
                from torchvision.utils import make_grid

                grid = make_grid(images_to_log, nrow=2, normalize=False)
                self._writer.add_image(category_name, grid, global_step=step, dataformats="CHW")

            except Exception as e:
                self.log_warning(f"Failed to log image batch '{category_name}' to TensorBoard: {e}")

    def close(self) -> None:
        """Clean up resources; also undoes the base global logging patches."""
        if self._writer:
            self._writer.close()
        super().close()


# Factory function for creating logging service
def create_logging_service(
    log_dir: str = "./logs",
    logger_name: str = "GANTraining",
) -> ComprehensiveLoggingService:
    """Factory function to create a comprehensive logging service.

    Args:
        log_dir: Directory for logs
        logger_name: Name of the logger

    Returns:
        ComprehensiveLoggingService instance

    """
    return ComprehensiveLoggingService(log_dir, logger_name)
