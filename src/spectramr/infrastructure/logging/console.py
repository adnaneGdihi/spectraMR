"""Console rendering: the colored formatter and the one handler-install routine.

Split out of ``services/logging_service.py`` so the log-level policy
(:mod:`spectramr.infrastructure.logging.log_policy`) can install a console
handler without importing ``torch``, which ``logging_service`` pulls in.
"""

import logging as std_logging
import os
import sys
from datetime import datetime
from typing import ClassVar

__all__ = [
    "ColoredConsoleFormatter",
    "install_or_upgrade_colored_console",
    "set_console_stream",
]

#: Marks a handler this module created, so a stream change replaces ours and never a caller's.
_MANAGED_ATTR = "_spectramr_console_stream"

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

    _LEVEL_STYLES: ClassVar[dict[int, tuple[str, str, str, str]]] = {
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


class _Below(std_logging.Filter):
    """Pass records under ``floor``; the stdout half of a split console."""

    def __init__(self, floor: int) -> None:
        super().__init__()
        self._floor = floor

    def filter(self, record: std_logging.LogRecord) -> bool:
        return record.levelno < self._floor


class _FromLevel(_Below):
    """Pass records at or over ``floor``; the stderr half of a split console."""

    def filter(self, record: std_logging.LogRecord) -> bool:
        return record.levelno >= self._floor


def _new_handlers(stream: str, level: int) -> list[std_logging.StreamHandler]:
    """The managed console handlers for ``stream`` (``split``, ``stdout`` or ``stderr``)."""
    if stream not in ("split", "stdout", "stderr"):
        raise ValueError(f"Unsupported console stream: {stream!r}")
    plan = {
        "split": [(sys.stdout, _Below), (sys.stderr, _FromLevel)],
        "stdout": [(sys.stdout, None)],
        "stderr": [(sys.stderr, None)],
    }[stream]
    handlers = []
    for target, gate in plan:
        handler = std_logging.StreamHandler(target)
        handler.setFormatter(ColoredConsoleFormatter())
        handler.setLevel(level)
        if gate is not None:
            handler.addFilter(gate(std_logging.WARNING))
        setattr(handler, _MANAGED_ATTR, stream)
        handlers.append(handler)
    return handlers


def install_or_upgrade_colored_console(
    target: std_logging.Logger,
    level: int,
    *,
    install_if_missing: bool = True,
    stream: str = "split",
) -> bool:
    """Ensure ``target`` renders to the console via :class:`ColoredConsoleFormatter`.

    The single source of truth for "put a colored console handler on this
    logger", shared by the log policy and :meth:`LoggingService.setup`.

    * Any existing plain (non-colored, non-file) ``StreamHandler`` is UPGRADED
      in place -- its formatter is swapped to the colored one and the handler
      identity is preserved, so a caller's custom stream / level / filters
      survive. Upgrade rather than stack, so a log line never emits twice.
    * Fresh handlers for ``stream`` are installed ONLY when ``target`` has no
      console handler at all AND ``install_if_missing`` is True (``setup`` passes
      ``log_to_console`` so silent mode never adds a sink).

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
        for handler in _new_handlers(stream, level):
            target.addHandler(handler)
        changed = True
    return changed


def set_console_stream(target: std_logging.Logger, stream: str, level: int) -> bool:
    """Point the console handlers this module installed at ``stream``.

    A caller's own handler is never replaced. Returns True if the handlers changed;
    a no-op when they already match, so applying the same config twice is free.
    """
    managed = [h for h in target.handlers if hasattr(h, _MANAGED_ATTR)]
    if not managed or all(getattr(h, _MANAGED_ATTR) == stream for h in managed):
        return False
    for handler in managed:
        target.removeHandler(handler)
    for handler in _new_handlers(stream, level):
        target.addHandler(handler)
    return True
