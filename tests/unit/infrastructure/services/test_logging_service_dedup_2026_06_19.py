"""Logging-service de-duplication (2026-06-19 audit).

Pins the fixes that consolidated redundant root-logger configuration in
:mod:`spectramr.infrastructure.services.logging_service`:

* #1/#2 — the "upgrade-or-install a colored console handler" routine, once
  copy-pasted in ``bootstrap_console_logging`` and ``LoggingService.setup``,
  now lives in a single ``_install_or_upgrade_colored_console`` helper.
* #3 — the file-handler format string is a single ``_FILE_LOG_FORMAT`` constant
  (was duplicated verbatim across the try/except branches of ``setup``).
* #4 — ``ComprehensiveLoggingService.log`` re-throttled on the base class's
  ``_throttle_counts`` (double-counting info/debug and wrongly throttling
  warnings/errors). The class is deleted, so ``LoggingService.log`` is the single
  throttle authority and the pin is that no second service class exists.

The helper tests run on throwaway *named* loggers — no ``LoggingService`` is
constructed, so the global ``getLogger`` monkey-patch is never installed. The
``#4`` check is AST-based for the same reason.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from spectramr.infrastructure.services.logging_service import (
    ColoredConsoleFormatter,
    _install_or_upgrade_colored_console,
)

_SRC = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "services"
    / "logging_service.py"
)


def _fresh_logger(name: str) -> logging.Logger:
    """Return a uniquely-named logger with no handlers (no cross-test bleed)."""
    lg = logging.getLogger(name)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.setLevel(logging.NOTSET)
    return lg


def _stream_handlers(lg: logging.Logger) -> list[logging.Handler]:
    return [
        h
        for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]


# --------------------------------------------------------------------------- #
#  #1 / #2 — shared colored-console helper
# --------------------------------------------------------------------------- #


def test_helper_installs_fresh_colored_handler_when_none() -> None:
    lg = _fresh_logger("test_dedup.install_fresh")
    changed = _install_or_upgrade_colored_console(lg, logging.INFO, stream="stderr")
    sh = _stream_handlers(lg)
    assert changed is True
    assert len(sh) == 1  # `split` installs a stdout/stderr pair; see test_console.py
    assert isinstance(sh[0].formatter, ColoredConsoleFormatter)
    assert sh[0].level == logging.INFO


def test_helper_upgrades_plain_handler_in_place() -> None:
    """The load-bearing contract: upgrade in place, preserve handler identity."""
    lg = _fresh_logger("test_dedup.upgrade_in_place")
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(plain)

    changed = _install_or_upgrade_colored_console(lg, logging.INFO)

    sh = _stream_handlers(lg)
    assert changed is True
    assert len(sh) == 1, "must upgrade in place, not stack a second handler"
    assert sh[0] is plain, "handler identity must be preserved"
    assert isinstance(plain.formatter, ColoredConsoleFormatter)


def test_helper_is_noop_when_already_colored() -> None:
    lg = _fresh_logger("test_dedup.already_colored")
    h = logging.StreamHandler()
    h.setFormatter(ColoredConsoleFormatter())
    lg.addHandler(h)

    changed = _install_or_upgrade_colored_console(lg, logging.INFO)

    assert changed is False, "already-colored handler needs no change"
    assert len(_stream_handlers(lg)) == 1, "must not add a second handler"


def test_helper_install_if_missing_false_adds_nothing() -> None:
    """Silent-mode gate: setup passes install_if_missing=log_to_console."""
    lg = _fresh_logger("test_dedup.no_install")
    changed = _install_or_upgrade_colored_console(lg, logging.INFO, install_if_missing=False)
    assert changed is False
    assert _stream_handlers(lg) == [], "silent mode must not add a console sink"


def test_helper_install_if_missing_false_still_upgrades_existing() -> None:
    """install_if_missing=False suppresses fresh installs but still upgrades."""
    lg = _fresh_logger("test_dedup.upgrade_no_install")
    plain = logging.StreamHandler()
    plain.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(plain)

    changed = _install_or_upgrade_colored_console(lg, logging.INFO, install_if_missing=False)
    assert changed is True
    assert isinstance(plain.formatter, ColoredConsoleFormatter)
    assert len(_stream_handlers(lg)) == 1


# --------------------------------------------------------------------------- #
#  #3 — single file-format constant
# --------------------------------------------------------------------------- #


def test_file_log_format_literal_defined_once() -> None:
    from spectramr.infrastructure.logging import sinks

    literal = '"%(asctime)s - %(name)s - %(levelname)s - %(message)s"'
    sinks_src = Path(sinks.__file__).read_text()
    assert sinks.FILE_LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert sinks_src.count(literal) == 1
    assert literal not in _SRC.read_text(), "the format string lives in infrastructure/logging/sinks.py"


# --------------------------------------------------------------------------- #
#  #4 — one service class, so one throttle authority
# --------------------------------------------------------------------------- #


def test_there_is_one_logging_service_class() -> None:
    """``ComprehensiveLoggingService`` re-throttled on the base class's counters (#4) and, with a
    TensorBoard writer that was always ``None``, added nothing the base class lacked. It and its
    factory are gone; a second service class must not come back under another name."""
    from spectramr.infrastructure.services import logging_service

    assert not hasattr(logging_service, "ComprehensiveLoggingService")
    assert not hasattr(logging_service, "create_logging_service")
    tree = ast.parse(_SRC.read_text())
    subclasses = [
        n.name
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == "LoggingService" for b in n.bases)
    ]
    assert subclasses == [], f"a second logging service class is back: {subclasses}"
