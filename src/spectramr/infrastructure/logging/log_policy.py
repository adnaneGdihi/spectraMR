"""The one owner of the process's log level and console handler.

Before this module the root level had five writers (CLI bootstrap hard-coded
INFO, the rank clamp, ``LoggingService.__init__``, ``LoggingService.setup`` and a
one-off helper in ``main.py``), and only ``train`` ever applied the level the user
declared before ``setup`` ran -- so ``logging.sinks.level: warning`` still printed
INFO from every other verb and from everything logged before the container built.

Precedence, highest first: ``--log-level`` > ``SPECTRAMR_LOG_LEVEL`` > the config's
``logging.sinks.level`` > INFO. A secondary rank of a distributed launch is floored
at WARNING whatever was asked: only the informational narration is de-duplicated,
per-rank warnings survive.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from spectramr.core import env, env_names
from spectramr.infrastructure.logging.console import (
    ColoredConsoleFormatter,
    install_or_upgrade_colored_console,
    set_console_stream,
)
from spectramr.infrastructure.logging.torch_logging import apply_level_to_torch

__all__ = [
    "LEVELS",
    "ConsolePolicy",
    "ResolvedLevel",
    "adopt_logging_policy",
    "apply_config_level",
    "apply_level",
    "apply_logging_config",
    "bootstrap_console_logging",
    "configure_entry",
    "console_enabled_on_this_rank",
    "console_stream",
    "current_level",
    "current_policy",
    "effective_level",
    "parse_level",
    "quiet_secondary_ranks",
    "rank_floor",
    "rank_log_path",
    "resolve_level",
]

logger = logging.getLogger(__name__)

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


@dataclass(frozen=True)
class ResolvedLevel:
    """A level plus where it came from, so provenance can say who decided."""

    name: str
    value: int
    source: str  # "cli" | "env" | "config" | "default"


# What the CLI entry point was told; consulted by every later resolution so a
# user-dictated level is not undone by the config load or ``LoggingService.setup``.
_entry: dict[str, str | None] = {"cli": None, "env": None}
_current: ResolvedLevel | None = None


@dataclass(frozen=True)
class ConsolePolicy:
    """The distributed and writing policy: who logs, at what floor, to which stream.

    The defaults are what a run gets before its config loads, so the first lines of
    a torchrun launch are already held to rank 0.
    """

    console_ranks: str = "rank0"
    secondary_level: int = logging.ERROR
    stream: str = "split"
    file_ranks: str = "rank0"


_policy = ConsolePolicy()


def parse_level(level: object) -> int:
    """Return the stdlib level for ``'warning'``, ``LogLevel.WARNING`` or ``30``; raise otherwise."""
    if isinstance(level, bool):
        raise ValueError(f"Unsupported log level: {level!r}")
    if isinstance(level, int):
        return level
    key = str(getattr(level, "value", level)).strip().lower()
    if key not in LEVELS:
        raise ValueError(f"Unsupported log level: {level!r}; expected one of {sorted(LEVELS)}")
    return LEVELS[key]


def rank_floor(level: int, floor: int | None = None) -> int:
    """Return ``level``, but never below the policy's floor on a secondary rank.

    Rank 0, single-process runs and ``console_ranks: all`` get ``level`` back
    unchanged. ``floor`` overrides the policy's ``secondary_level``.
    """
    if not env.is_secondary_rank():
        return level
    if floor is None:
        if _policy.console_ranks == "all":
            return level
        floor = _policy.secondary_level
    return max(level, floor)


def console_enabled_on_this_rank() -> bool:
    """False on a secondary rank unless the policy lets every rank log (progress follows this)."""
    return not env.is_secondary_rank() or _policy.console_ranks == "all"


def console_stream() -> str:
    """The declared console stream: ``split``, ``stdout`` or ``stderr``."""
    return _policy.stream


def rank_log_path(path: str) -> str | None:
    """Where this rank writes the text log, or ``None`` if it must not write one.

    Rank 0 keeps ``path``. Another rank writes nothing under ``file_ranks: rank0`` and
    its own ``<stem>.rank<N><ext>`` under ``all``, so ranks never interleave in one file.
    """
    rank = env.distributed_rank()
    if rank in (None, 0):
        return path
    if _policy.file_ranks != "all":
        return None
    stem, ext = os.path.splitext(path)
    return f"{stem}.rank{rank}{ext}"


def current_policy() -> dict[str, str]:
    """The policy in force, for provenance."""
    return {
        "console_ranks": _policy.console_ranks,
        "secondary_level": logging.getLevelName(_policy.secondary_level).lower(),
        "console_stream": _policy.stream,
        "file_ranks": _policy.file_ranks,
    }


def adopt_logging_policy(logging_config: Any) -> ConsolePolicy:
    """Read the distributed and stream policy from a loaded ``LoggingConfigSchema``."""
    global _policy
    dist = logging_config.distributed
    _policy = ConsolePolicy(
        console_ranks=str(getattr(dist.console_ranks, "value", dist.console_ranks)),
        secondary_level=parse_level(dist.secondary_level),
        stream=str(
            getattr(
                logging_config.sinks.console_stream, "value", logging_config.sinks.console_stream
            )
        ),
        file_ranks=str(getattr(dist.file_ranks, "value", dist.file_ranks)),
    )
    return _policy


def resolve_level(config: object | None = None, *, config_declared: bool = False) -> ResolvedLevel:
    """Apply the precedence rule; raises on an unknown level from any source."""
    for source, raw in (("cli", _entry["cli"]), ("env", _entry["env"])):
        if raw:
            value = parse_level(raw)
            return ResolvedLevel(logging.getLevelName(value).lower(), value, source)
    if config is not None:
        value = parse_level(config)
        return ResolvedLevel(
            logging.getLevelName(value).lower(), value, "config" if config_declared else "default"
        )
    return ResolvedLevel("info", logging.INFO, "default")


def effective_level(config: object) -> int:
    """The int ``LoggingService.setup`` must apply for a config level, rank-floored."""
    return rank_floor(resolve_level(config).value)


def current_level() -> ResolvedLevel | None:
    """The level last applied, for provenance; ``None`` before any application."""
    return _current


def apply_level(resolved: ResolvedLevel) -> int:
    """Put ``resolved`` on the root logger, its handlers and torch; return the applied int."""
    global _current
    level = rank_floor(resolved.value)
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        if handler.level != logging.NOTSET:
            handler.setLevel(level)
    apply_level_to_torch(level)
    _current = ResolvedLevel(logging.getLevelName(level).lower(), level, resolved.source)
    return level


def apply_config_level(config: object, *, declared: bool = False) -> ResolvedLevel:
    """Apply the config's ``logging.sinks.level`` under the precedence rule."""
    resolved = resolve_level(config, config_declared=declared)
    apply_level(resolved)
    return _current or resolved


def apply_logging_config(logging_config: Any) -> ResolvedLevel:
    """Apply a loaded ``LoggingConfigSchema``; the call every config-loading verb makes."""
    adopt_logging_policy(logging_config)
    sinks = logging_config.sinks
    resolved = apply_config_level(sinks.level, declared="level" in sinks.model_fields_set)
    if sinks.to_console and not sinks.silent:
        set_console_stream(logging.root, _policy.stream, resolved.value)
    return resolved


def bootstrap_console_logging(level: int | str = logging.INFO, force: bool = False) -> None:
    """Install the colored console handler on the root logger. Idempotent.

    ``force`` removes every existing root handler first; use it only to recover
    from a misconfigured handler set. An existing colored handler is kept, and the
    level is only ever lowered here -- raising it is :func:`apply_level`'s job.
    """
    root = logging.root
    resolved_level = rank_floor(parse_level(level))
    has_colored = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and isinstance(h.formatter, ColoredConsoleFormatter)
        for h in root.handlers
    )
    if has_colored and not force:
        if root.level == logging.NOTSET or root.level > resolved_level:
            root.setLevel(resolved_level)
        return
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
            with contextlib.suppress(Exception):  # close() is best-effort
                h.close()
    install_or_upgrade_colored_console(root, resolved_level, stream=_policy.stream)
    root.setLevel(resolved_level)


def configure_entry(cli_level: str | None = None) -> ResolvedLevel:
    """CLI entry: install the console, record what the user dictated, apply it.

    Called once after argument parsing. ``cli_level`` is ``--log-level`` (or
    ``debug`` for the legacy ``--debug`` flag); ``SPECTRAMR_LOG_LEVEL`` is read here.
    """
    _entry["cli"] = cli_level
    _entry["env"] = os.environ.get(env_names.SPECTRAMR_LOG_LEVEL) or None
    resolved = resolve_level()  # raises on a bad flag/env value before anything runs
    bootstrap_console_logging(logging.INFO)
    logging.captureWarnings(True)  # warnings.warn then obeys the rank floor and the stream split
    apply_level(resolved)
    return _current or resolved


def quiet_secondary_ranks(level: int | None = None) -> bool:
    """Hold the root level on every rank but 0 to the policy's floor. Idempotent.

    Returns ``True`` if applied (a non-zero rank under ``console_ranks: rank0``),
    ``False`` otherwise, so the gating is testable and a single-process run is
    provably untouched. ``level`` overrides the policy's ``secondary_level``.
    """
    if not env.is_secondary_rank() or (_policy.console_ranks == "all" and level is None):
        return False
    logging.root.setLevel(_policy.secondary_level if level is None else level)
    return True
