"""Progress: a tqdm bar on a terminal, a sparse tracking line in the log, nothing off rank 0.

tqdm redraws on every refresh, which on a terminal is one moving line and in a SLURM
file is one new line per refresh. The bar is therefore console-only and the log gets a
generic tracking line every ``logging.progress.log_every_n_iterations`` iterations:

* ``bar``   -- tqdm on the terminal, plus the tracking line in the log;
* ``lines`` -- the tracking line only (what a SLURM file gets under ``auto``);
* ``off``   -- plain iteration, also what every rank but 0 gets unless
  ``logging.distributed.console_ranks`` is ``all``.

``auto`` picks ``bar`` on a TTY and ``lines`` otherwise.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from spectramr.infrastructure.logging.console import ColoredConsoleFormatter
from spectramr.infrastructure.logging.log_policy import (
    console_enabled_on_this_rank,
    current_level,
    effective_level,
)

try:
    from tqdm import tqdm as _tqdm

    HAVE_TQDM = True
except ImportError:  # pragma: no cover - tqdm is a hard dependency in practice
    _tqdm = None
    HAVE_TQDM = False

__all__ = ["HAVE_TQDM", "Progress", "make_progress", "resolve_progress_mode"]

logger = logging.getLogger("spectramr.progress")


class _TrackingConsole(logging.StreamHandler):
    """Prints tracking lines the root console handler drops (``sinks.level`` above INFO).

    Tracking lines are the one INFO output a warning-level run keeps; at INFO the root
    handler already prints them, so this stays silent to avoid a duplicate.
    """

    @property
    def stream(self):  # resolved per call so a redirected stderr is honoured
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass

    def emit(self, record: logging.LogRecord) -> None:
        if logging.getLogger().getEffectiveLevel() > logging.INFO:
            super().emit(record)


def _install_tracking_console() -> None:
    if any(isinstance(h, _TrackingConsole) for h in logger.handlers):
        return
    handler = _TrackingConsole()
    handler.setFormatter(ColoredConsoleFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_install_tracking_console()

#: tqdm's redraw floor on a terminal, and off one (a forced bar must not write a line per refresh).
_TTY_REFRESH_S = 0.25
_FILE_REFRESH_S = 60.0
_DEFAULT_EVERY = 1500

Postfix = Mapping[str, Any] | Callable[[], Mapping[str, Any]]


def resolve_progress_mode(logging_config: Any | None = None, *, is_tty: bool | None = None) -> str:
    """``bar``, ``lines`` or ``off`` for a ``LoggingConfigSchema`` (``None`` means defaults)."""
    if logging_config is None:
        declared, enabled = "auto", True
    else:
        declared = str(getattr(logging_config.progress.mode, "value", logging_config.progress.mode))
        enabled = (
            logging_config.progress_bar_enabled and not logging_config.progress_bar_no_progress
        )
        if enabled and not logging_config.progress_bar_on_warning:
            applied = current_level()
            level = applied.value if applied else effective_level(logging_config.sinks.level)
            enabled = level < logging.WARNING
    if not enabled or declared == "off" or not console_enabled_on_this_rank():
        return "off"
    if declared == "auto":
        tty = sys.stderr.isatty() if is_tty is None else is_tty
        return "bar" if tty else "lines"
    if declared == "bar" and not HAVE_TQDM:
        logger.warning("tqdm is not installed: progress falls back to log lines")
        return "lines"
    return declared


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Progress:
    """A tqdm-shaped iterator (``for x in p``, ``set_postfix``, ``close``) with a policy behind it.

    ``set_postfix`` takes a mapping or a zero-argument callable. A callable is only
    evaluated when something will be shown, so a per-step formatting loop costs nothing
    on the steps that display nothing. A tracking line is written every ``every`` iterations.
    """

    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        mode: str,
        every: int = _DEFAULT_EVERY,
        desc: str = "progress",
        total: int | None = None,
        leave: bool = True,
    ) -> None:
        if mode not in ("bar", "lines", "off"):
            raise ValueError(f"Unsupported progress mode: {mode!r}; expected bar, lines or off")
        self._iterable = iterable
        self._mode = mode
        self._desc = desc
        self._leave = leave
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self._total = total
        self._n = 0
        self._postfix_source: Postfix | None = None
        self._closed = False
        self._t0 = time.monotonic()
        if every < 1:
            raise ValueError(f"log_every_n_iterations must be >= 1, got {every}")
        self._every = every
        self._refresh = _TTY_REFRESH_S if sys.stderr.isatty() else _FILE_REFRESH_S
        self._next = self._t0 + self._refresh
        self._bar = None
        if mode == "bar":
            self._bar = _tqdm(total=total, desc=desc, leave=leave, mininterval=self._refresh)

    def __iter__(self) -> Iterator[Any]:
        if self._mode == "off":
            yield from self._iterable
            return
        try:
            for item in self._iterable:
                yield item
                self._advance()
        finally:
            self.close()

    def set_postfix(self, data: Postfix) -> None:
        if self._mode == "off":
            return
        # Kept and evaluated when a line is written: evaluating only when THIS call fell on a
        # line step froze the postfix whenever the caller's cadence and `every` disagreed.
        self._postfix_source = data
        now = time.monotonic()
        if self._bar is not None and now >= self._next:
            self._bar.set_postfix_str(self._render_postfix(), refresh=False)
            self._next = now + self._refresh

    def _render_postfix(self) -> str:
        data = self._postfix_source
        if data is None:
            return ""
        values = data() if callable(data) else data
        return ", ".join(f"{k}={v}" for k, v in values.items())

    def _advance(self) -> None:
        self._n += 1
        if self._bar is not None:
            self._bar.update(1)
        finished = self._leave and self._total and self._n >= self._total
        if self._n % self._every == 0 and not finished:  # `close` writes the line at 100%
            self._emit(time.monotonic())

    def _emit(self, now: float, *, done: bool = False) -> None:
        elapsed = now - self._t0
        rate = self._n / elapsed if elapsed > 0 else 0.0
        head = f"{self._desc} {self._n}"
        if self._total:
            head += f"/{self._total} ({100 * self._n / self._total:.1f}%)"
        parts = [head, f"{rate:.2f} it/s" if rate >= 1 else f"{1 / rate:.1f} s/it" if rate else "-"]
        parts.append(f"elapsed {_hms(elapsed)}")
        if self._total and rate and not done:
            parts.append(f"eta {_hms((self._total - self._n) / rate)}")
        postfix = self._render_postfix()
        if postfix:
            parts.append(postfix)
        logger.info(" | ".join(parts) + (" | done" if done else ""))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            self._bar.close()
        if self._mode != "off" and self._leave and self._n:
            self._emit(time.monotonic(), done=True)


def make_progress(
    iterable: Iterable[Any],
    logging_config: Any | None = None,
    *,
    desc: str = "progress",
    total: int | None = None,
    leave: bool = True,
) -> Progress:
    """Wrap ``iterable`` under the run's progress policy."""
    every = _DEFAULT_EVERY if logging_config is None else logging_config.progress.log_every_n_iterations
    return Progress(
        iterable,
        mode=resolve_progress_mode(logging_config),
        every=every,
        desc=desc,
        total=total,
        leave=leave,
    )
