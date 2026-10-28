"""Tests for :mod:`spectramr.infrastructure.logging.console` (formatter + handler install)."""

from __future__ import annotations

import logging

from spectramr.infrastructure.logging.console import (
    ColoredConsoleFormatter,
    install_or_upgrade_colored_console,
    set_console_stream,
)


def _record(level: int = logging.INFO, name: str = "spectramr.x") -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, "hello", None, None)


def test_plain_fallback_without_color():
    line = ColoredConsoleFormatter(use_color=False).format(_record())
    assert "[INFO" in line and "spectramr.x: hello" in line
    assert "\033[" not in line


def test_colored_line_carries_badge_and_message():
    line = ColoredConsoleFormatter(use_color=True).format(_record(logging.WARNING))
    assert "WRN" in line and "hello" in line and "\033[" in line


def test_long_logger_names_are_abbreviated_but_keep_the_leaf():
    fmt = ColoredConsoleFormatter(use_color=False)
    short = fmt._abbreviate_name("spectramr.infrastructure.training.strategies.diffusion")
    assert short.endswith(".diffusion") and len(short) < 40


def test_installs_one_colored_handler_when_missing():
    logger = logging.Logger("t-install")
    assert install_or_upgrade_colored_console(logger, logging.WARNING, stream="stderr") is True
    (handler,) = logger.handlers
    assert isinstance(handler.formatter, ColoredConsoleFormatter)
    assert handler.level == logging.WARNING


def test_silent_mode_never_adds_a_sink():
    logger = logging.Logger("t-silent")
    assert (
        install_or_upgrade_colored_console(logger, logging.INFO, install_if_missing=False) is False
    )
    assert logger.handlers == []


def test_a_plain_handler_is_upgraded_in_place_not_stacked():
    logger = logging.Logger("t-upgrade")
    plain = logging.StreamHandler()
    logger.addHandler(plain)
    assert install_or_upgrade_colored_console(logger, logging.INFO) is True
    assert logger.handlers == [plain]
    assert isinstance(plain.formatter, ColoredConsoleFormatter)


def test_second_call_is_a_no_op():
    logger = logging.Logger("t-idem")
    install_or_upgrade_colored_console(logger, logging.INFO, stream="stderr")
    assert install_or_upgrade_colored_console(logger, logging.INFO, stream="stderr") is False
    assert len(logger.handlers) == 1


def test_a_file_handler_is_not_treated_as_the_console(tmp_path):
    logger = logging.Logger("t-file")
    logger.addHandler(logging.FileHandler(tmp_path / "x.log"))
    install_or_upgrade_colored_console(logger, logging.INFO, stream="stderr")
    kinds = sorted(type(h).__name__ for h in logger.handlers)
    assert kinds == ["FileHandler", "StreamHandler"]


def _out_err(monkeypatch):
    import io

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    return out, err


def _emit_all(logger):
    logger.setLevel(logging.DEBUG)
    logger.debug("d-line")
    logger.info("i-line")
    logger.warning("w-line")
    logger.error("e-line")


def test_split_sends_info_to_stdout_and_warnings_to_stderr(monkeypatch):
    out, err = _out_err(monkeypatch)
    logger = logging.Logger("t-split")
    install_or_upgrade_colored_console(logger, logging.DEBUG, stream="split")
    _emit_all(logger)
    assert "d-line" in out.getvalue() and "i-line" in out.getvalue()
    assert "w-line" not in out.getvalue() and "e-line" not in out.getvalue()
    assert "w-line" in err.getvalue() and "e-line" in err.getvalue()
    assert "i-line" not in err.getvalue()


def test_a_line_is_never_written_twice_in_split_mode(monkeypatch):
    out, err = _out_err(monkeypatch)
    logger = logging.Logger("t-once")
    install_or_upgrade_colored_console(logger, logging.INFO, stream="split")
    logger.warning("only-once")
    assert (out.getvalue() + err.getvalue()).count("only-once") == 1


def test_stdout_mode_keeps_err_off_stderr(monkeypatch):
    out, err = _out_err(monkeypatch)
    logger = logging.Logger("t-stdout")
    install_or_upgrade_colored_console(logger, logging.INFO, stream="stdout")
    _emit_all(logger)
    assert "e-line" in out.getvalue() and err.getvalue() == ""


def test_stderr_mode_keeps_everything_off_stdout(monkeypatch):
    out, err = _out_err(monkeypatch)
    logger = logging.Logger("t-stderr")
    install_or_upgrade_colored_console(logger, logging.INFO, stream="stderr")
    _emit_all(logger)
    assert "i-line" in err.getvalue() and out.getvalue() == ""


def test_unknown_stream_raises():
    import pytest

    with pytest.raises(ValueError, match="Unsupported console stream"):
        install_or_upgrade_colored_console(logging.Logger("t-bad"), logging.INFO, stream="stdin")


def test_set_console_stream_repoints_only_managed_handlers(monkeypatch):
    _out_err(monkeypatch)
    logger = logging.Logger("t-repoint")
    foreign = logging.StreamHandler()
    logger.addHandler(foreign)
    install_or_upgrade_colored_console(logger, logging.INFO, stream="split")
    assert logger.handlers == [foreign]  # a caller's handler is upgraded, not stacked on
    mine = logging.Logger("t-repoint2")
    install_or_upgrade_colored_console(mine, logging.INFO, stream="split")
    assert set_console_stream(mine, "stdout", logging.INFO) is True
    assert len(mine.handlers) == 1
    assert set_console_stream(mine, "stdout", logging.INFO) is False  # already there
    assert set_console_stream(logger, "stdout", logging.INFO) is False  # nothing of ours
