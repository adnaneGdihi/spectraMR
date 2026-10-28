"""Tests for :mod:`spectramr.infrastructure.logging.progress`.

A tqdm bar in a SLURM file is one new line per refresh, and ``set_postfix`` refreshes on
every call, so a per-step postfix wrote a line per step. These tests pin the policy that
replaces it: a bar only on a terminal, one log line per interval otherwise, nothing off
rank 0.
"""

from __future__ import annotations

import logging

import pytest

from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.infrastructure.logging import log_policy, progress
from spectramr.infrastructure.logging.progress import (
    Progress,
    make_progress,
    resolve_progress_mode,
)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    log_policy._policy = log_policy.ConsolePolicy()
    yield
    log_policy._policy = log_policy.ConsolePolicy()


class _Clock:
    """A controllable monotonic clock, so cadence is asserted without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(progress.time, "monotonic", c)
    return c


def _cfg(**blocks):
    return LoggingConfigSchema(**blocks)


class TestResolveMode:
    def test_auto_is_a_bar_on_a_tty_and_lines_otherwise(self):
        assert resolve_progress_mode(_cfg(), is_tty=True) == "bar"
        assert resolve_progress_mode(_cfg(), is_tty=False) == "lines"

    def test_none_config_means_defaults(self):
        assert resolve_progress_mode(None, is_tty=False) == "lines"

    def test_on_warning_false_turns_progress_off_at_warning_and_above(self, monkeypatch):
        monkeypatch.setattr(log_policy, "_current", None)
        warn = _cfg(sinks={"level": "warning"}, progress_bar_on_warning=False)
        info = _cfg(sinks={"level": "info"}, progress_bar_on_warning=False)
        keep = _cfg(sinks={"level": "warning"})
        assert resolve_progress_mode(warn, is_tty=True) == "off"
        assert resolve_progress_mode(info, is_tty=True) == "bar"
        assert resolve_progress_mode(keep, is_tty=True) == "bar"

    @pytest.mark.parametrize("mode", ["bar", "lines", "off"])
    def test_an_explicit_mode_wins_over_the_tty(self, mode):
        cfg = _cfg(progress={"mode": mode})
        assert resolve_progress_mode(cfg, is_tty=True) == mode
        assert resolve_progress_mode(cfg, is_tty=False) == mode

    def test_progress_bar_enabled_false_turns_progress_off(self):
        assert resolve_progress_mode(_cfg(progress_bar_enabled=False), is_tty=True) == "off"

    def test_progress_bar_no_progress_turns_progress_off(self):
        assert resolve_progress_mode(_cfg(progress_bar_no_progress=True), is_tty=True) == "off"

    def test_other_ranks_get_nothing_by_default(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        assert resolve_progress_mode(_cfg(), is_tty=True) == "off"

    def test_console_ranks_all_lets_every_rank_show_progress(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        log_policy._policy = log_policy.ConsolePolicy(console_ranks="all")
        assert resolve_progress_mode(_cfg(), is_tty=False) == "lines"

    def test_bar_without_tqdm_degrades_loudly(self, monkeypatch, caplog):
        monkeypatch.setattr(progress, "HAVE_TQDM", False)
        with caplog.at_level(logging.WARNING, logger="spectramr.progress"):
            mode = resolve_progress_mode(_cfg(progress={"mode": "bar"}), is_tty=True)
        assert mode == "lines"
        assert "tqdm is not installed" in caplog.text


class TestLines:
    def _run(self, caplog, steps, every=1500, **kw):
        p = Progress(range(steps), mode="lines", every=every, desc="Training", **kw)
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for _ in p:
                pass
        return [r.getMessage() for r in caplog.records if r.name == "spectramr.progress"]

    def test_one_line_per_n_iterations_not_per_step(self, caplog):
        lines = self._run(caplog, steps=100, every=30)
        assert [m.split(" | ")[0] for m in lines if "done" not in m] == [
            "Training 30/100 (30.0%)",
            "Training 60/100 (60.0%)",
            "Training 90/100 (90.0%)",
        ]

    def test_default_cadence_is_1500(self, caplog):
        lines = self._run(caplog, steps=3000, leave=False)
        assert len(lines) == 2 and lines[0].startswith("Training 1500/3000")

    def test_cadence_ignores_wall_clock(self, clock, caplog):
        p = Progress(range(10), mode="lines", every=5, desc="T", leave=False)
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for _ in p:
                clock.now += 10_000.0
        assert len(caplog.records) == 2

    def test_line_carries_count_percent_rate_and_eta(self, clock, caplog):
        p = Progress(range(10), mode="lines", every=5, desc="Training")
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for _ in p:
                clock.now += 60.0
        first = caplog.records[0].getMessage()
        assert first.startswith("Training 5/10 (50.0%)")
        assert "s/it" in first and "eta" in first and "elapsed" in first

    def test_a_final_summary_is_written_when_left(self, caplog):
        lines = self._run(caplog, steps=4, every=1000)
        assert len(lines) == 1 and lines[0].endswith("| done")

    def test_completion_is_one_line_not_two(self, caplog):
        lines = self._run(caplog, steps=3, every=3)
        at_full = [m for m in lines if "3/3" in m]
        assert len(at_full) == 1 and at_full[0].endswith("| done")

    def test_no_summary_for_a_transient_bar(self, caplog):
        assert self._run(caplog, steps=4, every=1000, leave=False) == []

    def test_every_must_be_positive(self):
        with pytest.raises(ValueError, match="log_every_n_iterations"):
            Progress(range(1), mode="lines", every=0)

    def test_total_comes_from_len_or_the_argument(self):
        p = Progress(iter(range(3)), mode="lines", total=3, desc="x")
        assert p._total == 3
        assert Progress(iter(range(3)), mode="lines")._total is None


class TestPostfix:
    def test_a_callable_is_not_evaluated_between_lines(self):
        calls = []
        p = Progress(range(10), mode="lines", every=100)
        for _ in p:
            p.set_postfix(lambda: calls.append(1) or {"loss": 1.0})
        assert len(calls) == 1  # once, for the closing line at 100%

    def test_it_is_evaluated_only_for_the_step_that_logs(self, caplog):
        calls = []
        p = Progress(range(6), mode="lines", every=3, desc="T")
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for _ in p:
                p.set_postfix(lambda: calls.append(1) or {"loss": "0.5000"})
        assert len(calls) == 2
        assert any("3/6" in r.getMessage() and "loss=0.5000" in r.getMessage() for r in caplog.records)

    def test_a_line_shows_the_latest_postfix_when_the_cadences_disagree(self, caplog):
        """PLANTED: log every 2500, line every 1500 froze 7500's values until 15000."""
        p = Progress(range(15), mode="lines", every=3, total=15, desc="T")
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for i in p:
                if i % 5 == 0:
                    value = i
                    p.set_postfix(lambda v=value: {"loss": v})
        lines = {r.getMessage().split()[1]: r.getMessage() for r in caplog.records}
        assert "loss=5" in lines["9/15"] and "loss=10" in lines["12/15"]

    def test_a_plain_mapping_is_accepted(self, caplog):
        p = Progress(range(3), mode="lines", every=1)
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for _ in p:
                p.set_postfix({"psnr": "30.1"})
        assert any("psnr=30.1" in r.getMessage() for r in caplog.records)


class TestOffAndModes:
    def test_off_is_plain_iteration_and_silent(self, caplog):
        p = Progress(range(5), mode="off")
        with caplog.at_level(logging.DEBUG, logger="spectramr.progress"):
            assert list(p) == [0, 1, 2, 3, 4]
            p.set_postfix({"a": 1})
            p.close()
        assert caplog.records == []

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported progress mode"):
            Progress(range(1), mode="fancy")

    def test_close_is_idempotent_and_survives_a_break(self, clock, caplog):
        p = Progress(range(5), mode="lines", every=1000)
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            for i in p:
                if i == 2:
                    break
            p.close()
            p.close()
        assert len([r for r in caplog.records if "done" in r.getMessage()]) <= 1

    def test_empty_iterable_writes_nothing(self, caplog):
        with caplog.at_level(logging.INFO, logger="spectramr.progress"):
            assert list(make_progress((), None, desc="Training")) == []
        assert caplog.records == []


class TestBar:
    def test_bar_mode_wraps_tqdm_without_a_refresh_per_postfix(self, monkeypatch):
        seen = {}

        class _FakeBar:
            def __init__(self, **kw):
                seen["init"] = kw
                self.updates = 0
                self.refreshes = []

            def update(self, n):
                self.updates += n

            def set_postfix_str(self, s, refresh=True):
                self.refreshes.append(refresh)

            def close(self):
                seen["closed"] = True

        monkeypatch.setattr(progress, "_tqdm", _FakeBar)
        c = _Clock()
        monkeypatch.setattr(progress.time, "monotonic", c)
        p = Progress(range(3), mode="bar", every=2, desc="Training")
        for _ in p:
            c.now += 100.0
            p.set_postfix({"loss": 1})
        bar = p._bar
        assert bar.updates == 3 and seen["closed"] is True
        assert bar.refreshes and not any(bar.refreshes)  # never refresh=True
        assert seen["init"]["mininterval"] == progress._FILE_REFRESH_S  # off a TTY: no line per refresh


def test_make_progress_reads_the_cadence_from_the_config():
    p = make_progress(
        range(2), _cfg(progress={"mode": "lines", "log_every_n_iterations": 7}), desc="x"
    )
    assert p._every == 7


class TestTrackingLinesSurviveWarningLevel:
    def test_tracking_line_prints_when_root_is_warning(self, monkeypatch, capsys):
        root = logging.getLogger()
        monkeypatch.setattr(root, "level", logging.WARNING)
        for _ in Progress(range(4), mode="lines", every=2, desc="train"):
            pass
        assert "train 2" in capsys.readouterr().err

    def test_no_duplicate_when_root_is_info(self, monkeypatch, capsys):
        monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
        for _ in Progress(range(4), mode="lines", every=2, desc="train"):
            pass
        assert capsys.readouterr().err.count("train 2") <= 1
