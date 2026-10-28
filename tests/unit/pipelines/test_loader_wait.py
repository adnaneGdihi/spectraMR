import logging

import pytest

from spectramr.pipelines.loader_wait import LoaderWaitMonitor


def _feed(monitor, wait, wall, steps):
    return [monitor.record(wait, wall) for _ in range(steps)]


def test_starved_window_warns(caplog):
    monitor = LoaderWaitMonitor(window=10)
    with caplog.at_level(logging.WARNING):
        out = _feed(monitor, 0.9, 1.0, 10)
    assert out[-1] == pytest.approx(0.9)
    assert "starved" in caplog.text


def test_fed_window_is_silent(caplog):
    monitor = LoaderWaitMonitor(window=10)
    with caplog.at_level(logging.WARNING):
        out = _feed(monitor, 0.01, 1.0, 10)
    assert out[-1] == pytest.approx(0.01)
    assert caplog.text == ""


def test_returns_none_inside_window():
    assert LoaderWaitMonitor(window=5).record(1.0, 1.0) is None


def test_warnings_are_capped(caplog):
    monitor = LoaderWaitMonitor(window=1)
    with caplog.at_level(logging.WARNING):
        _feed(monitor, 1.0, 1.0, 20)
    assert caplog.text.count("starved") == 5


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        LoaderWaitMonitor(window=0)


def test_totals_cover_the_whole_run_not_just_closed_windows():
    monitor = LoaderWaitMonitor(window=50)
    _feed(monitor, 0.9, 1.0, 7)  # never closes a window, so never warns
    assert monitor.as_result_fields() == {
        "loader_wait": {"fraction": 0.9, "wait_s": 6.3, "wall_s": 7.0, "steps": 7}
    }


def test_no_timed_step_leaves_the_summary_key_out():
    assert LoaderWaitMonitor().as_result_fields() == {}
