"""Tests for the v6.1 logging hygiene helpers (banner / phase / progress)."""

from __future__ import annotations

import logging

from spectramr.infrastructure.logging import banner, make_progress, phase


def test_banner_emits_three_lines(caplog) -> None:
    caplog.set_level(logging.INFO)
    banner("hello", logger=logging.getLogger("testbanner"))
    msgs = [r.message for r in caplog.records if r.name == "testbanner"]
    assert len(msgs) == 3
    assert msgs[0] == msgs[2]  # top + bottom borders match
    assert "hello" in msgs[1]


def test_phase_emits_banner_plus_elapsed(caplog) -> None:
    caplog.set_level(logging.INFO)
    log = logging.getLogger("testphase")
    with phase("loading", logger=log):
        x = 1 + 1  # noqa: F841
    msgs = [r.message for r in caplog.records if r.name == "testphase"]
    assert any("loading" in m for m in msgs)
    assert any("finished in" in m for m in msgs)


def test_make_progress_yields_all_items() -> None:
    assert list(make_progress(range(10), desc="t")) == list(range(10))


def test_make_progress_with_known_total() -> None:
    assert list(make_progress(range(5), total=5, desc="t")) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Additional coverage for spectramr.infrastructure.logging.banners
# (the module was listed as unreferenced; these tests wire it explicitly)
# ---------------------------------------------------------------------------


import pytest  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize(
    "width,char",
    [
        (40, "="),
        (60, "-"),
        (20, "*"),
    ],
)
def test_banner_custom_width_and_char(width: int, char: str, caplog) -> None:
    """Banner borders should use the requested char and have the requested width."""
    from spectramr.infrastructure.logging.banners import banner as _banner

    log = logging.getLogger(f"testbanner_custom_{width}_{char}")
    caplog.set_level(logging.INFO, logger=log.name)
    _banner("T", width=width, char=char, logger=log)
    borders = [r.message for r in caplog.records if r.name == log.name and r.message[0] == char]
    assert len(borders) >= 2
    for b in borders:
        assert len(b) == width


@pytest.mark.unit
def test_banner_debug_level(caplog) -> None:
    """Banner should be emittable at DEBUG level."""
    from spectramr.infrastructure.logging.banners import banner as _banner

    log = logging.getLogger("testbanner_debug")
    caplog.set_level(logging.DEBUG, logger=log.name)
    _banner("debug_banner", level=logging.DEBUG, logger=log)
    msgs = [r.message for r in caplog.records if r.name == log.name]
    assert len(msgs) == 3


@pytest.mark.unit
def test_phase_elapsed_time_non_negative(caplog) -> None:
    """phase() elapsed-time message should report a non-negative float."""
    import re

    from spectramr.infrastructure.logging.banners import phase as _phase

    log = logging.getLogger("testphase_elapsed")
    caplog.set_level(logging.INFO, logger=log.name)
    with _phase("work", logger=log):
        pass
    elapsed_msgs = [
        r.message for r in caplog.records if "finished in" in r.message
    ]
    assert elapsed_msgs, "No elapsed-time message emitted"
    m = re.search(r"(\d+\.\d+)s", elapsed_msgs[0])
    assert m is not None, f"Could not parse elapsed time from: {elapsed_msgs[0]}"
    assert float(m.group(1)) >= 0.0


@pytest.mark.unit
def test_phase_reraises_exception(caplog) -> None:
    """phase() context manager must propagate exceptions unchanged."""
    from spectramr.infrastructure.logging.banners import phase as _phase

    log = logging.getLogger("testphase_reraise")
    caplog.set_level(logging.INFO, logger=log.name)
    with pytest.raises(ValueError, match="sentinel"):
        with _phase("crash_phase", logger=log):
            raise ValueError("sentinel")


@pytest.mark.unit
def test_banner_edge_empty_title(caplog) -> None:
    """banner() with an empty title should still emit 3 log lines."""
    from spectramr.infrastructure.logging.banners import banner as _banner

    log = logging.getLogger("testbanner_empty")
    caplog.set_level(logging.INFO, logger=log.name)
    _banner("", logger=log)
    msgs = [r.message for r in caplog.records if r.name == log.name]
    assert len(msgs) == 3
