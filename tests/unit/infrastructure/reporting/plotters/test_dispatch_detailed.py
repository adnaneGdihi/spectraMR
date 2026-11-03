"""Paired tests for ``plotters.dispatch_detailed``.

``dispatch`` collapsed three unrelated situations into a bare ``None``: the
figure id was not registered, the plotter raised, or the plotter reported it had
nothing to draw. All three then wrote the same ``status: skipped`` manifest
entry. They call for different responses -- fix the id, fix the plotter, or
accept that this run has no such data -- so they are recorded separately.

This is what makes a report after `predict` comparable to one after training.
The requested figure set is identical for both (it derives from the task preset,
never from the caller), so the only difference is which figures had data; unless
that is written down, the difference reads as an unexplained absence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spectramr.infrastructure.reporting import plotters
from spectramr.infrastructure.reporting.plotters import (
    SKIP_NO_DATA,
    SKIP_RAISED,
    SKIP_UNREGISTERED,
    dispatch,
    dispatch_detailed,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"metric": ["psnr"], "value": [30.0], "split": ["test"]})


class TestTheThreeCausesAreDistinguished:
    def test_an_unregistered_id_says_so(self, df, tmp_path):
        out = dispatch_detailed(df, ["definitely_not_a_figure"], tmp_path)
        outcome = out["definitely_not_a_figure"]
        assert outcome.path is None
        assert outcome.reason == SKIP_UNREGISTERED

    def test_a_plotter_with_nothing_to_draw_says_no_data(
        self, df, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(plotters, "get", lambda fid: (lambda d, p, **k: None))
        out = dispatch_detailed(df, ["fig_pretend"], tmp_path)
        assert out["fig_pretend"].reason == SKIP_NO_DATA

    def test_a_raising_plotter_carries_the_error(self, df, tmp_path, monkeypatch):
        def _boom(d, p, **k):
            raise ValueError("bad axis")

        monkeypatch.setattr(plotters, "get", lambda fid: _boom)
        out = dispatch_detailed(df, ["fig_pretend"], tmp_path)
        reason = out["fig_pretend"].reason
        assert reason.startswith(SKIP_RAISED)
        assert "ValueError" in reason and "bad axis" in reason

    def test_no_data_and_raised_are_not_the_same_reason(self, df, tmp_path, monkeypatch):
        """The distinction is the point; a shared reason would be no better than None."""
        monkeypatch.setattr(plotters, "get", lambda fid: (lambda d, p, **k: None))
        quiet = dispatch_detailed(df, ["f"], tmp_path)["f"].reason

        def _boom(d, p, **k):
            raise ValueError("x")

        monkeypatch.setattr(plotters, "get", lambda fid: _boom)
        loud = dispatch_detailed(df, ["f"], tmp_path)["f"].reason
        assert quiet != loud
        assert SKIP_UNREGISTERED not in (quiet, loud)


class TestASuccessfulFigureCarriesNoReason:
    def test_ok_outcome(self, df, tmp_path, monkeypatch):
        def _draw(d, p, **k):
            out = Path(p)
            out.write_bytes(b"%PDF-1.4\n")
            return out

        monkeypatch.setattr(plotters, "get", lambda fid: _draw)
        outcome = dispatch_detailed(df, ["fig_pretend"], tmp_path)["fig_pretend"]
        assert outcome.ok
        assert outcome.reason is None
        assert outcome.path is not None and outcome.path.exists()


class TestDispatchDelegatesRatherThanDuplicating:
    """Two loops would drift, and the one that drifted is the one nothing renders from."""

    def test_dispatch_returns_the_same_paths(self, df, tmp_path, monkeypatch):
        def _draw(d, p, **k):
            out = Path(p)
            out.write_bytes(b"%PDF-1.4\n")
            return out

        monkeypatch.setattr(plotters, "get", lambda fid: _draw)
        flat = dispatch(df, ["a", "b"], tmp_path)
        detailed = dispatch_detailed(df, ["a", "b"], tmp_path)
        assert flat == {k: o.path for k, o in detailed.items()}

    def test_dispatch_still_returns_none_for_a_skip(self, df, tmp_path):
        assert dispatch(df, ["definitely_not_a_figure"], tmp_path) == {
            "definitely_not_a_figure": None
        }

    def test_dispatch_has_no_loop_of_its_own(self):
        """Guards the delegation structurally, not just by result agreement."""
        import inspect

        src = inspect.getsource(dispatch)
        assert "dispatch_detailed(" in src
        assert "try:" not in src, "dispatch grew its own error handling again"


def test_a_plotter_restyling_matplotlib_leaves_rcparams_as_found(df, tmp_path, monkeypatch):
    import matplotlib as mpl

    from spectramr.infrastructure.reporting.style import use_default_style

    def _restyle(d, p, **k):
        use_default_style("ieee")
        mpl.rcParams["lines.linewidth"] = 9.0
        return None

    monkeypatch.setattr(plotters, "get", lambda fid: _restyle)
    before = dict(mpl.rcParams)
    dispatch_detailed(df, ["fig_pretend"], tmp_path)
    assert dict(mpl.rcParams) == before
