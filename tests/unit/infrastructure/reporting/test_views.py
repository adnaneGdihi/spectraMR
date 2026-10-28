"""Checkpoint views: which validation pass a report reads each run at."""

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.metric_names import annotate_frame
from spectramr.infrastructure.reporting.views import checkpoint_steps, snapshot


def _run(run_id, steps, selected=None):
    rows = [
        {"run_id": run_id, "method": run_id, "split": "val", "step": s, "metric": m, "value": v}
        for s in steps
        for m, v in (("val_psnr", 20.0 + s / 100), ("val_ssim", 0.5 + s / 10_000))
    ]
    if selected is not None:
        rows.append(
            {
                "run_id": run_id,
                "method": run_id,
                "split": "final",
                "step": -1,
                "metric": "early_stopping_best_iteration",
                "value": float(selected),
            }
        )
    # per-metric "best" rows: must never be what a view reads
    rows.append(
        {
            "run_id": run_id,
            "method": run_id,
            "split": "best",
            "step": -1,
            "metric": "val_psnr",
            "value": 99.0,
        }
    )
    return rows


def _frame():
    return annotate_frame(
        pd.DataFrame(_run("a", (100, 200, 300), selected=200) + _run("b", (100, 200)))
    )


def test_selected_reads_the_early_stopping_pass():
    assert checkpoint_steps(_frame(), "selected") == {"a": 200, "b": 200}


def test_final_reads_each_runs_last_pass():
    assert checkpoint_steps(_frame(), "final") == {"a": 300, "b": 200}


def test_common_reads_one_budget_for_every_run():
    assert checkpoint_steps(_frame(), "common") == {"a": 200, "b": 200}


def test_a_selected_iteration_between_passes_takes_the_pass_before_it():
    df = annotate_frame(pd.DataFrame(_run("a", (100, 200, 300), selected=250)))
    assert checkpoint_steps(df, "selected") == {"a": 200}


def test_a_missing_selection_falls_back_to_final_and_says_so(caplog):
    import logging

    from spectramr.infrastructure.reporting import views

    views._WARNED.clear()
    df = annotate_frame(pd.DataFrame(_run("b", (100, 200))))
    with caplog.at_level(logging.WARNING):
        assert checkpoint_steps(df, "selected") == {"b": 200}
    assert any("early_stopping_best_iteration" in r.getMessage() for r in caplog.records)


def test_a_single_pass_run_falls_back_silently(caplog):
    """A validate-only grading run has one pass, so selected and final are the same pass."""
    import logging

    from spectramr.infrastructure.reporting import views

    views._WARNED.clear()
    df = annotate_frame(pd.DataFrame(_run("inference", (60000,))))
    with caplog.at_level(logging.WARNING):
        assert checkpoint_steps(df, "selected") == {"inference": 60000}
    assert not caplog.records


def test_a_single_pass_after_its_selection_still_says_so(caplog):
    import logging

    from spectramr.infrastructure.reporting import views

    views._WARNED.clear()
    df = annotate_frame(pd.DataFrame(_run("a", (300,), selected=100)))
    with caplog.at_level(logging.WARNING):
        assert checkpoint_steps(df, "selected") == {"a": 300}
    assert any("precedes every pass" in r.getMessage() for r in caplog.records)


def test_snapshot_reads_every_metric_at_one_pass_never_the_best_block():
    snap = snapshot(_frame(), "selected")
    a = snap[snap["run_id"] == "a"].set_index("metric")["value"]
    assert a["val_psnr"] == pytest.approx(22.0)
    assert a["val_ssim"] == pytest.approx(0.52)
    assert 99.0 not in set(snap["value"])


def test_unknown_view_raises():
    with pytest.raises(ValueError, match="checkpoint view"):
        checkpoint_steps(_frame(), "best")


def test_frames_without_run_id_are_one_run():
    df = pd.DataFrame(_run("x", (1, 2))).drop(columns=["run_id", "method"])
    assert checkpoint_steps(df, "final") == {"run": 2}

def test_common_names_a_run_with_no_pass_inside_the_budget(caplog):
    import logging

    df = annotate_frame(pd.DataFrame(_run("a", (100, 200)) + _run("b", (500, 900))))
    with caplog.at_level(logging.WARNING):
        steps = checkpoint_steps(df, "common")
    assert steps == {"a": 200}
    assert any("common budget" in r.getMessage() for r in caplog.records)


def test_a_selection_before_every_pass_says_so(caplog):
    import logging

    df = annotate_frame(pd.DataFrame(_run("a", (100, 200), selected=50)))
    with caplog.at_level(logging.WARNING):
        assert checkpoint_steps(df, "selected") == {"a": 200}
    assert any("precedes every pass" in r.getMessage() for r in caplog.records)
