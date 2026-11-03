"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.runs`."""

from __future__ import annotations

from spectramr.infrastructure.reporting.dashboard.runs import run_records, runs_section


def test_state_rows_match_run_folders_by_name_prefix(tmp_path):
    (tmp_path / "x_base_m4raw" / "report").mkdir(parents=True)
    (tmp_path / "x_base_m4raw" / "report" / "qc_report.html").write_text("<html></html>")
    state = {
        "experiments": [
            {"name": "x_base", "role": "baseline", "status": "completed"},
            {"name": "x_gone", "status": "failed"},
        ]
    }
    recs = run_records(tmp_path, state, None)
    assert recs[0]["report"] == tmp_path / "x_base_m4raw" / "report" / "qc_report.html"
    assert recs[1]["report"] is None


def test_failed_batch_row_shows_its_error_and_no_link(tmp_path):
    (tmp_path / "run_a").mkdir()
    recs = run_records(tmp_path, {}, [{"run": tmp_path / "run_a", "ok": False, "error": "boom<1>"}])
    doc = runs_section(tmp_path / "index.html", recs).body
    assert "title='boom&lt;1&gt;'" in doc and "No report" in doc and "0 of 1" not in doc
    assert runs_section(tmp_path / "index.html", recs).note == "0 of 1 runs completed."


def test_no_runs_no_section(tmp_path):
    assert runs_section(tmp_path / "index.html", []) is None


def test_output_dir_wins_and_a_name_never_claims_a_longer_sibling(tmp_path):
    from spectramr.infrastructure.reporting.dashboard.runs import _run_dir

    for name in ("g_theta10_m4raw", "g_theta1_m4raw", "bound_m4raw_r4"):
        (tmp_path / name).mkdir()
    assert _run_dir(tmp_path, {"name": "g_theta1"}).name == "g_theta1_m4raw"
    assert (
        _run_dir(tmp_path, {"name": "bound_r4", "output_dir": "/c/bound_m4raw_r4"}).name
        == "bound_m4raw_r4"
    )


def test_finished_runs_missing_from_the_state_are_listed(tmp_path):
    for name in ("a_m4raw", "b_m4raw"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "final_metrics.json").write_text("{}")
    recs = run_records(tmp_path, {"experiments": [{"name": "a", "status": "completed"}]}, None)
    assert [(r["name"], r["status"]) for r in recs] == [("a", "completed"), ("b_m4raw", "on disk")]
