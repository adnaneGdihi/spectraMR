"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.campaign`."""

from __future__ import annotations

import json

from spectramr.infrastructure.reporting.dashboard import campaign


def test_one_finished_run_gets_the_dashboard_but_no_comparison(tmp_path, monkeypatch):
    (tmp_path / "x_base_m4raw").mkdir()
    (tmp_path / "campaign_state.json").write_text(
        json.dumps(
            {
                "campaign_name": "c",
                "experiments": [
                    {"name": "x_base", "role": "baseline", "status": "completed"},
                    {"name": "x_new", "status": "failed"},
                ],
            }
        )
    )
    called = []
    monkeypatch.setattr(
        "spectramr.infrastructure.reporting.comparison.generate_comparison",
        lambda *a, **k: called.append(a),
    )
    pages = campaign.build_campaign_report(tmp_path)
    assert called == [] and pages["index"].exists()


def test_finished_runs_are_compared_against_the_declared_baseline(tmp_path, monkeypatch):
    for name in ("x_base_m4raw", "x_new_m4raw"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "final_metrics.json").write_text("{}")  # a run marker
    (tmp_path / "campaign_state.json").write_text(
        json.dumps(
            {
                "campaign_name": "c",
                "experiments": [
                    {"name": "x_base", "role": "baseline", "status": "completed"},
                    {"name": "x_new", "role": "variant", "status": "completed"},
                ],
            }
        )
    )
    seen, reported = {}, []
    monkeypatch.setattr(
        "spectramr.infrastructure.reporting.comparison.generate_comparison",
        lambda runs, out, **k: seen.update(runs=runs, out=out, **k),
    )
    monkeypatch.setattr(
        "spectramr.infrastructure.reporting.pipeline.generate_report",
        lambda run, **k: reported.append((run.name, k["figures"])),
    )
    campaign.build_campaign_report(tmp_path)
    assert [p.name for p in seen["runs"]] == ["x_base_m4raw", "x_new_m4raw"]
    assert seen["out"] == tmp_path / "comparison" and seen["baseline"] == "base"
    # a campaign attempts every registered figure, per run and in the comparison
    assert seen["figures"] == "all" and seen["metrics"] == "all"
    assert reported == [("x_base_m4raw", ["all"]), ("x_new_m4raw", ["all"])]


def test_a_stale_state_still_reports_every_run_on_disk(tmp_path, monkeypatch):
    # Downloaded states lag: names that are not folder prefixes, statuses never updated, and
    # finished runs the state does not list at all. The runs on disk are what get reported.
    for name in ("bound_m4raw_r4", "ei_m4raw_r4", "extra_m4raw_r4"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "final_metrics.json").write_text("{}")
    (tmp_path / "campaign_state.json").write_text(
        json.dumps(
            {
                "campaign_name": "c",
                "experiments": [
                    {
                        "name": "bound_r4",
                        "role": "baseline",
                        "status": "submitted",
                        "output_dir": "/cluster/x/bound_m4raw_r4",
                    },
                    {
                        "name": "ei_r4",
                        "status": "submitted",
                        "output_dir": "/cluster/x/ei_m4raw_r4",
                    },
                ],
            }
        )
    )
    seen, reported = {}, []
    monkeypatch.setattr(
        "spectramr.infrastructure.reporting.comparison.generate_comparison",
        lambda runs, out, **k: seen.update(runs=runs, **k),
    )
    monkeypatch.setattr(
        "spectramr.infrastructure.reporting.pipeline.generate_report",
        lambda run, **k: reported.append(run.name),
    )
    campaign.build_campaign_report(tmp_path)
    assert reported == ["bound_m4raw_r4", "ei_m4raw_r4", "extra_m4raw_r4"]
    assert seen["baseline"] == "bound" and len(seen["runs"]) == 3
