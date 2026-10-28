"""``hotspots.json``: measured flags, the static join, class roll-up, absent columns."""

from __future__ import annotations

import json

import pytest

from spectramr.cli.profile_hotspots import SCHEMA, build_hotspot_report, write_hotspots

SRC = """\
class Net:
    def slow(self, xs):
        for x in xs:
            x.item()

    def cold(self, xs):
        for x in xs:
            x.item()


def quiet():
    return 1
"""


def _fn(name, lineno, *, python=0.0, c=0.0, sys=0.0, gpu=0.0, peak=0.0, copy=0.0):
    return {
        "line": name,
        "lineno": lineno,
        "n_cpu_percent_python": python,
        "n_cpu_percent_c": c,
        "n_sys_percent": sys,
        "n_gpu_percent": gpu,
        "n_peak_mb": peak,
        "n_copy_mb_s": copy,
    }


@pytest.fixture
def profile(tmp_path):
    src = tmp_path / "src" / "spectramr" / "models" / "net.py"
    src.parent.mkdir(parents=True)
    src.write_text(SRC)
    f = str(src)
    return {
        "gpu": True,
        "memory": True,
        "elapsed_time_sec": 3.0,
        "files": {
            f: {
                "functions": [
                    _fn("Net.slow", 2, python=9.0, sys=3.0, gpu=6.0, peak=300.0, copy=150.0),
                    _fn("quiet", 11, python=0.1),
                ]
            }
        },
        "stacks": [
            [
                [
                    "/x/spectramr/pipelines/training_loop.py _execute_training_loop:10;",
                    f"{f} Net.slow:4;",
                ],
                {"cpu_samples": 0.75},
            ],
            [
                [
                    "/x/spectramr/pipelines/train.py _run_validation:5;",
                    f"{f} Net.slow:4;",
                ],
                {"cpu_samples": 0.25},
            ],
        ],
    }


def test_every_measured_flag_fires_on_a_planted_function(profile):
    report = build_hotspot_report(profile)
    row = next(r for r in report["functions"] if r["function"] == "Net.slow")
    assert report["schema"] == SCHEMA
    assert set(row["measured_flags"]) == {
        "cpu_hot",
        "python_bound",
        "system_heavy",
        "gpu_hot",
        "memory_peak",
        "copy_heavy",
    }
    assert row["file"] == "models/net.py"
    assert row["class"] == "Net"


def test_a_quiet_function_is_not_listed(profile):
    names = {r["function"] for r in build_hotspot_report(profile)["functions"]}
    assert "quiet" not in names


def test_static_findings_join_onto_the_measured_row(profile):
    row = build_hotspot_report(profile)["functions"][0]
    assert [f["rule"] for f in row["static_findings"]] == ["host_sync_in_loop"]


def test_unsampled_function_with_a_finding_is_reported_as_static_only(profile):
    cold = build_hotspot_report(profile)["static_only"]
    assert [c["function"] for c in cold] == ["Net.cold"]


def test_phase_cpu_share_splits_by_stack_membership(profile):
    row = build_hotspot_report(profile)["functions"][0]
    assert row["phase_cpu_share"]["train"] == pytest.approx(0.75)
    assert row["phase_cpu_share"]["validation"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("gpu", "memory", "null_cols", "absent_flags"),
    [
        (False, True, ["gpu_percent"], {"gpu_hot"}),
        (True, False, ["peak_mb", "copy_mb_s"], {"memory_peak", "copy_heavy"}),
    ],
)
def test_a_column_the_mode_did_not_collect_is_null_and_cannot_flag(
    profile, gpu, memory, null_cols, absent_flags
):
    profile["gpu"], profile["memory"] = gpu, memory
    row = build_hotspot_report(profile)["functions"][0]
    assert all(row[c] is None for c in null_cols)
    assert not absent_flags & set(row["measured_flags"])


def test_class_rollup_sums_its_listed_methods(profile):
    (cls,) = build_hotspot_report(profile)["classes"]
    assert cls["class"] == "Net"
    assert cls["cpu_percent"] == pytest.approx(12.0)
    assert cls["methods"] == ["Net.slow"]


def test_unreadable_source_is_recorded_not_fatal(profile):
    profile["files"]["/nonexistent/spectramr/x.py"] = {"functions": [_fn("g", 1, python=9.0)]}
    report = build_hotspot_report(profile)
    assert "x.py" in report["static_scan_skipped"]
    assert any(r["function"] == "g" for r in report["functions"])


def test_write_hotspots_round_trips_and_returns_a_manifest_record(profile, tmp_path):
    src, out = tmp_path / "p.json", tmp_path / "hotspots.json"
    src.write_text(json.dumps(profile))
    record = write_hotspots(src, out)
    assert record["status"] == "written"
    assert (
        json.loads(out.read_text())["summary"]["functions_flagged"] == record["functions_flagged"]
    )


def test_malformed_profile_raises_value_error(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        write_hotspots(bad, tmp_path / "o.json")
