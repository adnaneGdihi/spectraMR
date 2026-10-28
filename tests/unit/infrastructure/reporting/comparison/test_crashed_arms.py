"""A crashed arm never reaches a comparison table unmarked (#2525).

The cohort has three arms: ``ok`` (``success: true``), ``crashed`` (the
summary the training pipeline writes on a crash) and ``nosummary`` (no
``run_summary.json``, like runs downloaded before the flag existed).
"""

import json
import re

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.comparison import arms as arms_mod
from spectramr.infrastructure.reporting.comparison import generate_comparison, load_arms
from spectramr.infrastructure.reporting.run_status import RunStatus

from ._runs import make_run

CRASH = {
    "success": False,
    "error_type": "RuntimeError",
    "error": "Traceback (most recent call last):\nRuntimeError: CUDA error",
    "best_metrics": None,
    "duration_sec": None,
    "model_params": None,
}


def _arm(root, name, summary):
    run = make_run(root, name, per_call=False, pngs=False)
    if summary is not None:
        (run / "run_summary.json").write_text(json.dumps(summary))
    return run


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path / "cohort"
    _arm(root, "ok", {"success": True, "model_params": 2_000_000})
    _arm(root, "crashed", CRASH)
    _arm(root, "nosummary", None)
    return root


def _compare(cohort, tmp_path, **kw):
    return generate_comparison(
        [cohort], tmp_path / "cmp", formats=("png",), dpi=40, html_report=False, **kw
    )


def unmarked_crashed_rows(table_md: str, crashed: set[str]) -> list[str]:
    """Rows of the Markdown table naming a crashed arm without a marker."""
    rows = [ln for ln in table_md.splitlines() if ln.startswith("| ") and "---" not in ln]
    names = [ln.split("|")[1].strip() for ln in rows[1:]]
    return [n for n in names if n in crashed]


def test_planted_the_old_behaviour_is_what_the_check_rejects(cohort, tmp_path, monkeypatch):
    # Every arm read as finished reproduces the tabulation before the fix.
    monkeypatch.setattr(arms_mod, "read_run_status", lambda run: RunStatus("ok"))
    res = _compare(cohort, tmp_path)
    table = res["tables"]["tab_2_1_main_results"]["markdown"].read_text()
    assert unmarked_crashed_rows(table, {"crashed"}) == ["crashed"]


def test_the_crashed_arm_is_not_tabulated(cohort, tmp_path):
    res = _compare(cohort, tmp_path)
    table = res["tables"]["tab_2_1_main_results"]["markdown"].read_text()
    assert unmarked_crashed_rows(table, {"crashed"}) == []
    assert "Not tabulated: crashed (crashed: RuntimeError)." in table
    assert res["arms"] == ["nosummary", "ok"]
    csv = pd.read_csv(res["tables"]["tab_2_1_main_results"]["csv"])
    assert "crashed" not in set(csv["method"])


def test_the_crashed_arm_is_absent_from_the_figure_frame(cohort):
    arms = load_arms(sorted(cohort.iterdir()), with_images=False)
    assert "crashed" not in set(arms.df["method"])
    assert set(arms.excluded) == {"crashed"}
    assert arms.labels == ["nosummary", "ok"]


def test_an_arm_without_a_summary_is_kept_and_marked(cohort, tmp_path):
    res = _compare(cohort, tmp_path)
    md = res["tables"]["tab_2_1_main_results"]["markdown"].read_text()
    tex = res["tables"]["tab_2_1_main_results"]["latex"].read_text()
    assert re.search(r"^\| nosummary ‡ \|", md, re.M)
    assert re.search(r"^\| ok \|", md, re.M)
    assert "‡ nosummary: status unknown: no run_summary.json." in md
    assert r"nosummary $^\ddagger$" in tex
    assert r"Not tabulated: crashed" in tex


def test_the_manifest_and_summary_name_the_excluded_arm(cohort, tmp_path):
    res = _compare(cohort, tmp_path)
    manifest = json.loads((res["out_dir"] / "report_manifest.json").read_text())
    runs = {r["arm"]: r for r in manifest["runs"]}
    assert runs["crashed"]["included"] is False
    assert runs["crashed"]["status"] == "failed"
    assert runs["crashed"]["error_type"] == "RuntimeError"
    assert runs["nosummary"]["included"] is True
    assert runs["nosummary"]["status"] == "unknown"
    assert runs["ok"]["status"] == "ok"
    summary = res["summary"].read_text()
    assert "## Excluded arms" in summary
    assert "**crashed**" in summary and "RuntimeError: CUDA error" in summary
    assert "Traceback" not in summary


def test_a_crashed_baseline_raises(cohort, tmp_path):
    with pytest.raises(ValueError, match="baseline 'crashed' is excluded"):
        _compare(cohort, tmp_path, baseline="crashed")


def test_fewer_than_two_finished_arms_raises(tmp_path):
    root = tmp_path / "cohort"
    _arm(root, "ok", {"success": True})
    _arm(root, "crashed", CRASH)
    with pytest.raises(ValueError, match=r"crashed \(crashed: RuntimeError\)"):
        load_arms(sorted(root.iterdir()), with_images=False)


def test_excluding_an_arm_does_not_rename_the_others(tmp_path):
    root = tmp_path / "cohort"
    _arm(root, "exp_a_run", {"success": True})
    _arm(root, "exp_b_run", {"success": True})
    _arm(root, "other_c_run", CRASH)
    arms = load_arms(sorted(root.iterdir()), with_images=False)
    # shortened over all three runs; over the two survivors they would be a, b
    assert arms.labels == ["exp_a", "exp_b"]
    assert set(arms.excluded) == {"other_c"}
