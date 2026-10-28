"""``scripts/n2n_paper_common.py``: which row of which arm the paper reads.

The three checkpoint views are the whole contract: the common budget is the largest
iteration every arm validated at, the final view is each arm's last row, and the
selected view is the optimum of the selector each run recorded for its early
stopping (``val_psnr`` on the v1 download, the corrected MSE under v2). A wrong
row here silently changes every table and mosaic, so each rule is pinned on a
fixture where the three views disagree.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.scripts.n2n_paper_fixture import build_runs

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "n2n_paper_common.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("n2n_paper_common", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common() -> Any:
    return _module()


@pytest.fixture
def arms(common: Any, tmp_path: Path) -> list[Any]:
    return common.discover_arms(build_runs(tmp_path))


def test_arms_are_discovered_a_to_g_then_by_theta(arms: list[Any]) -> None:
    assert [a.key for a in arms] == ["A", "B", "G20", "G"]
    assert [a.letter for a in arms] == ["A", "B", "G", "G"]
    assert arms[2].label.startswith("G20: NuR2R") and "theta=20 deg" in arms[2].label


def test_tau_is_tan_theta_on_r2r_arms_only(arms: list[Any]) -> None:
    by_key = {a.key: a for a in arms}
    assert by_key["A"].tau == pytest.approx(1.0)
    assert by_key["B"].tau is None
    assert by_key["G20"].tau == pytest.approx(math.tan(math.radians(20.0)))


def test_common_iteration_is_the_largest_shared_one(common: Any, arms: list[Any]) -> None:
    assert common.common_iteration(arms) == 2000
    assert {a.key: a.final_iteration for a in arms} == {
        "A": 3000,
        "B": 2000,
        "G20": 2000,
        "G": 2000,
    }


def test_v1_runs_select_on_the_val_psnr_they_recorded(arms: list[Any]) -> None:
    by_key = {a.key: a for a in arms}
    assert {(a.selector, a.selector_mode) for a in arms} == {("val_psnr", "max")}
    assert by_key["A"].selected_iteration == 2000
    assert by_key["B"].selected_iteration == 1000
    assert by_key["G"].selected_iteration == 1000


def test_v2_runs_select_on_the_corrected_mse_minimum(common: Any, tmp_path: Path) -> None:
    by_key = {a.key: a for a in common.discover_arms(build_runs(tmp_path, protocol="v2"))}
    assert by_key["A"].selector == "val_reference_noise_corrected_mse"
    assert by_key["A"].selector_mode == "min"
    assert by_key["A"].selected_iteration == 2000
    assert by_key["B"].selected_iteration == 1000
    # G's PSNR peaks at 1000 and its corrected MSE bottoms out at 2000: read as
    # "higher is better", or on val_psnr, the selection would be 1000.
    g = by_key["G"]
    assert max(g.rows, key=lambda r: r["val_psnr"])["iteration"] == 1000
    assert g.selected_iteration == 2000


def test_a_run_without_a_recorded_selector_raises(common: Any, tmp_path: Path) -> None:
    """Planted: no early_stopping block anywhere, so no row can be called selected."""
    with pytest.raises(ValueError, match=r"no early_stopping\.metric"):
        common.discover_arms(build_runs(tmp_path, record_selector=False))


def test_selected_iteration_refuses_an_arm_without_a_finite_selector(
    common: Any, arms: list[Any]
) -> None:
    a = {x.key: x for x in arms}["A"]
    blind = common.Arm(
        **{**a.__dict__, "rows": tuple({**r, a.selector: float("nan")} for r in a.rows)}
    )
    with pytest.raises(ValueError, match="no finite val_psnr"):
        _ = blind.selected_iteration


def test_slice_pass_reads_the_package_reader_at_one_iteration(arms: list[Any]) -> None:
    from tests.unit.scripts.n2n_paper_fixture import SLICES_PER_SUBJECT, SUBJECTS, add_slice_rows

    a = {x.key: x for x in arms}["A"]
    assert a.slice_pass(2000) is None
    add_slice_rows(a.run_dir.parent)
    rows = a.slice_pass(2000)
    assert rows.iteration == 2000 and len(rows.frame) == SUBJECTS * SLICES_PER_SUBJECT
    with pytest.raises(ValueError, match="no per-slice rows at iteration 2500"):
        a.slice_pass(2500)


def test_view_rows_follow_the_three_rules(common: Any, arms: list[Any]) -> None:
    a = {x.key: x for x in arms}["A"]
    shared = common.common_iteration(arms)
    assert a.view_row("common", shared)["iteration"] == 2000
    assert a.view_row("final", shared)["iteration"] == 3000
    assert a.view_row("selected", shared)["iteration"] == 2000
    with pytest.raises(ValueError, match="unknown view"):
        a.view_iteration("best", shared)


def test_plateau_and_series_read_the_last_rows(arms: list[Any]) -> None:
    a = {x.key: x for x in arms}["A"]
    mean, sd = a.plateau("val_psnr", 2)
    assert mean == pytest.approx(22.45)
    assert sd == pytest.approx(0.05)
    assert a.series("val_nex_equivalent") == []
    assert a.plateau("val_nex_equivalent", 2) is None


def test_provenance_and_declared_metrics_come_from_the_run(arms: list[Any]) -> None:
    g20 = {x.key: x for x in arms}["G20"]
    assert g20.provenance["git_sha"] == "abc123def456"
    assert g20.provenance["git_dirty"] is True
    assert g20.provenance["gpu"] == "Tesla V100 x1"
    assert "nex_equivalent" in g20.declared_metrics
    assert g20.iterations_run == 2600


def test_non_n2n_dirs_are_skipped_and_an_empty_root_raises(common: Any, tmp_path: Path) -> None:
    root = build_runs(tmp_path)
    (root / "ei_unet_m4raw_r4" / "logs").mkdir(parents=True)
    assert len(common.discover_arms(root)) == 4
    with pytest.raises(ValueError, match="not an n2n arm"):
        common.load_arm(root / "ei_unet_m4raw_r4")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        common.discover_arms(empty)


def test_write_table_emits_csv_md_and_booktabs_tex(common: Any, tmp_path: Path) -> None:
    stem = tmp_path / "tab"
    common.write_table(stem, ["Arm", "x_y"], [["A", "1.0"]], tex_header=["Arm", "$x_y$"])
    assert stem.with_suffix(".csv").read_text().splitlines() == ["Arm,x_y", "A,1.0"]
    assert "| A | 1.0 |" in stem.with_suffix(".md").read_text()
    tex = stem.with_suffix(".tex").read_text()
    assert "\\toprule" in tex and "Arm & $x_y$ \\\\" in tex and "A & 1.0 \\\\" in tex
    assert common.fmt(None, "{:.2f}") == "--" and common.fmt(1.234, "{:.2f}") == "1.23"
