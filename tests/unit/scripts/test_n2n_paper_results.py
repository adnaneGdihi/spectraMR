"""``scripts/n2n_paper_results.py``: the tables and figures the paper quotes.

Pins the shape of every table on the synthetic cohort: an arm that never declared a
metric shows ``--`` rather than a number from somewhere else, the identity row
carries the input's own PSNR, contrasts name the iterations when the two sides
were read at different ones, and the theta sweep is ordered by angle with
``tau = tan(theta)`` beside it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.scripts.n2n_paper_fixture import build_runs

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "n2n_paper_results.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("n2n_paper_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results() -> Any:
    return _module()


@pytest.fixture(scope="module")
def built(results: Any, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("runs")
    out = tmp_path_factory.mktemp("out")
    payload = results.build(build_runs(root), out, 2)
    return out, payload


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_table_and_figure_is_written(built: tuple[Path, dict]) -> None:
    out, _ = built
    for view in ("common", "final", "selected"):
        for kind in ("main", "contrasts"):
            for suffix in (".csv", ".md", ".tex"):
                assert (out / "tables" / f"tab_n2n_{kind}_{view}{suffix}").exists()
    for stem in ("tab_n2n_theta_sweep", "tab_n2n_plateau", "tab_n2n_provenance"):
        assert (out / "tables" / f"{stem}.tex").exists()
    for figure in ("fig_n2n_curves", "fig_n2n_repbv", "fig_n2n_theta_sweep"):
        assert (out / "figures" / f"{figure}.pdf").exists()
        assert (out / "figures" / f"{figure}.png").exists()


def test_main_table_reads_each_view_at_its_own_iteration(built: tuple[Path, dict]) -> None:
    out, _ = built
    common = {r["Arm"].split(":")[0]: r for r in _rows(out / "tables" / "tab_n2n_main_common.csv")}
    assert common["A"]["iteration"] == "2000" and common["A"]["PSNR (dB)"] == "22.50"
    assert common["A"]["N_eq"] == "--"
    assert common["G20"]["N_eq"] == "6.50"
    identity = common["identity (the input itself)"]
    assert identity["PSNR (dB)"] == "23.80" and identity["SSIM"] == "--"
    final = {r["Arm"].split(":")[0]: r for r in _rows(out / "tables" / "tab_n2n_main_final.csv")}
    assert final["A"]["iteration"] == "3000" and final["A"]["PSNR (dB)"] == "22.40"
    selected = {
        r["Arm"].split(":")[0]: r for r in _rows(out / "tables" / "tab_n2n_main_selected.csv")
    }
    assert selected["B"]["iteration"] == "1000" and selected["B"]["PSNR (dB)"] == "22.70"


def test_contrasts_are_signed_point_differences(built: tuple[Path, dict]) -> None:
    out, _ = built
    common = {
        r["variant - baseline"]: r for r in _rows(out / "tables" / "tab_n2n_contrasts_common.csv")
    }
    assert common["A - B"]["iteration"] == "2000"
    assert common["A - B"]["d PSNR (dB)"] == "-0.10"
    assert common["A - B"]["d N_eq"] == "--"
    assert common["G20 - G"]["d N_eq"] == "+0.50"
    assert "D - A" not in common
    selected = {
        r["variant - baseline"]: r for r in _rows(out / "tables" / "tab_n2n_contrasts_selected.csv")
    }
    assert selected["A - B"]["iteration"] == "2000 vs 1000"


def test_theta_sweep_is_ordered_by_angle_with_tau(built: tuple[Path, dict]) -> None:
    out, _ = built
    rows = _rows(out / "tables" / "tab_n2n_theta_sweep.csv")
    assert [r["theta (deg)"] for r in rows] == ["20", "45"]
    assert [r["tau = tan theta"] for r in rows] == ["0.36", "1.00"]
    assert all(r["iteration"] == "2000" for r in rows)


def test_plateau_and_provenance_tables(built: tuple[Path, dict]) -> None:
    out, _ = built
    plateau = {r["Arm"]: r for r in _rows(out / "tables" / "tab_n2n_plateau.csv")}
    assert plateau["A"]["validations"] == "last 2 of 3"
    assert plateau["A"]["PSNR (dB) mean +- sd"] == "22.45 +- 0.05"
    assert plateau["B"]["N_eq mean +- sd"] == "--"
    provenance = {r["Arm"]: r for r in _rows(out / "tables" / "tab_n2n_provenance.csv")}
    assert provenance["G20"]["dirty"] == "yes" and provenance["A"]["dirty"] == "no"
    assert provenance["A"]["iterations run"] == "3600"
    assert provenance["B"]["selected iteration"] == "1000"
    assert provenance["B"]["selector"] == "val_psnr (max)"


def test_a_v2_download_is_read_at_its_corrected_mse_selection(
    results: Any, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    results.build(build_runs(tmp_path / "runs", protocol="v2"), out, 2)
    selected = {
        r["Arm"].split(":")[0]: r for r in _rows(out / "tables" / "tab_n2n_main_selected.csv")
    }
    assert selected["G"]["iteration"] == "2000" and selected["B"]["iteration"] == "1000"
    provenance = {r["Arm"]: r for r in _rows(out / "tables" / "tab_n2n_provenance.csv")}
    assert provenance["G"]["selector"] == "val_reference_noise_corrected_mse (min)"


def test_json_holds_every_number(built: tuple[Path, dict]) -> None:
    out, payload = built
    on_disk = json.loads((out / "tables" / "n2n_results.json").read_text())
    assert on_disk["common_iteration"] == payload["common_iteration"] == 2000
    a = on_disk["arms"]["A"]
    assert a["rows"]["final"]["iteration"] == 3000
    assert a["per_metric_best_over_iterations"] == {"val_psnr_best": 22.5}
    assert a["tau"] == pytest.approx(1.0)
    assert on_disk["arms"]["B"]["tau"] is None
    assert on_disk["arms"]["G"]["plateau_last_k"]["val_nex_equivalent"] == [5.5, 0.5]


def test_cli_runs_end_to_end(
    results: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = build_runs(tmp_path / "runs")
    assert (
        results.main(["--runs", str(runs), "--out", str(tmp_path / "out"), "--plateau", "2"]) == 0
    )
    assert "4 arms read at common iteration 2000" in capsys.readouterr().out


def test_paired_table_is_skipped_without_per_slice_rows(
    built: tuple[Path, dict], results: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, _ = built
    assert not list((out / "tables").glob("tab_n2n_paired_*"))
    results.build(build_runs(tmp_path / "runs"), tmp_path / "out", 2)
    assert "wrote no per-slice rows (#2478)" in capsys.readouterr().out


def test_paired_table_pairs_subjects_from_the_per_slice_rows(
    results: Any, tmp_path: Path
) -> None:
    from tests.unit.scripts.n2n_paper_fixture import SLICE_NEQ, SUBJECTS, add_slice_rows

    runs = build_runs(tmp_path / "runs")
    add_slice_rows(runs)
    results.build(runs, tmp_path / "out", 2)
    rows = {
        (r["metric"], r["variant - baseline"]): r
        for r in _rows(tmp_path / "out" / "tables" / "tab_n2n_paired_common.csv")
    }
    g_a = rows[("val_nex_equivalent", "G - A")]
    assert g_a["scale"].startswith("1/N_eq") and g_a["subjects"] == str(SUBJECTS)
    assert g_a["non-finite"] == "0/0"
    # On 1/N_eq the G arm (N_eq about 3) sits below A (about 2): a negative difference.
    assert float(g_a["mean diff"]) < 0
    expected = sum(
        1 / (SLICE_NEQ["G"] + 0.2 * s) - 1 / (SLICE_NEQ["A"] + 0.2 * s) for s in range(SUBJECTS)
    ) / SUBJECTS
    assert float(g_a["mean diff"]) == pytest.approx(expected, abs=0.02)
    assert float(g_a["p (Holm)"]) >= float(g_a["p (t)"])
    assert ("val_reference_noise_corrected_mse", "A - B") in rows


def _paired_rows(results: Any, tmp_path: Path, contrasts: Any = None) -> dict:
    from tests.unit.scripts.n2n_paper_fixture import add_slice_rows

    runs = build_runs(tmp_path / "runs")
    add_slice_rows(runs)
    arms = results.discover_arms(runs)
    common = results.common_iteration(arms)
    kwargs = {} if contrasts is None else {"contrasts": contrasts}
    header, rows, _ = results.paired_table(arms, "common", common, **kwargs)
    return {(r[0], r[1]): dict(zip(header, r, strict=True)) for r in rows}


def test_holm_runs_within_each_family_not_pooled(results: Any, tmp_path: Path) -> None:
    """Planted: one Holm over every contrast gives different adjusted p-values."""
    from spectramr.core.metrics.statistical_tests import StatisticalTests

    rows = _paired_rows(results, tmp_path)
    neq = {k[1]: r for k, r in rows.items() if k[0] == "val_nex_equivalent"}
    assert {k: r["holm_family"] for k, r in neq.items()} == {
        "A - B": "primary",
        "G - A": "primary",
        "G20 - G": "theta_sweep",
    }
    # Alone in its family, the theta contrast keeps its raw p.
    assert neq["G20 - G"]["p (Holm)"] == neq["G20 - G"]["p (t)"]
    order = ["A - B", "G - A", "G20 - G"]
    pooled = StatisticalTests.holm_bonferroni_correction(
        [float(neq[k]["p (t)"]) for k in order]
    )["corrected_p_values"]
    split = [float(neq[k]["p (Holm)"]) for k in order]
    assert split != pytest.approx(pooled, rel=1e-2)


def test_a_contrast_in_no_family_is_reported_uncorrected(results: Any, tmp_path: Path) -> None:
    rows = _paired_rows(results, tmp_path, contrasts=(("A", "B", "primary"), ("G20", "A", None)))
    off = rows[("val_nex_equivalent", "G20 - A")]
    assert off["holm_family"] == "uncorrected" and off["p (Holm)"] == off["p (t)"]


def test_contrast_table_mirrors_the_campaign_reads_against_and_roles(results: Any) -> None:
    """The one table of contrasts and families agrees with the campaign manifest."""
    import re

    import yaml

    manifest = yaml.safe_load(
        (REPO_ROOT / "experiments/campaigns/n2n_synthetic_nex_m4raw.yaml").read_text()
    )
    names = re.compile(
        r"^n2n_(?P<letter>[a-g])(?:_theta(?P<theta>\d+)|_(?P<strength>(?:alpha|lambda)\d+))?"
    )

    def key(name: str) -> str:
        m = names.match(name)
        assert m is not None, name
        if m["strength"]:
            return f"{m['letter'].upper()}_{m['strength']}"
        return m["letter"].upper() + (m["theta"] or "")

    family_of_role = {"variant": "primary", "ablation": "theta_sweep"}
    from_manifest = set()
    for exp in manifest["experiments"]:
        against = (exp.get("tags") or {}).get("reads_against")
        if against and against != "none":
            from_manifest.add((key(exp["name"]), key(against), family_of_role[exp["role"]]))
    assert set(results.CONTRASTS) == from_manifest
