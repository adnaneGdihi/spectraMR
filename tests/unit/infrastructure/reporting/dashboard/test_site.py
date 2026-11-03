"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.site` (and the pages it drives)."""

from __future__ import annotations

import json

import pandas as pd

from spectramr.infrastructure.reporting.dashboard.pages import comparison_page
from spectramr.infrastructure.reporting.dashboard.runs import run_records
from spectramr.infrastructure.reporting.dashboard.site import build_site, short_labels


def _long(arms):
    return pd.DataFrame(
        [
            {
                "arm": a,
                "phase": "test",
                "metric": "psnr",
                "mean": 30 + i,
                "std": 1.0,
                "n": 9,
                "unit": "cases",
                "step": None,
                "direction": "higher",
            }
            for i, a in enumerate(arms)
        ]
    )


def _campaign(root):
    """A downloaded campaign: state, two runs (one reported), a comparison and one contrast."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign_state.json").write_text(
        json.dumps(
            {
                "campaign_name": "camp<1>",
                "description": "Two arms.",
                "experiments": [
                    {
                        "name": "x_base",
                        "role": "baseline",
                        "status": "completed",
                        "wall_time_seconds": 120,
                    },
                    {"name": "x_new", "role": "variant", "status": "failed"},
                ],
            }
        )
    )
    (root / "x_base_m4raw" / "report").mkdir(parents=True)
    (root / "x_base_m4raw" / "report" / "qc_report.html").write_text("<html></html>")
    (root / "x_new_m4raw").mkdir()
    cmp_ = root / "comparison"
    (cmp_ / "data").mkdir(parents=True)
    (cmp_ / "figures").mkdir()
    from PIL import Image

    Image.new("RGB", (64, 32)).save(cmp_ / "figures" / "cmp_metric_bars.png")
    _long(["base", "new"]).to_csv(cmp_ / "data" / "metrics_long.csv", index=False)
    con = root / "contrasts" / "x_new__vs__x_base"
    (con / "data").mkdir(parents=True)
    _long(["x_base", "x_new"]).to_csv(con / "data" / "metrics_long.csv", index=False)
    pd.DataFrame(
        [
            {
                "contrast": con.name,
                "phase": "test",
                "metric": "psnr",
                "variant": "x_new",
                "baseline": "x_base",
                "scale": "metric",
                "mean_diff": 1.0,
                "ci_low": 0.5,
                "ci_high": 1.5,
                "cohens_d": 2.0,
                "p_holm": 0.001,
                "n_subjects": 9,
            }
        ]
    ).to_csv(con / "data" / "paired_stats.csv", index=False)
    (con / "contrast.json").write_text(
        json.dumps(
            {
                "contrast": {
                    "name": con.name,
                    "baseline": "x_base",
                    "arms": [{"label": "x_base"}, {"label": "x_new"}],
                }
            }
        )
    )
    return root


def test_short_labels_strip_the_shared_prefix_and_suffix():
    assert short_labels(["n2n_a_x_m4raw", "n2n_b_y_m4raw"]) == {
        "n2n_a_x_m4raw": "a_x",
        "n2n_b_y_m4raw": "b_y",
    }
    assert short_labels(["only"]) == {"only": "only"}


def test_build_site_writes_every_page_and_links_them(tmp_path):
    root = _campaign(tmp_path / "camp")
    pages = build_site(root)
    assert set(pages) == {"index", "comparison", "contrast:x_new__vs__x_base"}
    index = pages["index"].read_text()
    assert "camp&lt;1&gt;" in index  # escaped campaign name
    assert "href='x_base_m4raw/report/qc_report.html'" in index
    assert "href='comparison/comparison.html'" in index
    assert "href='contrasts/x_new__vs__x_base/comparison.html'" in index
    assert "new vs base" in index  # contrast card uses the short labels
    assert (root / "_assets" / "plotly.min.js").exists() or "plotly" not in index


def test_contrast_page_reads_its_paired_stats(tmp_path):
    root = _campaign(tmp_path / "camp")
    con = root / "contrasts" / "x_new__vs__x_base"
    doc = comparison_page(con).read_text()
    assert "x_new vs x_base" in doc and "Paired statistics" in doc
    assert "1 of 1 metrics differ after Holm correction" in doc


def test_run_records_prefer_state_then_batch_rows(tmp_path):
    root = _campaign(tmp_path / "camp")
    state = json.loads((root / "campaign_state.json").read_text())
    recs = run_records(root, state, None)
    assert [r["status"] for r in recs] == ["completed", "failed"]
    assert recs[0]["report"] is not None and recs[1]["report"] is None
    rows = [{"run": root / "x_new_m4raw", "ok": False, "error": "boom"}]
    assert run_records(root, {}, rows)[0] | {"report": None} == {
        "name": "x_new_m4raw",
        "role": None,
        "status": "failed",
        "wall": None,
        "job": None,
        "report": None,
        "error": "boom",
    }


def test_empty_folder_gets_a_directive_not_a_blank_page(tmp_path):
    doc = build_site(tmp_path)["index"].read_text()
    assert "spectramr report --compare" in doc


def test_extra_pages_and_a_catalog_are_linked_from_the_index(tmp_path):
    from PIL import Image

    (tmp_path / "catalog" / "figures").mkdir(parents=True)
    Image.new("RGB", (64, 32)).save(tmp_path / "catalog" / "figures" / "cmp_mosaic.png")
    (tmp_path / "catalog" / "catalog.html").write_text("<html></html>")
    extra = tmp_path / "s2r" / "gallery.html"
    pages = build_site(tmp_path, extra_pages=[(extra, "Sim2Rank styles", "four styles")])
    doc = pages["index"].read_text()
    assert "href='catalog/catalog.html'" in doc and "href='s2r/gallery.html'" in doc
    assert "Reconstructions side by side" in pages["catalog"].read_text()
