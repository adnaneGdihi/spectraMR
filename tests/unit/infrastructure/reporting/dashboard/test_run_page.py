"""Tests for the QC HTML report assembler.

Targets ``spectramr.infrastructure.reporting.dashboard.run_page.build_html_report``:
linked (not embedded) figures, static group fallback (no plotly), table links,
empty-input skip, and (when installed) the interactive plotly path.
"""

from __future__ import annotations

import importlib.util

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from spectramr.infrastructure.reporting.dashboard.run_page import build_html_report

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def _png(path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(path, dpi=50)
    plt.close(fig)
    return path


def _per_case(n=10):
    return pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(n)],
            "split": ["val"] * n,
            "psnr": [30.0 + i for i in range(n)],
            "ssim": [0.8] * n,
        }
    )


def _cases_2d(n=3):
    rng = np.random.default_rng(0)
    out = []
    for i, rank in enumerate(["best", "median", "worst"][:n]):
        tgt = rng.random((32, 32)).astype(np.float32)
        out.append(
            {
                "prediction": tgt + 0.05,
                "target": tgt,
                "rank": rank,
                "case_id": f"v{i}",
                "metrics": {"psnr": 33.0, "ssim": 0.9},
            }
        )
    return out


def _cases_with_volume():
    rng = np.random.default_rng(1)
    vol = rng.random((8, 24, 24)).astype(np.float32)
    return [
        {
            "prediction": vol[4],
            "target": vol[4],
            "rank": "best",
            "case_id": "vol",
            "metrics": {"psnr": 33.0},
            "prediction_volume": vol + 0.03,
            "target_volume": vol,
        }
    ]


def test_returns_none_when_no_figures(tmp_path):
    assert build_html_report(tmp_path, figures={}) is None
    assert build_html_report(tmp_path, figures={"x": None}) is None


def test_links_figures_instead_of_embedding_them(tmp_path):
    figs = {"qc_subject_mosaic": _png(tmp_path / "qc_subject_mosaic.png")}
    out = build_html_report(tmp_path, figures=figs, method="run1", task="reconstruction")
    assert out is not None and out.exists()
    doc = out.read_text()
    assert "src='qc_subject_mosaic.png'" in doc and "base64," not in doc
    # With no interactive cases, the mosaic PNG surfaces under the 2-D viewer's
    # static fallback heading.
    assert "Per-subject viewer" in doc
    assert "run1" in doc


def test_static_group_fallback_without_plotly(tmp_path):
    figs = {"qc_group_strip": _png(tmp_path / "qc_group_strip.png")}
    # pass per_call_df; without plotly it must still embed the static PNG section
    out = build_html_report(tmp_path, figures=figs, per_call_df=_per_case())
    doc = out.read_text()
    assert "Group IQM distribution" in doc
    if not _HAS_PLOTLY:
        assert "src='qc_group_strip.png'" in doc


def test_tables_rendered_as_links(tmp_path):
    tbl = tmp_path / "tables" / "main.csv"
    tbl.parent.mkdir()
    tbl.write_text("a,b\n1,2\n")
    figs = {"qc_carpet": _png(tmp_path / "qc_carpet.png")}
    out = build_html_report(tmp_path, figures=figs, tables={"tab_2_1_main_results": {"csv": tbl}})
    doc = out.read_text()
    assert "tables/main.csv" in doc
    assert "tab_2_1_main_results" in doc


@pytest.mark.skipif(not _HAS_PLOTLY, reason="plotly not installed")
def test_interactive_group_div_when_available(tmp_path):
    out = build_html_report(
        tmp_path,
        figures={"qc_group_strip": _png(tmp_path / "qc_group_strip.png")},
        per_call_df=_per_case(),
    )
    doc = out.read_text()
    assert "iqm-group" in doc  # the interactive group-IQM div


@pytest.mark.skipif(not _HAS_PLOTLY, reason="plotly not installed")
def test_interactive_viewers_and_offline_selfcontained(tmp_path):
    figs = {"qc_group_strip": _png(tmp_path / "qc_group_strip.png")}
    out = build_html_report(
        tmp_path, figures=figs, per_call_df=_per_case(), cases=_cases_with_volume(), method="run"
    )
    doc = out.read_text()
    # both viewers present
    assert "subject-2d" in doc and "volume-3d" in doc
    assert "Per-subject viewer" in doc and "Volume viewer" in doc
    # explainability captions
    assert "How to read this" in doc
    # offline: plotly.js written once beside the page, never fetched from a CDN.
    assert "Plotly.newPlot" in doc
    assert (
        "src='_assets/plotly.min.js'" in doc and (tmp_path / "_assets" / "plotly.min.js").exists()
    )
    assert '<script src="https://cdn' not in doc
    assert "cdn.plot.ly/plotly" not in doc


@pytest.mark.skipif(not _HAS_PLOTLY, reason="plotly not installed")
def test_volume_viewer_absent_without_volume(tmp_path):
    figs = {"qc_group_strip": _png(tmp_path / "qc_group_strip.png")}
    out = build_html_report(tmp_path, figures=figs, per_call_df=_per_case(), cases=_cases_2d())
    doc = out.read_text()
    assert "subject-2d" in doc  # 2-D viewer present
    assert "Volume viewer" not in doc  # 3-D viewer soft-skips (no volume)


def test_interactive_false_uses_static(tmp_path):
    figs = {
        "qc_group_strip": _png(tmp_path / "qc_group_strip.png"),
        "qc_subject_mosaic": _png(tmp_path / "qc_subject_mosaic.png"),
    }
    out = build_html_report(
        tmp_path, figures=figs, per_call_df=_per_case(), cases=_cases_2d(), interactive=False
    )
    doc = out.read_text()
    assert "Plotly.newPlot" not in doc  # no interactive layer
    assert "src='qc_group_strip.png'" in doc  # static PNGs still shown


def test_degrade_to_static_when_plotly_import_fails(tmp_path, monkeypatch):
    # Simulate plotly absent: every interactive builder returns None, so the
    # report must still render with the static group-strip PNG (no crash).
    from spectramr.infrastructure.reporting import interactive as itv

    for name in (
        "group_iqm_div",
        "subject_viewer_2d_div",
        "volume_viewer_3d_div",
        "learning_curve_div",
        "metric_distribution_div",
        "comparison_div",
    ):
        monkeypatch.setattr(itv, name, lambda *a, **k: None)
    figs = {"qc_group_strip": _png(tmp_path / "qc_group_strip.png")}
    out = build_html_report(tmp_path, figures=figs, per_call_df=_per_case(), cases=_cases_2d())
    assert out is not None and out.exists()
    doc = out.read_text()
    assert "Group IQM distribution" in doc
    assert "src='qc_group_strip.png'" in doc
    assert "Plotly.newPlot" not in doc
    assert not (tmp_path / "_assets" / "plotly.min.js").exists()  # written only when used


@pytest.mark.skipif(not _HAS_PLOTLY, reason="plotly not installed")
def test_no_install_hint_when_plotly_is_present_but_data_is_not(tmp_path):
    # No aggregated_df, so the learning-curve div is None and the static PNG stands in;
    # telling the reader to install plotly would be wrong.
    figs = {"fig_1_2_learning_curves": _png(tmp_path / "fig_1_2_learning_curves.png")}
    doc = build_html_report(tmp_path, figures=figs).read_text()
    assert "Training dynamics" in doc and "pip install spectramr[viz]" not in doc
