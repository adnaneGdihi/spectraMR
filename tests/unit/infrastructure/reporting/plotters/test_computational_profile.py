"""Render tests for the computational profile (fig_1_15).

The 2026-07-01 upgrade added the run-facts panel sourced from the
``split="run"`` rows — before that, the figure could never fire on a real
run (per-step stage timings are not logged in production).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from spectramr.infrastructure.reporting.plotters import computational_profile


def _run_facts_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"run_id": "r0", "method": "ours", "metric": "params_m",
         "value": 12.96, "split": "run", "step": -1},
        {"run_id": "r0", "method": "ours", "metric": "iterations_per_sec",
         "value": 5.77, "split": "run", "step": -1},
        {"run_id": "r0", "method": "ours", "metric": "duration_min",
         "value": 42.0, "split": "run", "step": -1},
    ])


def _timing_frame() -> pd.DataFrame:
    rows = []
    for step in range(5):
        for metric, val in (("forward_s", 0.1), ("backward_s", 0.2),
                            ("gpu_mem_gb", 11.0)):
            rows.append({"run_id": "r0", "method": "ours", "metric": metric,
                         "value": val, "split": "train", "step": step})
    return pd.DataFrame(rows)


def test_profile_fires_on_run_facts_alone(tmp_path):
    """Regression: real runs log no stage timings — run_summary facts must
    be enough to produce the figure."""
    out = tmp_path / "profile.pdf"
    res = computational_profile.make(_run_facts_frame(), out,
                                     formats=("pdf",), dpi=150)
    assert res is not None and res.exists() and res.stat().st_size > 0


def test_run_facts_are_panels_without_value_text(tmp_path, monkeypatch):
    figs = []
    real = computational_profile.save_figure
    monkeypatch.setattr(computational_profile, "save_figure",
                        lambda fig, *a, **k: (figs.append(fig), real(fig, *a, **k))[1])
    computational_profile.make(_run_facts_frame(), tmp_path / "p.pdf", formats=("pdf",), dpi=72)
    fig = figs[0]
    assert len(fig.axes) == 3  # one panel per fact in the fixture
    assert all(len(t.get_text()) == 1 for ax in fig.axes for t in ax.texts)  # panel letters only


def test_profile_renders_all_three_panels(tmp_path):
    df = pd.concat([_timing_frame(), _run_facts_frame()], ignore_index=True)
    out = tmp_path / "profile.pdf"
    res = computational_profile.make(df, out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_profile_total_on_empty_frame(tmp_path):
    assert computational_profile.make(pd.DataFrame(), tmp_path / "p.pdf") is None
