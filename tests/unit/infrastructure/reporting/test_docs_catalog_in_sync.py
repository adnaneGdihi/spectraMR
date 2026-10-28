from pathlib import Path

from spectramr.infrastructure.reporting import plotters


_NEW_FIGURES = (
    "mri_recon_panel", "kspace_error_spectrum", "metric_distribution",
    "bland_altman", "significance_matrix", "calibration_coverage",
    "forest_plot", "ablation_heatmap", "acceleration_sweep", "contact_sheet",
    # 2026-07-01 reporting upgrade
    "fig_1_16_run_summary_card", "fig_1_17_metric_correlation",
    "fig_1_18_train_val_gap",
    # QC QC report
    "qc_group_strip", "qc_subject_mosaic", "qc_carpet",
    # 2026-07-03 plotting-SSOT hardening — orphans registered via adapters.
    "certificate_summary", "ksd_certificate", "learnable_acquisition_pareto",
)


def test_new_figures_documented():
    doc = Path("docs/reporting.rst").read_text()
    for fid in _NEW_FIGURES:
        assert fid in doc, f"{fid} missing from docs/reporting.rst"


def test_documented_figures_are_registered():
    """Every figure named in the docs catalog must be a real registered plotter
    (catches a docs/registry drift where the catalog advertises a dead id)."""
    available = set(plotters.list_available())
    for fid in _NEW_FIGURES:
        assert fid in available, f"{fid} documented but not registered"


def test_every_registered_figure_documented():
    """Registry -> docs completeness: EVERY registered plotter id must appear in
    docs/reporting.rst. Since ``spectramr report`` is the plotting SSOT and
    renders all registered figures, the docs must be their catalog — a newly
    registered figure cannot ship undocumented (the drift direction the older
    ``_NEW_FIGURES`` allowlist does not cover)."""
    doc = Path("docs/reporting.rst").read_text()
    undocumented = sorted(fid for fid in plotters.list_available() if fid not in doc)
    assert not undocumented, (
        "registered but undocumented in docs/reporting.rst: "
        f"{undocumented} — document each so the reporting docs stay the catalog SSOT"
    )
