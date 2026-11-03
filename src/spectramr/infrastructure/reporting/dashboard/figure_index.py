"""Figure ids -> gallery family and human title, for every registered plotter.

One owner for how a figure is named and grouped on a page; comparison, contrast, run and
catalog pages all read it. A suffix ``_2``, ``_3`` (a second panel set of the same figure)
keeps the base title and is numbered.
"""

from __future__ import annotations

import re

# Family -> id prefixes, in display order. The first match wins, so a specific id
# (``fig_4_3``) sits in an earlier family than its prefix group (``fig_4_``).
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Images",
        (
            "cmp_mosaic",
            "fig_4_3",
            "mri_recon_panel",
            "mri_a1_",
            "mri_a2_",
            "contact_sheet",
            "qc_subject_mosaic",
            "fig_1_12_",
            "fig_3_9_",
        ),
    ),
    (
        "Headline metrics",
        (
            "fig_1_1_",
            "fig_1_16_",
            "fig_1_11_",
            "cmp_metric_bars",
            "cmp_group_bars",
            "cmp_metric_group_bars",
            "cmp_metric_ladder",
            "cmp_rank",
            "cmp_coverage",
            "acceleration_sweep",
        ),
    ),
    (
        "Effects against the baseline",
        (
            "cmp_effect",
            "cmp_subject_lines",
            "cmp_paired",
            "significance_matrix",
            "bland_altman",
        ),
    ),
    ("Ablations", ("ablation_heatmap", "cohort_task_ablation_bars")),
    ("Training dynamics", ("fig_5_",)),
    ("Where arms differ", ("fig_6_",)),
    ("Metrics by phase", ("cmp_phase",)),
    (
        "Per-case distributions",
        ("fig_3_", "metric_distribution", "qc_group_strip", "fig_1_17_", "fig_1_5_", "fig_1_4_"),
    ),
    ("Training", ("fig_1_2_", "fig_1_3_", "fig_1_18_", "fig_1_15_", "gen_")),
    (
        "Quality control",
        ("qc_", "kspace_error_spectrum", "mri_a11_", "mri_a12_", "calibration_coverage"),
    ),
    (
        "Physics and certificates",
        ("fig_b", "fig_c", "ksd_certificate", "certificate_summary", "fig_2_15_"),
    ),
    ("Model internals", ("fig_4_",)),
)
OTHER = "Other figures"

TITLES = {
    "ablation_heatmap": "Ablation effects per metric",
    "acceleration_sweep": "Metric against acceleration",
    "bland_altman": "Bland-Altman agreement",
    "calibration_coverage": "Uncertainty calibration coverage",
    "certificate_summary": "Certificate summary",
    "cmp_coverage": "Which arm logged which metric",
    "cmp_effect_bars": "Differences against the baseline",
    "cmp_effect_forest": "Effect sizes against the baseline",
    "cmp_group_bars": "Metric groups per arm",
    "cmp_metric_bars": "Metric means per arm",
    "cmp_metric_group_bars": "Metric families per arm",
    "cmp_metric_ladder": "Metric ladder across arms",
    "cmp_mosaic": "Reconstructions side by side",
    "cmp_paired_slice_scatter": "Paired per-slice scatter",
    "cmp_phase_bars": "Metric means by phase",
    "cmp_phase_metrics": "Phase-by-metric panels",
    "cmp_rank_heatmap": "Rank of each arm per metric",
    "cmp_rank_reversal": "Rank reversals between metrics",
    "cmp_subject_lines": "Per-subject paired lines",
    "cohort_task_ablation_bars": "Cohort ablation by task",
    "contact_sheet": "Contact sheet of cases",
    "fig_1_1_headline_pareto": "Headline Pareto front",
    "fig_1_2_learning_curves": "Learning curves",
    "fig_1_3_loss_decomposition": "Loss decomposition",
    "fig_1_4_residual_diagnostics": "Residual diagnostics",
    "fig_1_5_predicted_vs_true": "Predicted against true",
    "fig_1_11_stratified_performance": "Performance by stratum",
    "fig_1_12_failure_gallery": "Failure gallery",
    "fig_1_15_computational_profile": "Computational profile",
    "fig_1_16_run_summary_card": "Run summary card",
    "fig_1_17_metric_correlation": "Metric correlation",
    "fig_1_18_train_val_gap": "Train-validation gap",
    "fig_2_15_active_acquisition_trajectory": "Active acquisition trajectory",
    "fig_3_2_subject_bars": "Per-subject means",
    "fig_3_3_metric_scatter": "Metric against metric",
    "fig_3_6_error_decomposition": "Bias and variance decomposition",
    "fig_3_7_metric_embedding": "Metric embedding (t-SNE)",
    "fig_3_8_intensity_joint": "Joint intensity histogram",
    "fig_3_9_background_zoom": "Background zoom",
    "fig_4_1_weight_trajectory": "Weight trajectory (PCA)",
    "fig_4_2_layer_drift": "Layer drift from initialisation",
    "fig_4_3_progress_strip": "Progress strip over training",
    "fig_4_4_image_embedding": "Image embedding",
    "fig_5_1_spectral_recovery": "Spectral recovery: error per k-space band over training",
    "fig_5_2_gain_over_input": "Gain over the input over training",
    "fig_5_3_training_health": "Training health: learning rate, loss spikes, non-finite steps",
    "fig_6_1_critical_difference": "Critical difference: mean rank over subjects",
    "fig_6_2_win_rate": "Win rate between arms, per subject",
    "fig_6_3_band_fidelity": "Error per k-space band, per slice",
    "fig_6_4_slice_profile": "Metric against slice position",
    "fig_6_5_gain_distribution": "Per-slice gain over the input",
    "fig_b3_bloch_consistency_residual": "Bloch consistency residual",
    "fig_b7_qmap_riemannian_vs_euclidean": "q-map: Riemannian against Euclidean",
    "fig_c1_beltrami_field": "Beltrami field",
    "fig_c2_spd_geodesic": "SPD geodesic",
    "fig_c3_teichmuller_schedule": "Teichmüller schedule",
    "fig_c4_fingerprint_embedding": "Fingerprint embedding",
    "gen_diffusion_diagnostics": "Diffusion diagnostics",
    "gen_gan_diagnostics": "GAN diagnostics",
    "gen_vae_diagnostics": "VAE diagnostics",
    "ksd_certificate": "Kernel Stein discrepancy certificate",
    "kspace_error_spectrum": "k-space error spectrum",
    "metric_distribution": "Metric distribution",
    "mri_a11_cohort_table": "Cohort table",
    "mri_a12_fiducial_check": "Fiducial check",
    "mri_a1_sr_triptych": "Super-resolution triptych",
    "mri_a2_synthesis_c2c": "Contrast-to-contrast synthesis",
    "mri_recon_panel": "Reconstructions with error and radial k-space error",
    "qc_carpet": "Carpet and spike diagnostics",
    "qc_group_strip": "Group IQM strip",
    "qc_subject_mosaic": "Per-subject QC mosaic",
    "significance_matrix": "Pairwise significance matrix",
}


def caption(fig_id: str) -> str:
    """Human title for a figure id; a numbered repeat (``_2``) keeps its base title."""
    base, _, n = fig_id.rpartition("_")
    stem, suffix = (base, f" ({n})") if n.isdigit() and base in TITLES else (fig_id, "")
    fallback = re.sub(r"^(cmp|fig(_[a-z]?\d+)+)_", "", stem).replace("_", " ").capitalize()
    return TITLES.get(stem, fallback) + suffix


def family(fig_id: str) -> str:
    return next((name for name, pre in FAMILIES if fig_id.startswith(pre)), OTHER)


def slug(text: str) -> str:
    return re.sub(r"\W+", "-", text.lower()).strip("-")
