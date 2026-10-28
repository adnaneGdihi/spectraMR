r"""Every registered figure and table, rendered from built-in examples.

``EXAMPLES`` maps each registered figure id to ``(frame_factory, kwargs_factory)``:
a well-formed synthetic input it must render from. It is the one owner of those
inputs -- the plotter smoke test dispatches every entry (a figure registered but
unable to fire on good input is a facade) and ``render_catalog`` draws them all,
so the catalog shows exactly what the registry can produce.

The examples are synthetic. A catalog figure says what the plotter draws, not
what any run measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .examples_frames import (
    make_accel_frame,
    make_acquisition_frame,
    make_bland_frame,
    make_bloch_frame,
    make_calibration_frame,
    make_heatmap_frame,
    make_metric_frame,
    make_predictions_df,
    make_qmap_frame,
    make_stratified_df,
    make_task_ablation_frame,
)
from .examples_long import make_long_metrics_df
from .examples_payloads import (
    make_acquisition_arms,
    make_arms,
    make_beltrami_mu,
    make_certificate_paths,
    make_cohort_dict,
    make_fingerprint_embedding,
    make_image_cases,
    make_ksd_result,
    make_paired_rows,
    make_paired_subjects,
    make_phase_table,
    make_qmap_slices,
    make_slice_df,
    make_checkpoint_runs,
    make_png_run,
)


def _example_ladder() -> Any:
    from spectramr.infrastructure.reporting.comparison.contrasts import Ladder

    return Ladder(metric="psnr", references={"reference": 30.0})


EXAMPLES: dict[str, tuple[Any, Any]] = {
    "fig_1_1_headline_pareto": (
        make_long_metrics_df,
        lambda: {"metric": "psnr", "cost": "params_m"},
    ),
    "fig_1_2_learning_curves": (make_long_metrics_df, lambda: {"metrics": ["loss", "val_loss"]}),
    "fig_1_3_loss_decomposition": (make_long_metrics_df, dict),
    "fig_1_4_residual_diagnostics": (
        make_long_metrics_df,
        lambda: {"predictions_df": make_predictions_df()},
    ),
    "fig_1_5_predicted_vs_true": (
        make_long_metrics_df,
        lambda: {"predictions_df": make_predictions_df()},
    ),
    "fig_1_9_ablation_strip": (
        make_metric_frame,
        lambda: {"primary_metric": "psnr", "full_method": "full"},
    ),
    "fig_1_11_stratified_performance": (
        make_long_metrics_df,
        lambda: {"stratified_df": make_stratified_df()},
    ),
    "fig_1_12_failure_gallery": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "fig_1_15_computational_profile": (make_long_metrics_df, dict),
    "fig_1_16_run_summary_card": (make_long_metrics_df, dict),
    "fig_1_17_metric_correlation": (make_long_metrics_df, dict),
    "fig_1_18_train_val_gap": (make_long_metrics_df, dict),
    "gen_gan_diagnostics": (make_long_metrics_df, dict),
    "gen_vae_diagnostics": (make_long_metrics_df, dict),
    "gen_diffusion_diagnostics": (make_long_metrics_df, dict),
    "metric_distribution": (make_metric_frame, dict),
    "significance_matrix": (make_metric_frame, lambda: {"metric": "psnr"}),
    "bland_altman": (make_bland_frame, lambda: {"metric": "ssim"}),
    "calibration_coverage": (make_calibration_frame, dict),
    "forest_plot": (make_metric_frame, lambda: {"metric": "psnr"}),
    "ablation_heatmap": (make_heatmap_frame, dict),
    "acceleration_sweep": (make_accel_frame, lambda: {"metric": "psnr"}),
    "mri_a1_sr_triptych": (
        make_long_metrics_df,
        lambda: {"cases": make_image_cases(), "show_spectrum": True},
    ),
    "mri_a2_synthesis_c2c": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "mri_a7_kspace_recon": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "mri_recon_panel": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "kspace_error_spectrum": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "mri_a11_cohort_table": (make_long_metrics_df, lambda: {"cohort": make_cohort_dict()}),
    "mri_a12_fiducial_check": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "fig_2_15_active_acquisition_trajectory": (make_acquisition_frame, dict),
    "fig_b3_bloch_consistency_residual": (make_bloch_frame, dict),
    "fig_b7_qmap_riemannian_vs_euclidean": (
        make_qmap_frame,
        lambda: {"slices": make_qmap_slices()},
    ),
    "fig_c1_beltrami_field": (make_long_metrics_df, lambda: {"mu": make_beltrami_mu()}),
    "fig_c2_spd_geodesic": (
        make_long_metrics_df,
        lambda: {"distances": np.linspace(1.2, 0.05, 200)},
    ),
    "fig_c3_teichmuller_schedule": (
        make_long_metrics_df,
        lambda: {"r": np.linspace(0.0, 0.95, 100)},
    ),
    "fig_c4_fingerprint_embedding": (
        make_long_metrics_df,
        lambda: {
            "embedding": make_fingerprint_embedding()[0],
            "labels": make_fingerprint_embedding()[1],
        },
    ),
    # QC QC report.
    "qc_group_strip": (make_metric_frame, dict),
    "qc_subject_mosaic": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "qc_carpet": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "fig_3_1_metric_violin": (make_long_metrics_df, lambda: {"slice_df": make_slice_df()}),
    "fig_3_2_subject_bars": (make_long_metrics_df, lambda: {"slice_df": make_slice_df()}),
    "fig_3_3_metric_scatter": (make_long_metrics_df, lambda: {"slice_df": make_slice_df()}),
    "cmp_paired_slice_scatter": (
        make_long_metrics_df,
        lambda: {"slice_df": make_slice_df(), "baseline": "arm0"},
    ),
    "cmp_rank_heatmap": (make_long_metrics_df, lambda: {"slice_df": make_slice_df()}),
    "fig_3_6_error_decomposition": (make_long_metrics_df, lambda: {"slice_df": make_slice_df()}),
    "fig_3_7_metric_embedding": (make_long_metrics_df, lambda: {"slice_df": make_slice_df()}),
    "fig_3_8_intensity_joint": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "fig_3_9_background_zoom": (make_long_metrics_df, lambda: {"cases": make_image_cases()}),
    "fig_4_1_weight_trajectory": (
        make_long_metrics_df,
        lambda: {"weight_runs": make_checkpoint_runs()},
    ),
    "fig_4_2_layer_drift": (make_long_metrics_df, lambda: {"weight_runs": make_checkpoint_runs()}),
    "fig_4_3_progress_strip": (make_long_metrics_df, lambda: {"run_dir": make_png_run()}),
    "fig_4_4_image_embedding": (make_long_metrics_df, lambda: {"cases": make_image_cases(8)}),
    # contact_sheet composites the OTHER PNGs — handled specially in the test.
    "contact_sheet": (make_long_metrics_df, dict),
    # Certificate / learnable-acquisition figures (registered via soft-skipping
    # adapters in plotters/__init__.py; these fixtures prove each fires).
    "certificate_summary": (
        make_long_metrics_df,
        lambda: {"certificate_paths": make_certificate_paths()},
    ),
    "ksd_certificate": (make_long_metrics_df, lambda: {"ksd_result": make_ksd_result()}),
    "learnable_acquisition_pareto": (
        make_long_metrics_df,
        lambda: {"acquisition_arms": make_acquisition_arms()},
    ),
    # Cohort task-ablation bars (faceted validation metrics, method vs ablation).
    "cohort_task_ablation_bars": (make_task_ablation_frame, dict),
    "cmp_mosaic": (make_long_metrics_df, lambda: {"arms": make_arms()}),
    "cmp_metric_bars": (make_long_metrics_df, lambda: {"arms": make_arms(), "baseline": "a"}),
    "cmp_phase_metrics": (
        make_long_metrics_df,
        lambda: {"phase_df": make_phase_table(), "arms": ["a", "b"]},
    ),
    "cmp_coverage": (
        make_long_metrics_df,
        lambda: {
            "phase_df": make_phase_table(),
            "unavailable": [
                {"phase": "test", "metric": "ssim", "variant": "a", "reason": "no reference"}
            ],
            "arms": ["a", "b"],
        },
    ),
    "cmp_rank_reversal": (
        make_long_metrics_df,
        lambda: {"phase_df": make_phase_table(), "arms": ["a", "b"]},
    ),
    "cmp_effect_forest": (
        make_long_metrics_df,
        lambda: {"paired_df": make_paired_rows(), "arms": ["a", "b", "c"], "baseline": "a"},
    ),
    "cmp_subject_lines": (
        make_long_metrics_df,
        lambda: {"paired": make_paired_subjects(), "arms": ["a", "b", "c"], "baseline": "a"},
    ),
    "cmp_phase_bars": (
        make_long_metrics_df,
        lambda: {"phase_df": make_phase_table(), "arms": ["a", "b"]},
    ),
    "cmp_group_bars": (
        make_long_metrics_df,
        lambda: {
            "phase_df": make_phase_table(),
            "arms": ["a", "b"],
            "groups": {"a": "baseline", "b": "arms"},
            "baseline": "a",
        },
    ),
    "cmp_effect_bars": (
        make_long_metrics_df,
        lambda: {
            "paired_df": make_paired_rows(),
            "paired": make_paired_subjects(),
            "arms": ["a", "b", "c"],
            "groups": {"a": "baseline", "b": "constrained", "c": "free"},
            "baseline": "a",
        },
    ),
    "cmp_metric_group_bars": (
        make_long_metrics_df,
        lambda: {
            "paired_df": make_paired_rows(),
            "paired": make_paired_subjects(),
            "arms": ["a", "b", "c"],
            "baseline": "a",
        },
    ),
    "cmp_metric_ladder": (
        make_long_metrics_df,
        lambda: {
            "paired": make_paired_subjects(),
            "arms": ["a", "b", "c"],
            "baseline": "a",
            "ladder": _example_ladder(),
        },
    ),
}


#: The headline figure is shown with its default panels, not the explicit list
#: the smoke test pins.
CATALOG_OVERRIDES: dict[str, tuple[Any, Any]] = {
    "fig_1_2_learning_curves": (make_long_metrics_df, dict),
}

TABLE_EXAMPLES: dict[str, tuple[Any, Any]] = {
    "tab_2_1_main_results": (make_metric_frame, dict),
    "tab_2_2_ablation": (
        make_metric_frame,
        lambda: {"primary_metric": "psnr", "full_method": "full"},
    ),
    "tab_2_3_hyperparameters": (
        make_long_metrics_df,
        lambda: {
            "hyperparameters": [
                {
                    "name": "learning rate",
                    "distribution": "log-uniform",
                    "range": "1e-5 - 1e-3",
                    "final": "2e-4",
                    "criterion": "val PSNR",
                },
                {
                    "name": "batch size",
                    "distribution": "choice",
                    "range": "{2, 4, 8}",
                    "final": "4",
                    "criterion": "val PSNR",
                },
            ]
        },
    ),
    "tab_2_4_dataset_descriptor": (make_long_metrics_df, lambda: {"cohort": make_cohort_dict()}),
    "tab_run_summary": (make_long_metrics_df, dict),
}


def render_catalog(
    out_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 150,
    style: str = "nature",
) -> dict[str, Any]:
    """Render every registered figure and table from its example into ``out_dir``.

    Returns ``{"figures": {id: Path | None}, "reasons": {id: why}, "tables": {...}}``;
    an id with no example is reported as ``no_example``, never skipped quietly.
    """
    from .. import plotters, tables
    from ..style import use_default_style

    out_dir = Path(out_dir)
    use_default_style(style)
    figures: dict[str, Path | None] = {}
    reasons: dict[str, str] = {}
    for fid in plotters.list_available():
        example = CATALOG_OVERRIDES.get(fid) or EXAMPLES.get(fid)
        if example is None:
            figures[fid], reasons[fid] = None, "no_example"
            continue
        frame, kwargs = example
        outcome = plotters.dispatch_detailed(
            frame(), [fid], out_dir / "figures", formats=formats, dpi=dpi, **kwargs()
        )[fid]
        figures[fid] = outcome.path
        if outcome.reason:
            reasons[fid] = outcome.reason
    table_out: dict[str, Any] = {}
    for tid in tables.list_available():
        example = TABLE_EXAMPLES.get(tid)
        if example is None:
            table_out[tid] = None
            continue
        frame, kwargs = example
        table_out.update(tables.dispatch(frame(), [tid], out_dir / "tables", **kwargs()))
    return {"figures": figures, "reasons": reasons, "tables": table_out}
