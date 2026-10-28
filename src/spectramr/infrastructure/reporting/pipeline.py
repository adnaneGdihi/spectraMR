r"""Top-level orchestrator for the reporting pipeline.

Single entry-point ``generate_report(experiment_dir, ...)`` produces all
canonical figures + tables (configurable subset) into
``<experiment_dir>/report/``. Driven by `ReportingSettings`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import plotters, tables
from .aggregator import aggregate, aggregate_many
from .metadata import RunMetadata, detect_metadata, run_provenance
from .run_status import log_status, read_run_status, status_notes

logger = logging.getLogger(__name__)


# Default figure / table sets per experiment task.
TASK_PRESETS: dict[str, dict[str, list[str]]] = {
    "default": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "fig_1_15_computational_profile",
            "fig_1_16_run_summary_card",
            "fig_1_17_metric_correlation",
            "fig_1_18_train_val_gap",
            "mri_a11_cohort_table",
            "qc_group_strip",
            "qc_subject_mosaic",
            "qc_carpet",
            "contact_sheet",
        ],
        "tables": [
            "tab_2_1_main_results",
            "tab_2_4_dataset_descriptor",
        ],
    },
    "reconstruction": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "mri_recon_panel",
            "kspace_error_spectrum",
            "fig_1_5_predicted_vs_true",
            "fig_1_4_residual_diagnostics",
            "metric_distribution",
            "fig_1_11_stratified_performance",
            "fig_1_12_failure_gallery",
            "acceleration_sweep",
            "fig_1_15_computational_profile",
            "fig_1_16_run_summary_card",
            "fig_1_17_metric_correlation",
            "fig_1_18_train_val_gap",
            "fig_3_1_metric_violin",
            "fig_3_2_subject_bars",
            "fig_3_3_metric_scatter",
            "fig_3_6_error_decomposition",
            "fig_3_7_metric_embedding",
            "fig_3_8_intensity_joint",
            "fig_3_9_background_zoom",
            "fig_4_1_weight_trajectory",
            "fig_4_2_layer_drift",
            "fig_4_3_progress_strip",
            "fig_4_4_image_embedding",
            "mri_a7_kspace_recon",
            "mri_a11_cohort_table",
            "qc_group_strip",
            "qc_subject_mosaic",
            "qc_carpet",
            "contact_sheet",
        ],
        "tables": [
            "tab_2_1_main_results",
            "tab_2_4_dataset_descriptor",
        ],
    },
    "synthesis": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "mri_recon_panel",
            "fig_1_5_predicted_vs_true",
            "metric_distribution",
            "fig_1_11_stratified_performance",
            "fig_1_12_failure_gallery",
            "fig_1_16_run_summary_card",
            "fig_1_17_metric_correlation",
            "fig_1_18_train_val_gap",
            "fig_3_1_metric_violin",
            "fig_3_2_subject_bars",
            "fig_3_3_metric_scatter",
            "fig_3_6_error_decomposition",
            "fig_3_7_metric_embedding",
            "fig_3_8_intensity_joint",
            "fig_3_9_background_zoom",
            "fig_4_1_weight_trajectory",
            "fig_4_2_layer_drift",
            "fig_4_3_progress_strip",
            "fig_4_4_image_embedding",
            "mri_a2_synthesis_c2c",
            "mri_a11_cohort_table",
            "qc_group_strip",
            "qc_subject_mosaic",
            "qc_carpet",
            "contact_sheet",
        ],
        "tables": [
            "tab_2_1_main_results",
            "tab_2_4_dataset_descriptor",
        ],
    },
    "super_resolution": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "mri_recon_panel",
            "kspace_error_spectrum",
            "fig_1_5_predicted_vs_true",
            "metric_distribution",
            "fig_1_12_failure_gallery",
            "acceleration_sweep",
            "mri_a1_sr_triptych",
            "fig_1_16_run_summary_card",
            "fig_1_17_metric_correlation",
            "fig_1_18_train_val_gap",
            "fig_3_1_metric_violin",
            "fig_3_2_subject_bars",
            "fig_3_3_metric_scatter",
            "fig_3_6_error_decomposition",
            "fig_3_7_metric_embedding",
            "fig_3_8_intensity_joint",
            "fig_3_9_background_zoom",
            "fig_4_1_weight_trajectory",
            "fig_4_2_layer_drift",
            "fig_4_3_progress_strip",
            "fig_4_4_image_embedding",
            "mri_a11_cohort_table",
            "qc_group_strip",
            "qc_subject_mosaic",
            "qc_carpet",
            "contact_sheet",
        ],
        "tables": [
            "tab_2_1_main_results",
            "tab_2_4_dataset_descriptor",
        ],
    },
    "gan": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "gen_gan_diagnostics",
            "fig_1_15_computational_profile",
            "fig_1_16_run_summary_card",
            "fig_1_18_train_val_gap",
            "contact_sheet",
        ],
        "tables": ["tab_2_1_main_results"],
    },
    "diffusion": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "gen_diffusion_diagnostics",
            "fig_1_15_computational_profile",
            "fig_1_16_run_summary_card",
            "fig_1_18_train_val_gap",
            "contact_sheet",
        ],
        "tables": ["tab_2_1_main_results"],
    },
    "vae": {
        "figures": [
            "fig_1_2_learning_curves",
            "fig_1_3_loss_decomposition",
            "gen_vae_diagnostics",
            "fig_1_15_computational_profile",
            "fig_1_16_run_summary_card",
            "fig_1_18_train_val_gap",
            "contact_sheet",
        ],
        "tables": ["tab_2_1_main_results"],
    },
    # Conformal-calibration / certification arms (run_calibration strategies):
    # coverage-vs-target reliability, pairwise significance, agreement, and the
    # metric distribution — the figures that grade a CERTIFICATE, not a recon.
    "calibration": {
        "figures": [
            "fig_1_2_learning_curves",
            "calibration_coverage",
            "significance_matrix",
            "bland_altman",
            "metric_distribution",
            "fig_1_16_run_summary_card",
            "contact_sheet",
        ],
        "tables": [
            "tab_2_1_main_results",
            "tab_2_4_dataset_descriptor",
        ],
    },
}


def _resolve_preset(task: str) -> dict[str, list[str]]:
    key = str(getattr(task, "value", task)).lower()
    if key not in TASK_PRESETS:
        raise ValueError(f"unknown report task {task!r}; choose from {sorted(TASK_PRESETS)}")
    return TASK_PRESETS[key]


def _contact_sheet_last(ids: list[str]) -> list[str]:
    """Move ``contact_sheet`` to the end, preserving the order of the rest.

    ``contact_sheet`` composites the *other* emitted PNGs, so it must render
    after them. Any code path that BUILDS a figure list has to re-apply this --
    a preset that happens to list it early, or a union that appends after it,
    would otherwise composite a directory that is not finished yet.
    """
    rest = [f for f in ids if f != "contact_sheet"]
    if "contact_sheet" in ids:
        rest.append("contact_sheet")
    return rest


def _all_figure_ids() -> list[str]:
    """Every registered plotter, with ``contact_sheet`` last.

    No longer the default figure set -- ``figures: null`` resolves to the task
    preset (#719). Still the source for the ``qc_*`` union and for callers that
    genuinely want everything.
    """
    return _contact_sheet_last(plotters.list_available())  # list_available is sorted


def _load_domain_artifacts(
    experiment_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Source data for the domain-specific plotters the aggregator frame cannot
    carry (the novel-2026 physics / geometry figures).

    The report is the plotting SSOT, so a run advertises this data by dropping
    per-figure artifacts under ``<run>/report_artifacts/`` and the report consumes
    them; an absent artifact simply lets that figure soft-skip (honest "plot all
    that apply"):

      ``report_artifacts/<fig_id>.csv``        -> used as the DataFrame for
                                                  ``<fig_id>`` (the df-column
                                                  plotters: bloch / beltrami / spd /
                                                  teichmuller / fingerprint read
                                                  their columns from it)
      ``report_artifacts/active_acquisition/`` -> ``*.csv`` routed as ``csv_paths=``
      ``report_artifacts/qmap_slices.npz``     -> routed as ``slices={name: array}``

    Returns ``(df_overrides, domain_kwargs)``.
    """
    root = experiment_dir / "report_artifacts"
    df_overrides: dict[str, pd.DataFrame] = {}
    domain_kwargs: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return df_overrides, domain_kwargs
    for csv in sorted(root.glob("*.csv")):
        try:
            df_overrides[csv.stem] = pd.read_csv(csv)
        except Exception as exc:  # a bad artifact must never break the report
            logger.warning("report artifact %s unreadable: %s", csv, exc)
    aa_dir = root / "active_acquisition"
    if aa_dir.is_dir():
        paths = sorted(aa_dir.glob("*.csv"))
        if paths:
            domain_kwargs["fig_2_15_active_acquisition_trajectory"] = {"csv_paths": paths}
    qmap = root / "qmap_slices.npz"
    if qmap.exists():
        import numpy as np

        try:
            data = np.load(qmap)
            domain_kwargs["fig_b7_qmap_riemannian_vs_euclidean"] = {
                "slices": {k: data[k] for k in data.files}
            }
        except Exception as exc:
            logger.warning("report artifact %s unreadable: %s", qmap, exc)
    return df_overrides, domain_kwargs


def _validate_figure_ids(fig_ids: list[str]) -> None:
    available = set(plotters.list_available())
    unknown = [f for f in fig_ids if f not in available]
    if unknown:
        raise ValueError(f"unknown figure id(s) {unknown}; available {sorted(available)}")


def _validate_table_ids(tab_ids: list[str]) -> None:
    available = set(tables.list_available())
    unknown = [t for t in tab_ids if t not in available]
    if unknown:
        raise ValueError(f"unknown table id(s) {unknown}; available {sorted(available)}")


def generate_report(
    experiment_dir: str | Path,
    *,
    task: str = "default",
    style: str = "nature",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    panel_labels: bool = True,
    method_name: str | None = None,
    figures: list[str] | None = None,
    tables_: list[str] | None = None,
    metrics: list[str] | None = None,
    cohort: dict | None = None,
    cases_sr: list[dict] | None = None,
    cases_synth: list[dict] | None = None,
    cases_kspace: list[dict] | None = None,
    cases_failure: list[dict] | None = None,
    predictions_df: pd.DataFrame | None = None,
    stratified_df: pd.DataFrame | None = None,
    hyperparameters: list[dict] | None = None,
    seed: int | None = None,
    dataset_version: str | None = None,
    config_path: str | Path | None = None,
    extra_runs: list[str | Path] | None = None,
    out_subdir: str = "report",
    emit_manifest: bool = True,
    submission_bundle: bool = False,
    tikz: bool = False,
    qc_figures: bool = True,
    html_report: bool = True,
    interactive: bool = True,
    view: str = "selected",
) -> dict[str, Any]:
    """Generate the canonical report for a single experiment run.

    Returns a dict ``{"figures": {...}, "tables": {...}, "out_dir": Path}``.

    Soft-fails on per-plotter exceptions (logs and continues) so a
    bad plotter cannot break a long training run's wrap-up.
    """
    experiment_dir = Path(experiment_dir)
    out_dir = experiment_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"

    from .style import set_panel_labels, use_default_style

    use_default_style(style)
    set_panel_labels(panel_labels)

    method = method_name or experiment_dir.name
    preset = _resolve_preset(task)
    # ``figures: null`` means THE TASK PRESET (#719). It used to mean every
    # registered plotter, while `ReportingSettings.figures` and
    # docs/reporting_pipeline.rst both documented the preset -- a three-way
    # contradiction in which the code was the only party saying "all 44". The
    # presets are curated per task (default 11, reconstruction 20, gan 7), so
    # honouring them is also what makes `task:` mean anything: selecting a task
    # changed nothing about the figures before this.
    #
    # Data-less plotters still soft-skip, so "all 44" was never wrong output --
    # it was ~33 extra render attempts and a report whose contents did not
    # depend on the declared task.
    explicit = figures is not None
    fig_ids = list(figures) if explicit else list(preset["figures"])
    tab_ids = tables_ if tables_ is not None else preset["tables"]
    if qc_figures:
        # ADDITIVE, not a filter. `qc_figures` promises to "include the QC
        # plotters", and 4 of the 8 presets (gan, diffusion, vae, calibration)
        # list no `qc_*` at all -- so a pure filter would have left the knob
        # advertising an inclusion it never performed (pitfall #15) the moment
        # the preset became authoritative. An EXPLICIT `figures` list is still
        # exactly what was asked for; the union applies only to the preset path.
        if not explicit:
            fig_ids += [f for f in _all_figure_ids() if f.startswith("qc_") and f not in fig_ids]
    else:
        fig_ids = [f for f in fig_ids if not f.startswith("qc_")]
    fig_ids = _contact_sheet_last(fig_ids)
    _validate_figure_ids(fig_ids)
    _validate_table_ids(tab_ids)

    # Build the long-format frame
    if extra_runs:
        df = aggregate_many(
            [experiment_dir, *extra_runs],
            method_names=[method, *[Path(r).name for r in extra_runs]],
        )
    else:
        df = aggregate(experiment_dir, method_name=method)

    # A crashed run is still reported, since its report is how the crash gets
    # read; its status is stamped into the summary, the manifest and the table.
    statuses = {method: read_run_status(experiment_dir)}
    statuses.update({Path(r).name: read_run_status(r) for r in extra_runs or []})
    for name, status in statuses.items():
        log_status(name, status, action="reported with its status stamped")

    # What the caller did not pass, the run directory recorded (#2502).
    recorded = run_provenance(experiment_dir)
    metadata = detect_metadata(
        seed=seed,
        dataset_version=dataset_version or recorded["dataset_version"],
        config_path=config_path or recorded["config_path"],
        config_hash=recorded["config_hash"],
    )

    # Per-plotter kwargs router
    plot_kwargs: dict[str, dict[str, Any]] = {}
    if cohort is not None:
        plot_kwargs["mri_a11_cohort_table"] = {"cohort": cohort}
    if cases_sr is not None:
        plot_kwargs["mri_a1_sr_triptych"] = {"cases": cases_sr}
    if cases_synth is not None:
        plot_kwargs["mri_a2_synthesis_c2c"] = {"cases": cases_synth}
    if cases_kspace is not None:
        plot_kwargs["mri_a7_kspace_recon"] = {"cases": cases_kspace}
    if cases_failure is not None:
        plot_kwargs["fig_1_12_failure_gallery"] = {"cases": cases_failure}
    if predictions_df is not None:
        plot_kwargs["fig_1_4_residual_diagnostics"] = {"predictions_df": predictions_df}
        plot_kwargs["fig_1_5_predicted_vs_true"] = {"predictions_df": predictions_df}
    if stratified_df is not None:
        plot_kwargs["fig_1_11_stratified_performance"] = {"stratified_df": stratified_df}

    # Auto-load any recorded validation cases from disk and route them into the
    # qualitative plotters. Explicit caller kwargs (set above) always win — the
    # guards below only fill figures the caller did not already supply.
    from .cases.loader import load_report_cases
    from .views import checkpoint_steps

    # The run this report is about; extra runs never choose its checkpoint.
    selected = checkpoint_steps(df[df["method"] == method] if "method" in df else df, view)
    checkpoint_step = next(iter(selected.values()), None)
    disk_cases = load_report_cases(
        experiment_dir,
        task=str(getattr(task, "value", task)),
        step=checkpoint_step,
    )
    case_routing = {
        "cases_kspace": ("mri_a7_kspace_recon", "cases"),
        "cases_sr": ("mri_a1_sr_triptych", "cases"),
        "cases_synth": ("mri_a2_synthesis_c2c", "cases"),
        "cases_failure": ("fig_1_12_failure_gallery", "cases"),
    }
    img_cases = (
        disk_cases.get("cases_kspace")
        or disk_cases.get("cases_sr")
        or disk_cases.get("cases_synth")
    )
    for payload_key, (fid, kw) in case_routing.items():
        val = disk_cases.get(payload_key)
        if val and fid not in plot_kwargs:
            plot_kwargs.setdefault(fid, {})[kw] = val
    from .comparison.slice_cases import slice_case_rows
    from .plotters.distributions import CASE_FIGURES, SLICE_FIGURES

    slice_df = slice_case_rows(
        experiment_dir, step=checkpoint_step, run_id=str(method), method=str(method)
    )
    if not slice_df.empty:
        for fid in SLICE_FIGURES:
            plot_kwargs.setdefault(fid, {}).setdefault("slice_df", slice_df)
    plot_kwargs.setdefault("fig_4_1_weight_trajectory", {}).setdefault(
        "weight_runs", {str(method): experiment_dir}
    )
    plot_kwargs.setdefault("fig_4_2_layer_drift", {}).setdefault(
        "weight_runs", {str(method): experiment_dir}
    )
    plot_kwargs.setdefault("fig_4_3_progress_strip", {}).setdefault("run_dir", experiment_dir)
    if img_cases:
        plot_kwargs.setdefault("fig_4_4_image_embedding", {}).setdefault("cases", img_cases)
        for fid in CASE_FIGURES:
            plot_kwargs.setdefault(fid, {}).setdefault("cases", img_cases)
        plot_kwargs.setdefault("mri_recon_panel", {}).setdefault("cases", img_cases)
        plot_kwargs.setdefault("kspace_error_spectrum", {}).setdefault("cases", img_cases)
    if disk_cases.get("predictions_df") is not None:
        for fid in (
            "fig_1_4_residual_diagnostics",
            "fig_1_5_predicted_vs_true",
            "bland_altman",
            "metric_distribution",
        ):
            plot_kwargs.setdefault(fid, {}).setdefault(
                "predictions_df", disk_cases["predictions_df"]
            )
    if disk_cases.get("stratified_df") is not None:
        plot_kwargs.setdefault("fig_1_11_stratified_performance", {}).setdefault(
            "stratified_df", disk_cases["stratified_df"]
        )

    # QC report inputs: per_call_metrics.csv → group strip (one point per
    # case); recorded/PNG-recovered images → per-subject mosaic + carpet.
    per_call_df = None
    _pc_csv = experiment_dir / "per_call_metrics.csv"
    if _pc_csv.exists():
        try:
            per_call_df = pd.read_csv(_pc_csv)
        except Exception:
            per_call_df = None
    if per_call_df is not None and not per_call_df.empty:
        plot_kwargs.setdefault("qc_group_strip", {}).setdefault("per_call_df", per_call_df)
        from .per_case import call_unit, validation_batch_size

        sweep = plot_kwargs.setdefault("acceleration_sweep", {})
        sweep.setdefault("per_call_df", per_call_df)
        sweep.setdefault("call_unit", call_unit([validation_batch_size(experiment_dir)]))
    if disk_cases.get("predictions_df") is not None:
        plot_kwargs.setdefault("qc_group_strip", {}).setdefault(
            "predictions_df", disk_cases["predictions_df"]
        )
    if img_cases:
        plot_kwargs.setdefault("qc_subject_mosaic", {}).setdefault("cases", img_cases)
        plot_kwargs.setdefault("qc_carpet", {}).setdefault("cases", img_cases)
        # fiducial-check is an image-case plotter that was never fed -> feed it so
        # it renders on any run with recorded cases instead of always soft-skipping.
        plot_kwargs.setdefault("mri_a12_fiducial_check", {}).setdefault("cases", img_cases)

    # Domain-specific artifacts (report_artifacts/ convention) for the physics /
    # geometry figures whose data the aggregator frame cannot carry. Absent
    # artifacts leave those figures to soft-skip.
    df_overrides, domain_kwargs = _load_domain_artifacts(experiment_dir)
    for fid, kw in domain_kwargs.items():
        plot_kwargs.setdefault(fid, {}).update(kw)

    table_kwargs: dict[str, dict[str, Any]] = {}
    if cohort is not None:
        table_kwargs["tab_2_4_dataset_descriptor"] = {"cohort": cohort}
    if hyperparameters is not None:
        table_kwargs["tab_2_3_hyperparameters"] = {"hyperparameters": hyperparameters}
    table_kwargs.setdefault("tab_2_1_main_results", {}).update(
        view=view, run_notes=status_notes(statuses)
    )
    if metrics is not None:
        table_kwargs["tab_2_1_main_results"]["metrics"] = metrics
    for fid in ("acceleration_sweep", "qc_group_strip", "fig_1_16_run_summary_card"):
        plot_kwargs.setdefault(fid, {}).update(view=view, checkpoint_step=checkpoint_step)
    if extra_runs and "method" in df.columns:
        # one run's card and summary: the extra runs belong to the comparison figures
        df_overrides.setdefault("fig_1_16_run_summary_card", df[df["method"] == method])
        plot_kwargs["acceleration_sweep"]["expected_methods"] = [
            str(m) for m in pd.unique(df["method"])
        ]

    # Dispatch.
    #
    # `dispatch_detailed` rather than `dispatch` so a figure that produces no
    # file carries WHY. The requested set is identical whether this report
    # follows training or a prediction run -- `fig_ids` above derives only from
    # `task`/`figures`/`qc_figures`, never from the caller -- so what differs
    # between the two is purely which figures have data. Recording that as a
    # reason is what makes the two runs comparable: every requested figure is
    # either emitted or accounted for, instead of quietly absent (pitfall #16).
    fig_results: dict[str, Path | None] = {}
    fig_reasons: dict[str, str] = {}
    for fid in fig_ids:
        kwargs = {"metadata": metadata, **plot_kwargs.get(fid, {})}
        # A per-figure ``report_artifacts/<fid>.csv`` overrides the aggregator df
        # for that plotter (the df-column physics/geometry figures); default df
        # otherwise.
        fig_df = df_overrides.get(fid, df)
        out = plotters.dispatch_detailed(fig_df, [fid], fig_dir, formats=formats, dpi=dpi, **kwargs)
        for k, outcome in out.items():
            fig_results[k] = outcome.path
            if outcome.reason is not None:
                fig_reasons[k] = outcome.reason

    table_results: dict[str, dict[str, Path] | None] = {}
    for tid in tab_ids:
        kwargs = table_kwargs.get(tid, {})
        out = tables.dispatch(df, [tid], tab_dir, **kwargs)
        table_results.update(out)

    # LaTeX-native TikZ bundle (reporting.tikz knob) — emitted next to the
    # raster/vector figures and stamped into the manifest like any artifact.
    tikz_results: dict[str, Path] = {}
    if tikz:
        from .tikz_export import export_tikz_figures

        tikz_results = export_tikz_figures(df, fig_dir / "tikz")

    summary_path = out_dir / "report_summary.md"
    _write_summary(
        summary_path,
        fig_results,
        table_results,
        metadata,
        method,
        task,
        tikz=tikz_results,
        df=df[df["method"] == method] if "method" in df.columns else df,
        view=view,
        status=statuses[method].label,
    )
    if emit_manifest:
        from .report_manifest import write_report_manifest, write_submission_bundle

        # flatten figure results to primary paths; TikZ artifacts ride along
        fig_paths = dict(fig_results)
        write_report_manifest(
            out_dir,
            figures={**fig_paths, **tikz_results},
            tables=table_results,
            metadata=metadata,
            figure_reasons=fig_reasons,
            runs=[{"arm": n, "included": True, **s.to_dict()} for n, s in statuses.items()],
        )
        if submission_bundle:
            write_submission_bundle(out_dir, fig_paths)

    # Self-contained QC HTML report (reporting.html_report knob),
    # bundling the figures + interactive/static group IQM section + reportlets.
    html_path: Path | None = None
    if html_report:
        try:
            from .html_report import build_html_report

            html_path = build_html_report(
                out_dir,
                figures=fig_results,
                tables=table_results,
                metadata=metadata,
                per_call_df=per_call_df,
                predictions_df=disk_cases.get("predictions_df"),
                aggregated_df=df,
                cases=img_cases,
                interactive=interactive,
                method=method,
                task=str(getattr(task, "value", task)),
            )
        except Exception as exc:
            logger.warning("html report generation failed: %s", exc)

    return {
        "figures": fig_results,
        "tables": table_results,
        "tikz": tikz_results,
        "out_dir": out_dir,
        "summary": summary_path,
        "html": html_path,
    }


def _headline_lines(df: pd.DataFrame | None, view: str = "selected") -> list[str]:
    """Markdown table of the headline metrics at the ``view`` pass.

    Read through :mod:`.views`, never from ``final_metrics.json``'s ``best`` block,
    whose entries come from different checkpoints.
    """
    if df is None or df.empty or "split" not in df.columns:
        return []
    from .views import snapshot

    snap = snapshot(df, view)
    if snap.empty or "headline" not in snap.columns:
        return []
    snap = snap[snap["headline"].astype(bool)]
    if snap.empty:
        return []
    from .style import pretty_label

    step = int(snap["step"].iloc[0])
    rows = snap.groupby("metric")["value"].last().sort_index()
    lines = ["", f"## Headline numbers ({view} pass, step {step})", "", "| metric | value |",
             "|---|---|"]
    lines += [f"| {pretty_label(str(m))} | {v:.4g} |" for m, v in rows.items()]
    return lines


def _write_summary(
    path: Path,
    figures: dict[str, Path | None],
    tables_: dict[str, dict[str, Path] | None],
    metadata: RunMetadata,
    method: str,
    task: str,
    *,
    tikz: dict[str, Path] | None = None,
    df: pd.DataFrame | None = None,
    view: str = "selected",
    status: str = "not read",
) -> None:
    lines = [
        f"# Report — `{method}`",
        "",
        f"- Run status: `{status}`",
        f"- Task preset: `{task}`",
        f"- Checkpoint view: `{view}`",
        f"- Git: `{metadata.git_commit[:10]}`{' (dirty)' if metadata.git_dirty else ''}",
        f"- Seed: `{metadata.seed}`",
        f"- Dataset: `{metadata.dataset_version}`",
        f"- Generated (UTC): `{metadata.timestamp_utc}`",
    ]
    lines += _headline_lines(df, view)
    lines += ["", "## Figures"]
    for fid, p in figures.items():
        if p is None:
            lines.append(f"- **{fid}**: _skipped (no data)_")
            continue
        rel = p.relative_to(path.parent)
        lines.append(f"- **{fid}**: [{p.name}]({rel})")
        # embed the PNG twin inline so the summary reads as a visual report
        png = p.with_suffix(".png")
        if png.exists():
            lines += ["", f"  ![{fid}]({png.relative_to(path.parent)})", ""]
    if tikz:
        lines += ["", "## TikZ (LaTeX-native)"]
        lines += [
            f"- **{fid}**: [{p.name}]({p.relative_to(path.parent)})" for fid, p in tikz.items()
        ]
    lines += ["", "## Tables"]
    for tid, group in tables_.items():
        if group is None:
            lines.append(f"- **{tid}**: _skipped_")
        else:
            ext = ", ".join(f"[{p.name}]({p.relative_to(path.parent)})" for p in group.values())
            lines.append(f"- **{tid}**: {ext}")
    path.write_text("\n".join(lines) + "\n")
