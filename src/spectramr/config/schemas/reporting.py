r"""Reporting pipeline configuration schema (v6.0).

Drives `spectramr.infrastructure.reporting.generate_report` at the end of
every training run. Backward-compatible: when omitted from YAML the
reporting hook is disabled.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import ReportStyle, ReportTask
from .renames import reject_renamed_keys


class ReportingSettings(BaseModel):
    """Per-experiment reporting hook configuration.

    Add a ``reporting:`` block to any v6.0 YAML to enable the canonical
    end-of-training figure / table generator. Example:

    .. code-block:: yaml

        reporting:
          enabled: true
          task: reconstruction      # or: synthesis, super_resolution, gan, diffusion, vae
          method_name: my_baseline
          out_subdir: report
          figures: null            # null = use task preset
          tables:  null            # null = use task preset
    """

    # NN#1: all config schemas are frozen (no caller mutates reporting config;
    # grep-verified).
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), frozen=True)

    # `extra="forbid"` above already REFUSES `per_case_metrics`, so this is not
    # a silent-drop fix -- it is a message fix. Pydantic's own refusal reads
    # "Extra inputs are not permitted" and never names `per_call_metrics`, so
    # the record's guided text (what it became, why, and the one-line fixer)
    # was written on 2026-08-05 and had never once reached a user. Every other
    # block owning a `raise` record mounts this; `reporting` was the one that
    # did not.
    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("reporting")))

    enabled: bool = Field(
        default=False,
        description="If True, run the reporting pipeline at the end of training.",
    )
    task: ReportTask = Field(
        default=ReportTask.DEFAULT,
        description="Task preset that selects the default figure / table set.",
    )
    method_name: str | None = Field(
        default=None,
        description="Method label used in plots / tables. Defaults to the experiment dir name.",
    )
    out_subdir: str = Field(
        default="report",
        description="Subdirectory of the experiment output dir to write artifacts into.",
    )
    style: ReportStyle = Field(
        default=ReportStyle.NATURE,
        description="House style for figures: nature (default), ieee, poster or slides.",
    )
    formats: list[str] = Field(
        default_factory=lambda: ["pdf", "png"],
        description="Figure output formats. Subset of {pdf, png, tiff, eps, svg}.",
    )
    dpi: int = Field(
        default=600,
        ge=72,
        le=1200,
        description="Raster DPI for png/tiff outputs (Nature wants 300-600).",
    )
    panel_labels: bool = Field(
        default=True, description="Stamp bold lowercase panel letters (a, b, c)."
    )
    submission_bundle: bool = Field(
        default=False,
        description="Also emit submission/ (600-dpi TIFF + per-figure caption .txt).",
    )
    tikz: bool = Field(
        default=False,
        description=(
            "Also emit LaTeX-native TikZ/pgfplots figures (figures/tikz/*.tex + "
            "*.tikz) for the key data-driven plots. Pure text emission — no "
            "LaTeX needed at write time; compile with `tectonic -X compile`."
        ),
    )
    emit_manifest: bool = Field(
        default=True,
        description="Write report_manifest.json (artifact index + provenance).",
    )
    n_report_cases: int = Field(
        default=6, ge=0, description="Number of qualitative cases to record/render."
    )
    case_selection: Literal["best_median_worst", "random", "first"] = Field(
        default="best_median_worst",
        description="Which validation cases the recorder keeps.",
    )

    # --- QC report -------------------------
    # These fire only when ``enabled`` is True, so existing runs are untouched.
    qc_figures: bool = Field(
        default=True,
        description=(
            "Include the QC plotters (per-subject QC mosaic, group IQM "
            "strip distribution, carpet/spike diagnostics) in the figure set. The "
            "group strip auto-tracks whatever metrics/losses the run computed."
        ),
    )
    per_call_metrics: bool = Field(
        default=True,
        description=(
            "Emit <run_dir>/per_call_metrics.csv — one row per validation CALL "
            "over the whole run, unbounded, versus the recorder's retained "
            "best/median/worst pool. Each row carries the BATCH-AGGREGATE metrics "
            "under a case_id derived from the step, so it is one point per "
            "validation event, NOT per sample: the run computes metrics batch-wise "
            "and no per-sample value exists to record (#503). Consumers that need "
            "a genuine across-case distribution must not read it as one — "
            "qc_group_strip drops a metric whose values have no spread. "
            "Renamed from `per_case_metrics` on 2026-08-05: the old name promised "
            "a per-CASE distribution the artifact never contained, and a plotter "
            "believed it (rendering eight copies of one scalar as a box plot). "
            "The retired spelling raises — see config/schemas/renames.py."
        ),
    )
    html_report: bool = Field(
        default=True,
        description=(
            "Assemble a self-contained QC HTML report "
            "(<out_subdir>/qc_report.html) bundling the figures, group IQM "
            "distributions, per-subject reportlets and tables. Interactive group "
            "plots when plotly is installed; static-PNG fallback otherwise."
        ),
    )
    interactive: bool = Field(
        default=True,
        description=(
            "Emit the interactive plotly layer in the HTML report (hoverable "
            "group IQMs, a 2-D per-subject viewer, a 3-D volume viewer when "
            "volumes are recorded, interactive learning curves / metric "
            "distributions / Bland-Altman). plotly.js is inlined once so the "
            "report stays self-contained and works offline; when plotly is "
            "absent every section falls back to the embedded static PNG."
        ),
    )
    record_volumes: bool = Field(
        default=False,
        description=(
            "Preserve full 3-D [Z, H, W] volumes for the interactive volumetric "
            "viewer (opt-in). No-op on single-slice acquisitions (e.g. M4Raw), "
            "where the 3-D viewer soft-skips; useful for volumetric datasets."
        ),
    )

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("formats must list at least one output format")
        allowed = {"pdf", "png", "tiff", "eps", "svg"}
        bad = [f for f in v if f not in allowed]
        if bad:
            raise ValueError(f"unknown figure format(s) {bad}; allowed {sorted(allowed)}")
        return v

    figures: list[str] | None = Field(
        default=None,
        description="Override the task preset's figure list. None = preset default.",
    )
    tables: list[str] | None = Field(
        default=None,
        description="Override the task preset's table list. None = preset default.",
    )
    metrics: list[str] | None = Field(
        default=None,
        description=(
            "Metrics that the main-results table (tab_2_1) should include. "
            "Names must match either the existing spectramr.core.metrics registry or "
            "the Tier-1 IQMs in spectramr.infrastructure.reporting.metrics "
            "(vif, fsim, haarpsi, gmsd). None = let the table auto-detect from "
            "the experiment's eval JSON."
        ),
    )
    cohort: dict[str, Any] | None = Field(
        default=None,
        description="Cohort descriptor passed to the dataset / cohort plotters.",
    )
    hyperparameters: list[dict[str, Any]] | None = Field(
        default=None,
        description="Hyperparameter table rows.",
    )
    extra_runs: list[str] | None = Field(
        default=None,
        description="Other experiment dirs to fold into the same report (e.g. baselines).",
    )
    fail_on_error: bool = Field(
        default=False,
        description=(
            "If False (default), reporting failures only log a warning so they "
            "cannot break a long training run's wrap-up. Set True for CI."
        ),
    )
