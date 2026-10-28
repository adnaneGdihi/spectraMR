r"""Plotter registry for the reporting pipeline.

Every plotter module exposes a single ``make(df, out_path, **kwargs)``
function. The registry ``PLOTTERS`` lets the driver iterate over a list
of figure IDs and dispatch the right callable.

To add a new figure type:
1. Create ``my_plot.py`` exposing ``make(df, out_path, ...)``.
2. Register: ``register("fig_id", my_plot.make)`` in this file.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PlotterFn = Callable[..., Path | None]
PLOTTERS: dict[str, PlotterFn] = {}


def register(fig_id: str, fn: PlotterFn) -> None:
    if fig_id in PLOTTERS:
        raise ValueError(f"plotter '{fig_id}' is already registered")
    PLOTTERS[fig_id] = fn


def get(fig_id: str) -> PlotterFn:
    if fig_id not in PLOTTERS:
        raise KeyError(f"unknown plotter '{fig_id}'. Available: {sorted(PLOTTERS)}")
    return PLOTTERS[fig_id]


def list_available() -> list[str]:
    return sorted(PLOTTERS)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Canonical Phase 1 plotters
from . import (  # noqa: E402
    ablation_strip,
    computational_profile,
    failure_gallery,
    headline_pareto,
    learning_curves,
    loss_decomposition,
    predicted_vs_true,
    residual_diagnostics,
    stratified_performance,
)
from .generative import diffusion_diagnostics, gan_diagnostics, vae_diagnostics  # noqa: E402
from .mri_specific import (  # noqa: E402
    cohort_table,
    fiducial_check,
    kspace_error_spectrum,
    kspace_recon,
    recon_panel,
    sr_triptych,
    synthesis_c2c,
)

register("fig_1_1_headline_pareto", headline_pareto.make)
register("fig_1_2_learning_curves", learning_curves.make)
register("fig_1_3_loss_decomposition", loss_decomposition.make)
register("fig_1_4_residual_diagnostics", residual_diagnostics.make)
register("fig_1_5_predicted_vs_true", predicted_vs_true.make)
register("fig_1_9_ablation_strip", ablation_strip.make)
register("fig_1_11_stratified_performance", stratified_performance.make)
register("fig_1_12_failure_gallery", failure_gallery.make)
register("fig_1_15_computational_profile", computational_profile.make)

register("gen_diffusion_diagnostics", diffusion_diagnostics.make)
register("gen_gan_diagnostics", gan_diagnostics.make)
register("gen_vae_diagnostics", vae_diagnostics.make)

register("mri_a1_sr_triptych", sr_triptych.make)
register("mri_a2_synthesis_c2c", synthesis_c2c.make)
register("mri_a7_kspace_recon", kspace_recon.make)
register("mri_a11_cohort_table", cohort_table.make)
register("mri_a12_fiducial_check", fiducial_check.make)
register("mri_recon_panel", recon_panel.make)
register("kspace_error_spectrum", kspace_error_spectrum.make)

# audit_plan_novel.md — Ideas 3, 4, 5 reporting figures.
from . import (  # noqa: E402
    active_acquisition_trajectory,
    bloch_consistency_residual,
    qmap_riemannian_vs_euclidean,
)

register("fig_2_15_active_acquisition_trajectory", active_acquisition_trajectory.make)
register("fig_b3_bloch_consistency_residual", bloch_consistency_residual.make)
register("fig_b7_qmap_riemannian_vs_euclidean", qmap_riemannian_vs_euclidean.make)

# SFC/conformal + fMRI/MRF 2026 diagnostics.
from . import sfc_conformal_plots  # noqa: E402

register("fig_c1_beltrami_field", sfc_conformal_plots.make_beltrami_field)
register("fig_c2_spd_geodesic", sfc_conformal_plots.make_spd_geodesic_trajectory)
register("fig_c3_teichmuller_schedule", sfc_conformal_plots.make_teichmuller_schedule)
register("fig_c4_fingerprint_embedding", sfc_conformal_plots.make_fingerprint_embedding)

from . import metric_distribution  # noqa: E402

register("metric_distribution", metric_distribution.make)

from . import task_ablation_bars  # noqa: E402

register("cohort_task_ablation_bars", task_ablation_bars.make)

from . import bland_altman  # noqa: E402

register("bland_altman", bland_altman.make)

from . import significance_matrix  # noqa: E402

register("significance_matrix", significance_matrix.make)

from . import calibration_coverage  # noqa: E402

register("calibration_coverage", calibration_coverage.make)

from . import forest_plot  # noqa: E402

register("forest_plot", forest_plot.make)

from . import ablation_heatmap  # noqa: E402

register("ablation_heatmap", ablation_heatmap.make)

from .mri_specific import acceleration_sweep  # noqa: E402

register("acceleration_sweep", acceleration_sweep.make)

from . import contact_sheet  # noqa: E402

register("contact_sheet", contact_sheet.make)

# 2026-07-01 reporting upgrade — figures consuming the aggregator's
# final_metrics.json / run_summary.json rows + train/val diagnostics.
from . import metric_correlation, run_summary_card, train_val_gap  # noqa: E402

register("fig_1_16_run_summary_card", run_summary_card.make)
register("fig_1_17_metric_correlation", metric_correlation.make)
register("fig_1_18_train_val_gap", train_val_gap.make)

# QC report: group IQM distribution,
# per-subject QC mosaic, carpet + spike diagnostics.
from .qc import carpet_spike, group_strip, subject_mosaic  # noqa: E402

register("qc_group_strip", group_strip.make)
register("qc_subject_mosaic", subject_mosaic.make)
register("qc_carpet", carpet_spike.make)

# Per-slice distribution, bivariate, embedding and background figures.
from .distributions import FIGURES as _SLICE_FIGURES  # noqa: E402

for _fig_id, _make in _SLICE_FIGURES.items():
    register(_fig_id, _make)

# Figures read from a run's checkpoints and saved validation images.
from .images import FIGURES as _IMAGE_FIGURES  # noqa: E402
from .weights import FIGURES as _WEIGHT_FIGURES  # noqa: E402

for _fig_id, _make in {**_WEIGHT_FIGURES, **_IMAGE_FIGURES}.items():
    register(_fig_id, _make)

# Figures whose callable is not ``make(df, out_path, **kw)``: certificates,
# learnable acquisition and the cross-run plotters (see ``_adapters``).
from ._adapters import ADAPTERS  # noqa: E402

for _fig_id, _adapter in ADAPTERS.items():
    register(_fig_id, _adapter)


def _accepts(fn: PlotterFn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return True
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


#: Why a requested figure produced no file. Three causes used to collapse into a
#: bare ``None``, so a report could not tell "this run has no learning curve to
#: draw" from "this plotter crashed" from "this figure id does not exist" -- and
#: the manifest recorded all three identically as ``status: skipped``. They need
#: different responses from a reader, so they are named separately.
SKIP_UNREGISTERED = "unregistered"
SKIP_NO_DATA = "no_data"
SKIP_RAISED = "raised"


@dataclass(frozen=True)
class FigureOutcome:
    """What became of one requested figure."""

    fig_id: str
    path: Path | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None


def dispatch_detailed(
    df: pd.DataFrame,
    fig_ids: list[str],
    out_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    **kwargs: Any,
) -> dict[str, FigureOutcome]:
    """Run a list of plotters, recording WHY each one produced nothing.

    ``formats``/``dpi`` and any other kwargs are passed only to plotters whose
    signature accepts them (or that declare ``**kwargs``), so heterogeneous
    plotter signatures never raise a TypeError that would silently drop a figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, FigureOutcome] = {}
    for fid in fig_ids:
        out_path = out_dir / f"{fid}.{formats[0]}"
        try:
            fn = get(fid)
        except KeyError:
            results[fid] = FigureOutcome(fid, reason=SKIP_UNREGISTERED)
            continue
        candidate = {"formats": formats, "dpi": dpi, **kwargs}
        call_kwargs = {k: v for k, v in candidate.items() if _accepts(fn, k)}
        try:
            path = fn(df, out_path, **call_kwargs)
        except Exception as exc:  # report-time soft-fail
            import logging

            logging.getLogger(__name__).warning("plotter '%s' failed: %s", fid, exc)
            results[fid] = FigureOutcome(fid, reason=f"{SKIP_RAISED}:{type(exc).__name__}: {exc}")
            continue
        # A plotter returning None is its documented "I have nothing to draw"
        # signal, not a fault -- e.g. learning curves after a prediction run.
        results[fid] = (
            FigureOutcome(fid, path=path)
            if path is not None
            else FigureOutcome(fid, reason=SKIP_NO_DATA)
        )
    return results


def dispatch(
    df: pd.DataFrame,
    fig_ids: list[str],
    out_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    **kwargs: Any,
) -> dict[str, Path | None]:
    """Run a list of plotters; return ``{fig_id: primary_output_path | None}``.

    The reason-losing view of :func:`dispatch_detailed`, kept because callers that
    only need the paths should not have to unwrap outcomes. It delegates rather
    than duplicating the loop -- two dispatch implementations would drift, and the
    one that drifted would be the one nothing renders from.
    """
    return {
        fid: o.path
        for fid, o in dispatch_detailed(
            df, fig_ids, out_dir, formats=formats, dpi=dpi, **kwargs
        ).items()
    }
