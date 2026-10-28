"""Smoke sweep over every plotter under ``spectramr.infrastructure.reporting.plotters``.

This is the *first* test file targeting the plotter package. The intent is
broad smoke coverage rather than deep correctness:

* For every public plotter module (one per ``*.py`` file, recursively, minus
  helpers prefixed with ``_``), build a minimum-viable synthetic fixture that
  matches the plotter's documented input contract.
* Invoke the canonical entry point (``make`` for the Phase-1 family,
  ``render_*`` for the optional/wrapper plotters).
* Assert the plotter returns a non-``None`` output path and that the file on
  disk is non-empty — equivalent to "produces a non-empty matplotlib Figure"
  for plotters that save-and-close inside ``make`` (which all of them do).

Plotters whose smoke fixture cannot be synthesised cheaply (e.g. those that
need real NIfTI / model outputs / multi-stage calibration JSON) are marked
``xfail`` rather than silently skipped, per the repo CLAUDE.md pitfall #10
("warnings are not OK").

Notes:

* Matplotlib is forced to ``Agg`` *before* any pyplot import so the test
  works in headless CI and never opens a GUI window.
* ``plt.close("all")`` runs in fixture teardown to keep figure-manager
  memory bounded across the parametrised sweep.
* If the ``plotters`` package is not importable on the current branch the
  whole module is skipped via ``pytest.importorskip`` — this keeps the
  test honest on branches where the plotter work has not landed yet.
"""

from __future__ import annotations

# IMPORTANT: ``Agg`` must be selected before pyplot is imported anywhere.
import matplotlib

matplotlib.use("Agg")

import importlib
import pkgutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Skip the whole module on branches that don't carry the plotter package
# (e.g. ``main`` prior to the reporting-pipeline merge).
plotters_pkg = pytest.importorskip(
    "spectramr.infrastructure.reporting.plotters",
    reason="reporting.plotters package not available on this branch",
)


# ---------------------------------------------------------------------------
# Plotter discovery
# ---------------------------------------------------------------------------


def _discover_plotter_modules() -> list[tuple[str, Any]]:
    """Walk the ``plotters`` package and return ``(short_name, module)`` pairs.

    Helpers prefixed with ``_`` and the ``__init__`` registry are filtered out.
    The short name is used as the parametrise id so failing nodes are easy
    to read in pytest output.
    """
    discovered: list[tuple[str, Any]] = []
    pkg_path = Path(plotters_pkg.__file__).parent
    for mod_info in pkgutil.walk_packages(
        [str(pkg_path)], prefix=plotters_pkg.__name__ + "."
    ):
        short = mod_info.name.rsplit(".", 1)[-1]
        # Skip private helpers (``_comparison_helpers``) and sub-packages
        # themselves (e.g. ``generative``, ``mri_specific``).
        if short.startswith("_") or mod_info.ispkg:
            continue
        try:
            module = importlib.import_module(mod_info.name)
        except Exception as exc:
            discovered.append(
                (short, pytest.param(None, marks=pytest.mark.xfail(
                    reason=f"plotter module {mod_info.name} failed to import: {exc}",
                    strict=False,
                )))
            )
            continue
        discovered.append((short, module))
    return discovered


_PLOTTER_MODULES = _discover_plotter_modules()


# ---------------------------------------------------------------------------
# Synthetic fixtures: owned by the package catalog, aliased here
# ---------------------------------------------------------------------------

from spectramr.infrastructure.reporting.catalog import (  # noqa: E402
    examples_frames as _frames,
)
from spectramr.infrastructure.reporting.catalog import (  # noqa: E402
    examples_long as _long,
)
from spectramr.infrastructure.reporting.catalog import (  # noqa: E402
    examples_payloads as _payloads,
)
from spectramr.infrastructure.reporting.catalog.registry import (  # noqa: E402
    EXAMPLES as _REGISTRY_FIXTURES,
)

_make_long_metrics_df = _long.make_long_metrics_df
_make_predictions_df = _frames.make_predictions_df
_make_stratified_df = _frames.make_stratified_df
_make_arms = _payloads.make_arms
_make_image_cases = _payloads.make_image_cases
_make_cohort_dict = _payloads.make_cohort_dict
_make_metric_frame = _frames.make_metric_frame
_make_accel_frame = _frames.make_accel_frame
_make_calibration_frame = _frames.make_calibration_frame
_make_heatmap_frame = _frames.make_heatmap_frame
_make_bland_frame = _frames.make_bland_frame
_make_acquisition_frame = _frames.make_acquisition_frame
_make_bloch_frame = _frames.make_bloch_frame
_make_qmap_frame = _frames.make_qmap_frame
_make_qmap_slices = _payloads.make_qmap_slices
_make_beltrami_mu = _payloads.make_beltrami_mu
_make_fingerprint_embedding = _payloads.make_fingerprint_embedding
_make_task_ablation_frame = _frames.make_task_ablation_frame
_make_ksd_result = _payloads.make_ksd_result
_make_certificate_paths = _payloads.make_certificate_paths
_make_acquisition_arms = _payloads.make_acquisition_arms


# ---------------------------------------------------------------------------
# Plotter-specific dispatchers
# ---------------------------------------------------------------------------


# Plotters that need extra kwargs beyond the long metrics DataFrame.
# Each entry is ``(callable_attr, kwargs_factory)``. ``kwargs_factory`` is
# called lazily so per-test fixtures (e.g. cases lists) stay isolated.
_DISPATCH: dict[str, tuple[str, Any]] = {
    "headline_pareto": ("make", lambda: {"metric": "psnr", "cost": "params_m"}),
    "learning_curves": ("make", lambda: {"metrics": ["loss", "val_loss"]}),
    "loss_decomposition": ("make", lambda: {}),
    "residual_diagnostics": ("make", lambda: {"predictions_df": _make_predictions_df()}),
    "predicted_vs_true": ("make", lambda: {"predictions_df": _make_predictions_df()}),
    "ablation_strip": ("make", lambda: {"primary_metric": "psnr", "full_method": "full"}),
    "stratified_performance": ("make", lambda: {"stratified_df": _make_stratified_df()}),
    "failure_gallery": ("make", lambda: {"cases": _make_image_cases()}),
    "computational_profile": ("make", lambda: {}),
    # Generative.
    "gan_diagnostics": ("make", lambda: {}),
    "vae_diagnostics": ("make", lambda: {}),
    "diffusion_diagnostics": ("make", lambda: {}),
    # MRI-specific.
    "sr_triptych": ("make", lambda: {"cases": _make_image_cases(), "show_spectrum": True}),
    "synthesis_c2c": ("make", lambda: {"cases": _make_image_cases()}),
    "kspace_recon": ("make", lambda: {"cases": _make_image_cases()}),
    "cohort_table": ("make", lambda: {"cohort": _make_cohort_dict()}),
    "fiducial_check": ("make", lambda: {"cases": _make_image_cases()}),
}


# Modules whose ``make`` needs a frame OTHER than the long metrics df (or
# extra kwargs). ``(frame_factory, kwargs_factory)`` — both called lazily so
# per-test fixtures stay isolated. Keyed by module short name.
_MODULE_FIXTURES: dict[str, tuple[Any, Any]] = {
    "ablation_heatmap": (_make_heatmap_frame, dict),
    "bland_altman": (_make_bland_frame, lambda: {"metric": "ssim"}),
    "calibration_coverage": (_make_calibration_frame, dict),
    "forest_plot": (_make_metric_frame, lambda: {"metric": "psnr"}),
    "metric_distribution": (_make_metric_frame, dict),
    "significance_matrix": (_make_metric_frame, lambda: {"metric": "psnr"}),
    "acceleration_sweep": (_make_accel_frame, lambda: {"metric": "psnr"}),
    "metric_correlation": (_make_long_metrics_df, dict),
    "run_summary_card": (_make_long_metrics_df, dict),
    "train_val_gap": (_make_long_metrics_df, dict),
    "active_acquisition_trajectory": (_make_acquisition_frame, dict),
    "bloch_consistency_residual": (_make_bloch_frame, dict),
    "qmap_riemannian_vs_euclidean": (_make_qmap_frame, lambda: {"slices": _make_qmap_slices()}),
    "recon_panel": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "kspace_error_spectrum": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "group_strip": (_make_metric_frame, dict),
    "subject_mosaic": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "carpet_spike": (_make_long_metrics_df, lambda: {"cases": _make_image_cases()}),
    "task_ablation_bars": (_make_task_ablation_frame, dict),
}


def _invoke_plotter(module: Any, out_path: Path) -> Path | None:
    """Resolve the canonical entry point and call it with synthetic inputs."""
    short = module.__name__.rsplit(".", 1)[-1]

    if short in _DISPATCH:
        attr, kwargs_factory = _DISPATCH[short]
        fn = getattr(module, attr)
        return fn(_make_long_metrics_df(), out_path, **kwargs_factory())

    if short in _MODULE_FIXTURES:
        frame_factory, kwargs_factory = _MODULE_FIXTURES[short]
        return module.make(frame_factory(), out_path, **kwargs_factory())

    # Wrapper plotters with bespoke signatures.
    if short == "learnable_acquisition_pareto":
        fn = module.render_learnable_acquisition_pareto
        ArmResult = module.ArmResult  # noqa: N806 — holds a class, not a value
        arms = [
            ArmResult(name=f"arm_{i}", sampling_budget=1.0 / (i + 2),
                      psnr=28.0 + i, ssim=0.8 + 0.02 * i)
            for i in range(4)
        ]
        return fn(out_path, arms)

    if short == "certificate_summary":
        # Optional artefact paths default to None — the plotter should
        # still render a 2x3 grid with placeholder panels.
        fn = module.render_certificate_summary
        return fn(out_path)

    if short == "ksd_certificate":
        # Bespoke: takes a duck-typed KSDDefensibilityResult, not a df.
        class _KSDResult:
            passed, ksd_squared, p_value, threshold = True, 3.14e-4, 0.42, 0.05
            n_samples, bandwidth = 512, None
            message = "KSD GoF accepts H0."
        return module.render_ksd_certificate(out_path, result=_KSDResult())

    if short == "contact_sheet":
        # Composites sibling PNGs — populate the dir, then build the index.
        siblings = importlib.import_module(
            "spectramr.infrastructure.reporting.plotters"
        )
        siblings.dispatch(
            _make_long_metrics_df(),
            ["fig_1_2_learning_curves", "fig_1_15_computational_profile"],
            out_path.parent, formats=("png",), dpi=80, metrics=["loss", "val_loss"],
        )
        return module.make(_make_long_metrics_df(), out_path,
                           figure_dir=out_path.parent, formats=("png",))

    if short == "sfc_conformal_plots":
        # Multi-function module (make_beltrami_field / make_spd_geodesic / …);
        # the registry-coverage test exercises all four fig-ids. Here we just
        # prove the module renders by driving one representative entry point.
        return module.make_beltrami_field(None, out_path, mu=_make_beltrami_mu())

    pytest.xfail(f"no smoke fixture wired for plotter '{short}'")
    return None  # unreachable; xfail above raises


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test to bound memory use."""
    yield
    plt.close("all")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "plotter_smoke"
    d.mkdir()
    return d


def _ids(item: tuple[str, Any]) -> str:
    return item[0]


@pytest.mark.parametrize(
    ("short", "module"),
    _PLOTTER_MODULES,
    ids=[short for short, _ in _PLOTTER_MODULES],
)
def test_plotter_produces_nonempty_figure(short: str, module: Any, out_dir: Path) -> None:
    """Every plotter writes a non-empty figure file with synthetic inputs.

    The Phase-1 plotter family always returns ``Path | None`` and saves
    the figure to ``out_path`` inside ``make``; "non-empty Matplotlib
    Figure" therefore translates to "non-empty file on disk at the
    returned path".
    """
    if module is None:
        # Discovery-time import failure — already xfailed.
        pytest.xfail(f"plotter '{short}' could not be imported")

    out_path = out_dir / f"{short}.png"

    result = _invoke_plotter(module, out_path)

    assert result is not None, (
        f"plotter '{short}' returned None — synthetic fixture did not match "
        f"its expected schema"
    )
    result_path = Path(result)
    assert result_path.exists(), f"plotter '{short}' did not write {result_path}"
    assert result_path.stat().st_size > 0, (
        f"plotter '{short}' produced an empty file at {result_path}"
    )


def test_plotter_registry_lists_all_phase1_ids() -> None:
    """The ``PLOTTERS`` registry should expose at least one fig-ID per
    discovered Phase-1 module.

    Acts as a regression test against silent de-registration: if a new
    plotter module is added but ``register("fig_id", module.make)`` is
    forgotten in ``plotters/__init__.py``, the dispatcher won't see it.
    """
    plotters_module = importlib.import_module("spectramr.infrastructure.reporting.plotters")
    registered = set(plotters_module.list_available())
    assert registered, "PLOTTERS registry is empty — expected at least one entry"

    # Spot-check a few canonical IDs that should always be present.
    expected_present = {
        "fig_1_1_headline_pareto",
        "fig_1_2_learning_curves",
        "fig_1_4_residual_diagnostics",
    }
    missing = expected_present - registered
    assert not missing, f"expected fig-IDs missing from registry: {sorted(missing)}"


def test_plotter_dispatch_runs_against_synthetic_df(out_dir: Path) -> None:
    """End-to-end smoke for the public ``dispatch`` entry point.

    Drives the plotters that consume the long metrics DataFrame through
    the registry. Plotters that need richer inputs (``predictions_df``,
    ``cases``, ``cohort``) are excluded from this batch — they are
    exercised individually by ``test_plotter_produces_nonempty_figure``.
    """
    plotters_module = importlib.import_module("spectramr.infrastructure.reporting.plotters")
    df_long = _make_long_metrics_df()

    df_driven_ids = [
        "fig_1_1_headline_pareto",
        "fig_1_2_learning_curves",
        "fig_1_3_loss_decomposition",
        "fig_1_9_ablation_strip",
        "fig_1_15_computational_profile",
        "gen_gan_diagnostics",
        "gen_vae_diagnostics",
        "gen_diffusion_diagnostics",
    ]
    # Intersect with the actually-registered set so the test is forward-
    # compatible with registry pruning.
    available = [fid for fid in df_driven_ids if fid in set(plotters_module.list_available())]
    assert available, "no df-driven plotter ids available — registry empty?"

    results = plotters_module.dispatch(df_long, available, out_dir)

    assert set(results.keys()) == set(available)
    # At least one plotter must produce a non-None artefact — proves
    # the dispatch path is wired up end-to-end.
    produced = [v for v in results.values() if v is not None]
    assert produced, (
        "dispatch() returned None for every fig_id; plotters did not see "
        "the synthetic DataFrame as parseable"
    )
    for path in produced:
        assert Path(path).exists() and Path(path).stat().st_size > 0


def test_registry_fixture_map_covers_every_registered_id() -> None:
    """``_REGISTRY_FIXTURES`` must map EVERY registered plotter id.

    Guards against the split that motivated this test: a plotter can be
    registered (and shipped in a preset) yet have no fixture proving it
    fires, so nobody notices when it silently renders nothing. If a new
    plotter is registered without a fixture here, this fails loudly.
    """
    plotters_module = importlib.import_module("spectramr.infrastructure.reporting.plotters")
    registered = set(plotters_module.list_available())
    mapped = set(_REGISTRY_FIXTURES)
    missing = registered - mapped
    assert not missing, (
        f"registered plotters with no coverage fixture: {sorted(missing)} — "
        f"add each to _REGISTRY_FIXTURES so the anti-facade guard exercises it"
    )


@pytest.mark.parametrize("fig_id", sorted(_REGISTRY_FIXTURES))
def test_every_registered_plotter_renders(fig_id: str, out_dir: Path) -> None:
    """Every registered fig-id renders a non-empty artifact on good input.

    This is the reporting-layer anti-facade guard (CLAUDE.md pitfall #16):
    a figure that is registered but never fires — because a fixture key was
    dropped, a metric renamed, or a schema drifted — is a facade. The
    earlier module-level sweep xfailed ~18 of these; this closes that blind
    spot by driving the *registry* (``list_available``) directly.

    ``contact_sheet`` is the one figure that composites *other* PNGs, so it
    is rendered after a batch of sibling figures populate its ``figure_dir``.
    """
    plotters_module = importlib.import_module("spectramr.infrastructure.reporting.plotters")
    if fig_id not in set(plotters_module.list_available()):
        pytest.skip(f"{fig_id} not registered on this branch")

    frame_factory, kwargs_factory = _REGISTRY_FIXTURES[fig_id]
    df = frame_factory()
    kwargs = kwargs_factory()

    if fig_id == "contact_sheet":
        # Populate the dir with a couple of real figures first, then composite.
        plotters_module.dispatch(
            _make_long_metrics_df(),
            ["fig_1_2_learning_curves", "fig_1_15_computational_profile"],
            out_dir, formats=("png",), dpi=80,
            metrics=["loss", "val_loss"],
        )
        kwargs = {"figure_dir": out_dir}

    results = plotters_module.dispatch(
        df, [fig_id], out_dir, formats=("png",), dpi=80, **kwargs
    )
    produced = results.get(fig_id)
    assert produced is not None, (
        f"registered plotter '{fig_id}' rendered nothing on its coverage "
        f"fixture — a reporting-layer facade (pitfall #16)"
    )
    assert Path(produced).exists() and Path(produced).stat().st_size > 0
