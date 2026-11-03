"""Unit tests for infrastructure/reporting plotters and tables.

Strategy: every plotter callable is tested with the highest-ROI pattern —
a minimal synthetic fixture → assert non-empty file or non-None return.
Plotters that receive invalid/empty input must return None, not raise.

All tests use matplotlib Agg backend (no display required) and write to
pytest's tmp_path.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_long_df(
    methods: list[str] = ("methodA", "methodB"),
    metrics: list[str] = ("psnr", "ssim", "loss"),
    n_steps: int = 10,
    split: str = "train",
) -> pd.DataFrame:
    """Build a minimal long-format DataFrame consumed by most plotters."""
    rows = []
    for method in methods:
        for metric in metrics:
            for step in range(n_steps):
                rows.append({
                    "method": method,
                    "metric": metric,
                    "step": float(step),
                    "value": float(np.random.default_rng(42).uniform(0.5, 1.0)),
                    "split": split,
                    "seed": 0,
                })
    return pd.DataFrame(rows)


def _make_predictions_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    target = rng.uniform(0, 1, n)
    predicted = target + rng.normal(0, 0.05, n)
    return pd.DataFrame({"predicted": predicted, "target": target})


# ---------------------------------------------------------------------------
# headline_pareto (Fig 1.1)
# ---------------------------------------------------------------------------

class TestHeadlinePareto:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import headline_pareto

        df = _make_long_df(methods=["A", "B", "C"])
        # Add cost metric
        for i, method in enumerate(["A", "B", "C"]):
            df.loc[len(df)] = {
                "method": method, "metric": "params_m",
                "step": 0.0, "value": float(i + 1), "split": "test", "seed": 0,
            }
        out = headline_pareto.make(df, tmp_path / "pareto.pdf")
        assert out is not None
        assert out.is_file()
        assert out.stat().st_size > 100

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import headline_pareto

        out = headline_pareto.make(pd.DataFrame(), tmp_path / "pareto_empty.pdf")
        assert out is None

    @pytest.mark.unit
    def test_missing_metric_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import headline_pareto

        # df has psnr but not the cost column
        df = _make_long_df()
        out = headline_pareto.make(df, tmp_path / "pareto_no_cost.pdf",
                                   metric="psnr", cost="params_m")
        assert out is None


# ---------------------------------------------------------------------------
# learning_curves (Fig 1.2)
# ---------------------------------------------------------------------------

class TestLearningCurves:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import learning_curves

        df = _make_long_df(metrics=["loss", "val_loss"])
        out = learning_curves.make(df, tmp_path / "lc.pdf")
        assert out is not None
        assert out.is_file()
        assert out.stat().st_size > 100

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import learning_curves

        out = learning_curves.make(pd.DataFrame(), tmp_path / "lc_empty.pdf")
        assert out is None

    @pytest.mark.unit
    def test_no_metric_column_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import learning_curves

        out = learning_curves.make(pd.DataFrame({"x": [1, 2]}), tmp_path / "lc_no_col.pdf")
        assert out is None

    @pytest.mark.unit
    def test_explicit_metrics_list(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import learning_curves

        df = _make_long_df(metrics=["loss", "psnr", "ssim"])
        out = learning_curves.make(df, tmp_path / "lc_explicit.pdf", metrics=["loss"])
        assert out is not None

    @pytest.mark.unit
    def test_log_y_false(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import learning_curves

        df = _make_long_df(metrics=["loss"])
        out = learning_curves.make(df, tmp_path / "lc_linear.pdf",
                                   metrics=["loss"], log_y=False)
        assert out is not None


# ---------------------------------------------------------------------------
# loss_decomposition (Fig 1.3)
# ---------------------------------------------------------------------------

class TestLossDecomposition:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import loss_decomposition

        df = _make_long_df(
            metrics=["g_loss", "d_loss", "perceptual_loss", "loss"],
            split="train",
        )
        out = loss_decomposition.make(df, tmp_path / "ld.pdf")
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import loss_decomposition

        out = loss_decomposition.make(pd.DataFrame(), tmp_path / "ld_empty.pdf")
        assert out is None

    @pytest.mark.unit
    def test_no_component_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import loss_decomposition

        # Only has 'psnr' — no "loss" substring
        df = _make_long_df(metrics=["psnr", "ssim"])
        out = loss_decomposition.make(df, tmp_path / "ld_no_loss.pdf")
        assert out is None


# ---------------------------------------------------------------------------
# residual_diagnostics (Fig 1.4)
# ---------------------------------------------------------------------------

class TestResidualDiagnostics:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import residual_diagnostics

        pred_df = _make_predictions_df(50)
        out = residual_diagnostics.make(
            pd.DataFrame(),
            tmp_path / "rd.pdf",
            predictions_df=pred_df,
        )
        assert out is not None
        assert out.is_file()
        assert out.stat().st_size > 100

    @pytest.mark.unit
    def test_none_predictions_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import residual_diagnostics

        out = residual_diagnostics.make(pd.DataFrame(), tmp_path / "rd_none.pdf")
        assert out is None

    @pytest.mark.unit
    def test_empty_predictions_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import residual_diagnostics

        out = residual_diagnostics.make(
            pd.DataFrame(), tmp_path / "rd_empty.pdf",
            predictions_df=pd.DataFrame(),
        )
        assert out is None

    @pytest.mark.unit
    def test_missing_columns_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import residual_diagnostics

        bad_df = pd.DataFrame({"wrong_col": [1.0, 2.0]})
        out = residual_diagnostics.make(
            pd.DataFrame(), tmp_path / "rd_bad.pdf",
            predictions_df=bad_df,
        )
        assert out is None


# ---------------------------------------------------------------------------
# predicted_vs_true (Fig 1.5)
# ---------------------------------------------------------------------------

class TestPredictedVsTrue:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import predicted_vs_true

        pred_df = _make_predictions_df(60)
        out = predicted_vs_true.make(
            pd.DataFrame(), tmp_path / "pvt.pdf",
            predictions_df=pred_df,
        )
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_none_predictions_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import predicted_vs_true

        out = predicted_vs_true.make(pd.DataFrame(), tmp_path / "pvt_none.pdf")
        assert out is None

    @pytest.mark.unit
    def test_empty_predictions_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import predicted_vs_true

        out = predicted_vs_true.make(
            pd.DataFrame(), tmp_path / "pvt_empty.pdf",
            predictions_df=pd.DataFrame(),
        )
        assert out is None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# stratified_performance (Fig 1.11)
# ---------------------------------------------------------------------------

class TestStratifiedPerformance:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import stratified_performance

        rng = np.random.default_rng(7)
        strat_df = pd.DataFrame({
            "method": ["A"] * 30 + ["B"] * 30,
            "subgroup": (["low"] * 10 + ["mid"] * 10 + ["high"] * 10) * 2,
            "metric": ["psnr"] * 60,
            "value": rng.uniform(25, 40, 60),
        })
        out = stratified_performance.make(
            pd.DataFrame(), tmp_path / "strat.pdf",
            stratified_df=strat_df, metric="psnr",
        )
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_none_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import stratified_performance

        out = stratified_performance.make(pd.DataFrame(), tmp_path / "strat_none.pdf")
        assert out is None


# ---------------------------------------------------------------------------
# failure_gallery (Fig 1.12)
# ---------------------------------------------------------------------------

class TestFailureGallery:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import failure_gallery

        rng = np.random.default_rng(3)
        cases = [
            {
                "name": f"case_{i}",
                "predicted": rng.uniform(0, 1, (32, 32)).astype(np.float32),
                "target": rng.uniform(0, 1, (32, 32)).astype(np.float32),
                "residual": float(rng.uniform(0.1, 0.5)),
            }
            for i in range(4)
        ]
        out = failure_gallery.make(pd.DataFrame(), tmp_path / "fg.pdf", cases=cases)
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_cases_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import failure_gallery

        out = failure_gallery.make(pd.DataFrame(), tmp_path / "fg_empty.pdf", cases=[])
        assert out is None

    @pytest.mark.unit
    def test_none_cases_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import failure_gallery

        out = failure_gallery.make(pd.DataFrame(), tmp_path / "fg_none.pdf")
        assert out is None


# ---------------------------------------------------------------------------
# computational_profile (Fig 1.15)
# ---------------------------------------------------------------------------

class TestComputationalProfile:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import computational_profile

        df = _make_long_df(
            methods=["methodA"],
            metrics=["data_load_s", "forward_s", "backward_s", "gpu_mem_gb"],
            n_steps=5,
        )
        out = computational_profile.make(df, tmp_path / "cp.pdf")
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import computational_profile

        out = computational_profile.make(pd.DataFrame(), tmp_path / "cp_empty.pdf")
        assert out is None


# ---------------------------------------------------------------------------
# Generative diagnostics
# ---------------------------------------------------------------------------

class TestGANDiagnostics:

    @pytest.mark.unit
    def test_produces_file_with_d_acc(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import gan_diagnostics

        df = _make_long_df(
            methods=["gan"],
            metrics=["d_acc", "g_loss", "d_loss"],
            n_steps=10,
        )
        out = gan_diagnostics.make(df, tmp_path / "gan.pdf")
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import gan_diagnostics

        out = gan_diagnostics.make(pd.DataFrame(), tmp_path / "gan_empty.pdf")
        assert out is None

    @pytest.mark.unit
    def test_no_relevant_metrics_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import gan_diagnostics

        df = _make_long_df(metrics=["psnr", "ssim"])  # no GAN-specific columns
        out = gan_diagnostics.make(df, tmp_path / "gan_no_metrics.pdf")
        assert out is None


class TestVAEDiagnostics:

    @pytest.mark.unit
    def test_produces_file_with_rec_kl(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import vae_diagnostics

        df = _make_long_df(
            methods=["vae"],
            metrics=["reconstruction_loss", "kl_loss"],
            n_steps=10,
        )
        out = vae_diagnostics.make(df, tmp_path / "vae.pdf")
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import vae_diagnostics

        out = vae_diagnostics.make(pd.DataFrame(), tmp_path / "vae_empty.pdf")
        assert out is None

    @pytest.mark.unit
    def test_no_vae_metrics_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import vae_diagnostics

        df = _make_long_df(metrics=["psnr"])
        out = vae_diagnostics.make(df, tmp_path / "vae_no_metrics.pdf")
        assert out is None


class TestDiffusionDiagnostics:

    @pytest.mark.unit
    def test_produces_file_with_loss_per_timestep(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import diffusion_diagnostics

        df = _make_long_df(
            methods=["ddpm"],
            metrics=["loss_per_timestep"],
            n_steps=20,
        )
        out = diffusion_diagnostics.make(df, tmp_path / "diff.pdf")
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import diffusion_diagnostics

        out = diffusion_diagnostics.make(pd.DataFrame(), tmp_path / "diff_empty.pdf")
        assert out is None

    @pytest.mark.unit
    def test_no_diffusion_metrics_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.generative import diffusion_diagnostics

        df = _make_long_df(metrics=["psnr"])
        out = diffusion_diagnostics.make(df, tmp_path / "diff_no_m.pdf")
        assert out is None


# ---------------------------------------------------------------------------
# MRI-specific: sr_triptych
# ---------------------------------------------------------------------------

class TestSRTriptych:

    @pytest.mark.unit
    def test_produces_file(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.mri_specific import sr_triptych

        rng = np.random.default_rng(5)
        h, w = 32, 32
        cases = [
            {
                "lr": rng.uniform(0, 1, (h, w)).astype(np.float32),
                "pred": rng.uniform(0, 1, (h, w)).astype(np.float32),
                "hr": rng.uniform(0, 1, (h, w)).astype(np.float32),
            }
            for _ in range(2)
        ]
        out = sr_triptych.make(pd.DataFrame(), tmp_path / "sr.pdf", cases=cases)
        assert out is not None
        assert out.is_file()

    @pytest.mark.unit
    def test_empty_cases_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters.mri_specific import sr_triptych

        out = sr_triptych.make(pd.DataFrame(), tmp_path / "sr_empty.pdf", cases=[])
        assert out is None


# ---------------------------------------------------------------------------
# Registry: all registered plotters are callable
# ---------------------------------------------------------------------------

class TestPlotterRegistry:

    @pytest.mark.unit
    def test_registry_non_empty(self) -> None:
        from spectramr.infrastructure.reporting.plotters import PLOTTERS

        assert len(PLOTTERS) > 0

    @pytest.mark.unit
    def test_list_available(self) -> None:
        from spectramr.infrastructure.reporting.plotters import list_available

        keys = list_available()
        assert len(keys) > 0
        assert all(isinstance(k, str) for k in keys)

    @pytest.mark.unit
    def test_get_existing_key(self) -> None:
        from spectramr.infrastructure.reporting.plotters import get

        fn = get("fig_1_2_learning_curves")
        assert callable(fn)

    @pytest.mark.unit
    def test_get_missing_key_raises(self) -> None:
        from spectramr.infrastructure.reporting.plotters import get

        with pytest.raises(KeyError):
            get("nonexistent_plotter_xyz")

    @pytest.mark.unit
    def test_double_register_raises(self) -> None:
        from spectramr.infrastructure.reporting.plotters import PLOTTERS, register

        existing = next(iter(PLOTTERS.keys()))
        fn = PLOTTERS[existing]
        with pytest.raises(ValueError, match="already registered"):
            register(existing, fn)

    @pytest.mark.unit
    @pytest.mark.parametrize("fig_id", [
        "fig_1_1_headline_pareto",
        "fig_1_2_learning_curves",
        "fig_1_3_loss_decomposition",
        "fig_1_4_residual_diagnostics",
        "fig_1_5_predicted_vs_true",
        "fig_1_11_stratified_performance",
        "fig_1_12_failure_gallery",
        "fig_1_15_computational_profile",
        "gen_diffusion_diagnostics",
        "gen_gan_diagnostics",
        "gen_vae_diagnostics",
        "mri_a1_sr_triptych",
    ])
    def test_plotters_return_none_on_empty_input(self, fig_id: str, tmp_path: Path) -> None:
        """Every registered plotter must return None (not raise) on empty DataFrame."""
        from spectramr.infrastructure.reporting.plotters import get

        fn = get(fig_id)
        result = fn(pd.DataFrame(), tmp_path / f"{fig_id}.pdf")
        assert result is None, f"plotter {fig_id} should return None on empty input, got {result}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class TestAblationTable:

    @pytest.mark.unit
    def test_produces_markdown_and_latex(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.ablation_table import make

        rng = np.random.default_rng(99)
        rows = []
        for method in ["full", "no_perceptual", "no_adv"]:
            for _ in range(5):
                rows.append({
                    "method": method, "metric": "psnr",
                    "step": 0, "value": float(rng.uniform(25, 40)),
                    "split": "test", "seed": 0,
                })
        df = pd.DataFrame(rows)
        result = make(df, tmp_path / "ablation", primary_metric="psnr", full_method="full")
        assert result is not None
        assert result["markdown"].exists()
        assert result["latex"].exists()
        md_text = result["markdown"].read_text()
        assert "full" in md_text
        assert "psnr" in md_text

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.ablation_table import make

        result = make(pd.DataFrame(), tmp_path / "ablation_empty")
        assert result is None

    @pytest.mark.unit
    def test_missing_full_method_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.ablation_table import make

        rows = [{"method": "A", "metric": "psnr", "value": 30.0,
                 "step": 0, "split": "test", "seed": 0}]
        df = pd.DataFrame(rows)
        result = make(df, tmp_path / "ablation_no_full",
                      primary_metric="psnr", full_method="full")
        assert result is None


class TestMainResultsTable:

    @pytest.mark.unit
    def test_produces_markdown_and_latex(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.main_results import make

        rng = np.random.default_rng(88)
        rows = []
        for method in ["proposed", "baseline"]:
            for metric in ["psnr", "ssim"]:
                for seed in range(3):
                    rows.append({
                        "method": method, "metric": metric,
                        "step": 0, "split": "test",
                        "value": float(rng.uniform(0.5, 1.0)),
                        "seed": seed,
                    })
        df = pd.DataFrame(rows)
        result = make(df, tmp_path / "main", metrics=["psnr", "ssim"])
        assert result is not None
        assert result["markdown"].exists()
        assert result["latex"].exists()
        assert "proposed" in result["markdown"].read_text()

    @pytest.mark.unit
    def test_empty_df_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.main_results import make

        result = make(pd.DataFrame(), tmp_path / "main_empty")
        assert result is None


class TestHyperparameterTable:

    @pytest.mark.unit
    def test_produces_markdown_and_latex(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.hyperparameter_table import make

        hps = [
            {"name": "lr", "distribution": "log-uniform",
             "range": "[1e-5, 1e-2]", "final": "1e-4", "criterion": "val PSNR"},
            {"name": "batch_size", "distribution": "categorical",
             "range": "{8,16,32}", "final": "16", "criterion": "val PSNR"},
        ]
        result = make(None, tmp_path / "hp", hyperparameters=hps)
        assert result is not None
        assert result["markdown"].exists()
        assert "lr" in result["markdown"].read_text()

    @pytest.mark.unit
    def test_none_hyperparameters_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.hyperparameter_table import make

        result = make(None, tmp_path / "hp_none")
        assert result is None

    @pytest.mark.unit
    def test_empty_list_returns_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.tables.hyperparameter_table import make

        result = make(None, tmp_path / "hp_empty", hyperparameters=[])
        assert result is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestRunMetadata:

    @pytest.mark.unit
    def test_canary_construct(self) -> None:
        from spectramr.infrastructure.reporting.metadata import RunMetadata

        meta = RunMetadata(git_commit="abc1234", seed=42)
        assert meta.git_commit == "abc1234"
        assert meta.seed == 42

    @pytest.mark.unit
    def test_to_dict_keys(self) -> None:
        from spectramr.infrastructure.reporting.metadata import RunMetadata

        meta = RunMetadata(seed=7)
        d = meta.to_dict()
        for key in ("git_commit", "seed", "timestamp_utc"):
            assert key in d

    @pytest.mark.unit
    def test_short_label_contains_commit(self) -> None:
        from spectramr.infrastructure.reporting.metadata import RunMetadata

        meta = RunMetadata(git_commit="abcdef123456", seed=0)
        label = meta.short_label()
        assert "abcdef1" in label

    @pytest.mark.unit
    def test_stamp_figure_does_not_raise(self) -> None:
        import matplotlib.pyplot as plt

        from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure

        fig, _ = plt.subplots()
        meta = RunMetadata(git_commit="test1234", seed=1)
        stamp_figure(fig, meta)  # should not raise
        plt.close(fig)

    @pytest.mark.unit
    def test_write_sidecar(self, tmp_path: Path) -> None:
        import json

        from spectramr.infrastructure.reporting.metadata import RunMetadata, write_sidecar

        meta = RunMetadata(git_commit="cafebabe", seed=99)
        artifact = tmp_path / "figure.pdf"
        artifact.touch()
        sidecar = write_sidecar(artifact, meta)
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["git_commit"] == "cafebabe"
        assert data["seed"] == 99


# ---------------------------------------------------------------------------
# dispatch() smoke test
# ---------------------------------------------------------------------------

class TestDispatch:

    @pytest.mark.unit
    def test_dispatch_empty_df_all_none(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import dispatch

        fig_ids = ["fig_1_2_learning_curves", "fig_1_4_residual_diagnostics"]
        results = dispatch(pd.DataFrame(), fig_ids, tmp_path / "dispatch")
        for fid in fig_ids:
            assert fid in results
            assert results[fid] is None

    @pytest.mark.unit
    def test_dispatch_unknown_id_handled_gracefully(self, tmp_path: Path) -> None:
        from spectramr.infrastructure.reporting.plotters import dispatch

        results = dispatch(pd.DataFrame(), ["nonexistent_fig_xyz"], tmp_path)
        assert results["nonexistent_fig_xyz"] is None
