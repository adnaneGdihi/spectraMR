"""Campaign Evaluator — Cross-experiment comparative evaluation.

After all experiments complete, this module:
  1. Runs inference on shared test set (per_sample_inference)
  2. Computes per-sample metrics (PSNR, SSIM, LPIPS per slice)
  3. Runs pairwise statistical tests (variants vs. baseline)
  4. Computes bootstrap CIs + effect sizes
  5. Applies multiple-comparison correction
  6. Evaluates uncertainty calibration (compute_uncertainty)
  7. Builds leaderboard and significance matrices
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from spectramr.config.schemas.campaign import EvaluationConfigSchema
from spectramr.core.metrics.statistical_tests import StatisticalTests
from spectramr.core.metrics.uncertainty import UncertaintyMetrics
from spectramr.data.builders.eval_dataset_builder import (
    load_clean_reference,
    load_paired_samples,
    load_per_sample_metrics,
    load_uncertainty_bundle,
)
from spectramr.infrastructure.checkpointing.restore import weights
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)
from spectramr.infrastructure.reporting.per_slice import (
    SlicePass,
    early_stopping_iteration,
    find_rows_files,
    read_pass,
    run_root,
    slice_values,
    subject_means,
    summarise,
)
from spectramr.infrastructure.reporting.per_subject_stats import (
    PairedSubjects,
    compare_subjects,
    pair_on_slices,
)
from spectramr.infrastructure.validation.inference_settings import resolve_inference_settings

logger = logging.getLogger(__name__)
__all__ = [
    "CampaignEvaluator",
    "CampaignReport",
    "PairwiseResult",
    "UncertaintyReport",
]


@dataclass
class PairwiseResult:
    """Result of pairwise comparison between two experiments."""

    experiment_a: str
    experiment_b: str
    metric: str
    paired_ttest: dict[str, float] = field(default_factory=dict)
    wilcoxon: dict[str, float] = field(default_factory=dict)
    effect_size: dict[str, Any] = field(default_factory=dict)
    bootstrap_ci: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def p_value(self) -> float:
        return self.paired_ttest.get("p_value", float("nan"))

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class UncertaintyReport:
    """Per-experiment uncertainty calibration results."""

    experiment_name: str
    ece: float | None = None
    sharpness: float | None = None
    uncertainty_error_correlation: float | None = None


@dataclass
class CampaignReport:
    """Aggregated evaluation results for a campaign."""

    campaign_name: str
    leaderboard: pd.DataFrame | None = None
    pairwise_results: list[PairwiseResult] = field(default_factory=list)
    significance_matrix: dict[str, pd.DataFrame] = field(default_factory=dict)
    corrected_p_values: dict[str, dict] = field(default_factory=dict)
    uncertainty_reports: list[UncertaintyReport] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "campaign_name": self.campaign_name,
            "summary": self.summary,
            "corrected_p_values": self.corrected_p_values,
            "pairwise_results": [
                {
                    "experiment_a": pr.experiment_a,
                    "experiment_b": pr.experiment_b,
                    "metric": pr.metric,
                    "paired_ttest": pr.paired_ttest,
                    "wilcoxon": pr.wilcoxon,
                    "effect_size": pr.effect_size,
                    "bootstrap_ci": pr.bootstrap_ci,
                    "provenance": pr.provenance,
                }
                for pr in self.pairwise_results
            ],
            "uncertainty": [
                {
                    "experiment": ur.experiment_name,
                    "ece": ur.ece,
                    "sharpness": ur.sharpness,
                    "uncertainty_error_correlation": ur.uncertainty_error_correlation,
                }
                for ur in self.uncertainty_reports
            ],
        }
        if self.leaderboard is not None:
            result["leaderboard"] = self.leaderboard.to_dict("records")
        for metric_name, sig_df in self.significance_matrix.items():
            result[f"significance_matrix_{metric_name}"] = sig_df.to_dict()
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Campaign report saved to {path}")


class CampaignEvaluator:
    """Cross-experiment comparative evaluation engine."""

    def __init__(self, eval_config: EvaluationConfigSchema | None = None) -> None:
        self.config = eval_config or EvaluationConfigSchema()
        # Arms read from per-slice rows, and the column each metric resolved to.
        self._slice_passes: dict[str, SlicePass] = {}
        self._row_columns: dict[tuple[str, str], str] = {}
        self._nonfinite: dict[tuple[str, str], int] = {}

    # ── Main entry point ─────────────────────────────────────────

    def evaluate(self, campaign_state: CampaignState) -> CampaignReport:
        """Run full evaluation on a completed campaign.

        Respects all EvaluationConfigSchema flags:
        - per_sample_inference: run inference on shared test set
        - compute_uncertainty: evaluate calibration metrics
        - generate_plots / export_latex: consumed by CLI/report generator
        """
        report = CampaignReport(campaign_name=campaign_state.campaign_name)

        # Step 1: Run inference if configured and test_manifest available
        if self.config.per_sample_inference:
            self._run_inference_all(campaign_state)

        # Step 2: Load per-sample metrics
        all_metrics = self._load_all_metrics(campaign_state)

        if len(all_metrics) < 2:
            report.summary = (
                f"Insufficient evaluable experiments ({len(all_metrics)}). "
                "Need at least 2 for comparison."
            )
            return report

        # Step 3: Leaderboard
        report.leaderboard = self._build_leaderboard(all_metrics)

        # Step 4: Pairwise comparisons
        baseline_names = [e.name for e in campaign_state.experiments if e.role == "baseline"]
        if not baseline_names:
            baseline_names = [next(iter(all_metrics.keys()))]

        for bl in baseline_names:
            if bl not in all_metrics:
                continue
            for metric in self.config.metrics:
                report.pairwise_results.extend(self._pairwise_vs_baseline(bl, all_metrics, metric))

        # Step 5: Multiple comparison correction
        if report.pairwise_results:
            report.corrected_p_values = self._apply_correction(report.pairwise_results)

        # Step 6: Significance matrices
        for metric in self.config.metrics:
            sig = self._build_significance_matrix(all_metrics, metric)
            if sig is not None:
                report.significance_matrix[metric] = sig

        # Step 7: Uncertainty calibration
        if self.config.compute_uncertainty:
            report.uncertainty_reports = self._evaluate_uncertainty(campaign_state)

        # Step 8: Summary
        report.summary = self._generate_summary(report, all_metrics)
        return report

    # ── Inference Runner ─────────────────────────────────────────

    def _run_inference_all(self, state: CampaignState) -> None:
        """Run inference on shared test set for all completed experiments.

        Loads each experiment's checkpoint, runs forward passes on the
        test manifest, and saves per-sample metrics as NPZ files.

        Uses the campaign's test_manifest (stored on state) and each
        experiment's checkpoint_path.
        """
        test_manifest = getattr(state, "test_manifest", None)
        if not test_manifest:
            logger.info(
                "No test_manifest on campaign state — skipping per-sample inference. "
                "Will use existing CSV/NPZ metrics."
            )
            return

        manifest_path = Path(test_manifest)
        if not manifest_path.exists():
            logger.warning(f"Test manifest not found: {manifest_path}")
            return

        for exp in state.evaluable:
            # Skip if per-sample metrics already exist
            if exp.per_sample_metrics_path:
                existing = Path(exp.per_sample_metrics_path)
                if existing.exists():
                    logger.debug(f"Per-sample metrics already exist for {exp.name}, skipping")
                    continue

            if not exp.checkpoint_path or not Path(exp.checkpoint_path).exists():
                logger.warning(f"No checkpoint for {exp.name}, skipping inference")
                continue

            try:
                per_sample = self._run_single_inference(
                    checkpoint_path=exp.checkpoint_path,
                    config_path=exp.config_path,
                    manifest_path=str(manifest_path),
                    output_dir=exp.output_dir or str(Path(state.campaign_dir) / exp.name),
                )
                if per_sample:
                    exp.per_sample_metrics_path = per_sample
                    logger.info(f"Generated per-sample metrics: {exp.name}")
            except Exception as e:
                logger.error(f"Inference failed for {exp.name}: {e}")

    def _run_single_inference(
        self,
        checkpoint_path: str,
        config_path: str,
        manifest_path: str,
        output_dir: str,
    ) -> str | None:
        """Run inference for a single experiment and compute per-sample metrics.

        Loads the model from checkpoint, processes each sample in the
        test manifest, and computes per-sample PSNR/SSIM/LPIPS.

        Returns path to saved NPZ file, or None on failure.
        """
        from spectramr.core.compute_device import resolve_torch_device
        from spectramr.core.metrics.evaluation_metrics import LPIPS, PSNR, SSIMMetric

        # Per-sample evaluation runs the model over the whole test manifest and
        # computes LPIPS (itself a network) — a heavy pipeline. The former
        # ``"cuda" if is_available() else "cpu"`` silently produced CPU metrics
        # that would then be tabulated alongside GPU-run arms.
        device = torch.device(
            resolve_torch_device("auto", pipeline="evaluate", source="campaign_evaluator").device
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "per_sample_metrics.npz"

        # Outside the try below: a checkpoint that cannot be loaded raises against this
        # arm's name instead of folding into the blanket `except -> None`.
        model = self._reconstruct_model_from_config(config_path, checkpoint_path, device)
        model = model.to(device).eval()

        try:
            psnr_fn = PSNR()
            ssim_fn = SSIMMetric()

            # LPIPS requires pretrained weights — guard import failure
            lpips_fn = None
            if "lpips" in self.config.metrics:
                try:
                    lpips_fn = LPIPS().to(device)
                except Exception as e:
                    logger.warning(f"LPIPS unavailable, skipping: {e}")

            # Load test data from manifest
            test_samples = self._load_test_samples(manifest_path)
            if not test_samples:
                logger.warning("No test samples loaded from manifest")
                return None

            psnr_values: list[float] = []
            ssim_values: list[float] = []
            lpips_values: list[float] = []

            with torch.no_grad():
                for i, (input_tensor, target_tensor) in enumerate(test_samples):
                    input_tensor = input_tensor.to(device)
                    target_tensor = target_tensor.to(device)

                    # Ensure batch dimension
                    if input_tensor.ndim == 3:
                        input_tensor = input_tensor.unsqueeze(0)
                    if target_tensor.ndim == 3:
                        target_tensor = target_tensor.unsqueeze(0)

                    output = model(input_tensor)

                    psnr_values.append(psnr_fn(output, target_tensor).item())
                    ssim_values.append(ssim_fn(output, target_tensor).item())
                    if lpips_fn is not None:
                        lpips_values.append(lpips_fn(output, target_tensor).item())

            # Save per-sample metrics
            save_dict = {
                "psnr": np.array(psnr_values),
                "ssim": np.array(ssim_values),
            }
            if lpips_values:
                save_dict["lpips"] = np.array(lpips_values)

            np.savez(npz_path, **save_dict)
            logger.info(f"Saved {len(psnr_values)} per-sample metrics to {npz_path}")
            return str(npz_path)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    @staticmethod
    def _reconstruct_model_from_config(
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
    ) -> torch.nn.Module:
        """Rebuild the arm's generator and load the checkpoint into it.

        Construction goes through :class:`ModelBuilder`, the same path training
        uses. It used to call ``ModelFactory.create_generator`` with nothing but
        ``model_kwargs``, which meant the evaluated model was built *differently
        from the model that produced the checkpoint*: no ``acceleration_config``
        from ``undersampling``, no ``kspace_log_scaled`` from
        ``data.processing.enable_log_scaling``, and no
        ``physics.data_consistency`` block. Comparing arms on a leaderboard
        requires that the evaluated architecture be the trained one.

        It also defaulted ``in_channels``/``out_channels`` to ``2`` behind
        ``getattr``. A defaulted read standing in for a declared value is the
        substitution CLAUDE.md forbids: the schema has real defaults, and an arm
        that declares something else was silently evaluated at the wrong width.

        Args:
            config_path: Path to the experiment training YAML; the run's
                ``resolved_config.json`` beside the checkpoint wins when present.
            checkpoint_path: The run's checkpoint; its ``generator`` part is loaded.
            device: Target device for the model.

        Returns:
            Loaded model in eval mode.

        Raises:
            ZeroOverlapError: If the checkpoint shares no parameter name with the
                model the config describes (a ``ValueError``).
        """
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        settings, _source = resolve_inference_settings(config_path, checkpoint_path)
        model = ModelBuilder(settings, device).build_generator().validate().build()["generator"]

        # overlap, as before: missing or extra names are logged, not fatal; zero shared
        # names and shape mismatches still raise.
        report = weights(checkpoint_path, "generator", model, policy="overlap")
        if report.missing:
            logger.warning(
                f"Missing keys when loading {settings.model.model_type}: "
                f"{len(report.missing)} keys (e.g., {list(report.missing[:3])})"
            )
        if report.unexpected:
            logger.debug(
                f"Unexpected keys when loading {settings.model.model_type}: "
                f"{len(report.unexpected)} keys"
            )

        model = model.to(device).eval()
        logger.info(
            f"Reconstructed {settings.model.model_type} from {config_path} "
            f"({sum(p.numel() for p in model.parameters()):,} params)"
        )
        return model

    @staticmethod
    def _load_test_samples(
        manifest_path: str,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Load test samples from a manifest file.

        Supports:
        - .pkl manifests (list of dicts with 'input'/'target' paths)
        - .txt manifests (line-separated file paths)
        - .npz pre-computed arrays

        Returns list of (input_tensor, target_tensor) tuples.
        """
        path = Path(manifest_path)
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        if path.suffix == ".pkl":
            import pickle

            with open(path, "rb") as f:
                manifest = pickle.load(f)

            if isinstance(manifest, list):
                for entry in manifest:
                    if isinstance(entry, dict):
                        input_path = entry.get("input") or entry.get("source")
                        target_path = entry.get("target") or entry.get("ground_truth")
                        if input_path and target_path:
                            inp = _load_medical_image(input_path)
                            tgt = _load_medical_image(target_path)
                            if inp is not None and tgt is not None:
                                samples.append((inp, tgt))

        elif path.suffix == ".npz":
            # SSOT (CLAUDE.md #11): route .npz paired-sample loading
            # through spectramr.data.builders.eval_dataset_builder rather than
            # calling np.load directly here.
            samples.extend(load_paired_samples(path))

        elif path.suffix == ".txt":
            logger.info(
                "Text manifests require paired lines (input\\ttarget). "
                "Skipping — use .pkl or .npz manifest for per-sample inference."
            )

        else:
            logger.warning(f"Unsupported manifest format: {path.suffix}")

        return samples

    # ── Uncertainty Evaluation ───────────────────────────────────

    def _evaluate_uncertainty(self, state: CampaignState) -> list[UncertaintyReport]:
        """Compute uncertainty calibration metrics for each experiment.

        Looks for uncertainty estimates saved alongside predictions
        (e.g., from MC dropout or ensemble variance). Falls back to
        using prediction variance if available.
        """
        reports: list[UncertaintyReport] = []

        for exp in state.evaluable:
            report = UncertaintyReport(experiment_name=exp.name)
            output_dir = Path(exp.output_dir) if exp.output_dir else None
            if output_dir is None:
                reports.append(report)
                continue

            # Look for uncertainty estimates
            uncertainty_path = output_dir / "uncertainty_estimates.npz"
            if not uncertainty_path.exists():
                # Try inference output directory
                infer_dir = output_dir / "inference_test_split"
                uncertainty_path = infer_dir / "uncertainty_estimates.npz"

            if not uncertainty_path.exists():
                logger.debug(f"No uncertainty estimates for {exp.name} — skipping calibration")
                reports.append(report)
                continue

            try:
                # SSOT (CLAUDE.md #11): route uncertainty .npz loading
                # through spectramr.data.builders.eval_dataset_builder.
                bundle = load_uncertainty_bundle(uncertainty_path)
                if bundle is None:
                    logger.debug(
                        "Uncertainty bundle missing required keys for "
                        f"{exp.name} — skipping calibration"
                    )
                    reports.append(report)
                    continue
                predictions, targets, uncertainties = bundle

                # Compute errors for correlation
                errors = (predictions - targets).abs()

                report.ece = UncertaintyMetrics.expected_calibration_error(
                    predictions, targets, uncertainties
                )
                report.sharpness = UncertaintyMetrics.sharpness(uncertainties)
                report.uncertainty_error_correlation = UncertaintyMetrics.uncertainty_correlation(
                    uncertainties, errors
                )

                logger.info(
                    f"Uncertainty for {exp.name}: "
                    f"ECE={report.ece:.4f}, "
                    f"sharpness={report.sharpness:.4f}, "
                    f"corr={report.uncertainty_error_correlation:.4f}"
                )

            except Exception as e:
                logger.warning(f"Uncertainty evaluation failed for {exp.name}: {e}")

            reports.append(report)

        return reports

    # ── Metric Loading ───────────────────────────────────────────

    def _load_all_metrics(self, state: CampaignState) -> dict[str, dict[str, np.ndarray]]:
        """Per-sample (NPZ) or per-subject (per-slice rows) values for every arm.

        Raises:
            ValueError: an arm has only pass rows, or the arms mix NPZ and rows.
        """
        self._slice_passes.clear()
        self._row_columns.clear()
        self._nonfinite.clear()
        result: dict[str, dict[str, np.ndarray]] = {}
        pass_rows_only: list[str] = []
        for exp in state.evaluable:
            m = self._load_experiment_metrics(exp)
            if m:
                result[exp.name] = m
            elif exp.name in self._slice_passes:
                continue
            elif exp.metrics_csv_path and Path(exp.metrics_csv_path).exists():
                pass_rows_only.append(exp.name)
        if pass_rows_only:
            raise ValueError(
                f"{pass_rows_only} have only validation_metrics.csv, one row per "
                "validation pass; passes are not samples, so no leaderboard spread or "
                "paired test can be computed from them (#2478). Enable "
                "validation.per_batch_rows with validation.loader.batch_size: 1, or "
                "evaluation.per_sample_inference."
            )
        from_rows = {n for n in result if n in self._slice_passes}
        if from_rows and from_rows != set(result):
            raise ValueError(
                f"{sorted(from_rows)} are read from per-slice validation rows and "
                f"{sorted(set(result) - from_rows)} from per-sample NPZ; the two score "
                "different sets and cannot be paired (#2478)."
            )
        logger.info(f"Loaded metrics for {len(result)} experiments")
        return result

    @staticmethod
    def _resolve_metric_column(columns: list[str], metric_name: str) -> tuple[str | None, str]:
        """Pick the CSV column that holds ``metric_name``, or refuse to guess.

        This used to be ``[c for c in df.columns if mn in c.lower()]`` followed by
        ``cols[0]`` -- an unanchored substring match resolved by DataFrame column
        order. On a CSV carrying ``train_ssim``, ``val_ms_ssim`` and ``val_ssim``,
        the metric ``ssim`` resolved to **train_ssim**: the training split, and not
        even the right metric. Reorder the columns and it resolves to
        ``val_ms_ssim`` instead. That matters because ``ssim`` is the SELECTION
        metric across the ldm_two_stage_ulf_to_hf cohort (chosen over PSNR
        deliberately, since PSNR is maximised by the posterior mean and rewards the
        blurriest checkpoint), so the leaderboard ranked arms on a quantity they
        were not selected on. Silent by construction: a wrong-but-present column
        yields a plausible number (#1062).

        Resolution order, deterministic and independent of column order:

        1. exact ``val_<metric>`` -- both conventions exist in this repo, because
           some strategies emit ``val_``-prefixed keys while ``MetricsTracker``
           writes metric keys verbatim into a validation-specific FILE. The same
           two-step preference is already used by
           ``reporting/cohort_ablation.py`` (``for key in (f"val_{metric}",
           metric)``); this mirrors it rather than inventing a third rule.
        2. exact ``<metric>``
        3. a UNIQUE substring match, excluding ``train_*`` -- keeps decorated
           columns such as ``val_psnr_4x`` reachable, which an exact-only rule
           would have silently dropped from every existing campaign.

        Ambiguity is never resolved by picking one. Per the issue's own principle
        -- a wrong metric that looks right is worse than a missing column -- more
        than one surviving candidate returns ``None`` with a reason, and the caller
        warns and omits the metric.

        ``train_``-prefixed columns are never eligible: campaign comparison is on
        held-out data by definition.

        Args:
            columns: CSV column names, in file order (order must not matter).
            metric_name: Declared ``evaluation.metrics`` entry.

        Returns:
            ``(column_or_None, reason)``. ``reason`` is always human-readable and
            is what the caller logs, so a skipped metric explains itself.
        """
        m = metric_name.lower()
        # First spelling wins for a case-collision; sorted() keeps that deterministic.
        by_lower: dict[str, str] = {}
        for c in sorted(columns):
            by_lower.setdefault(c.lower(), c)

        if f"val_{m}" in by_lower:
            return by_lower[f"val_{m}"], f"exact 'val_{m}'"
        if m in by_lower:
            return by_lower[m], f"exact '{m}'"

        train_only = sorted(c for c in columns if m in c.lower() and c.lower().startswith("train_"))
        candidates = sorted(
            c for c in columns if m in c.lower() and not c.lower().startswith("train_")
        )

        if len(candidates) == 1:
            return candidates[0], f"unique substring match '{candidates[0]}'"
        if len(candidates) > 1:
            return None, (
                f"ambiguous: {candidates} all contain {metric_name!r}. Refusing to "
                "guess -- declare the exact column name as the metric instead."
            )
        if train_only:
            return None, (
                f"only TRAINING columns matched ({train_only}); campaign metrics "
                "are held-out by definition, so this is not usable."
            )
        return None, f"no column matches {metric_name!r}"

    def _load_experiment_metrics(self, exp: ExperimentStatus) -> dict[str, np.ndarray]:
        metrics: dict[str, np.ndarray] = {}

        # Priority 1: per-sample NPZ
        # SSOT (CLAUDE.md #11): route .npz metric loading through
        # spectramr.data.builders.eval_dataset_builder.
        if exp.per_sample_metrics_path:
            try:
                loaded = load_per_sample_metrics(exp.per_sample_metrics_path)
                if loaded is not None:
                    metrics.update(loaded)
                    return metrics
            except Exception as e:
                logger.warning(f"Failed to load NPZ {exp.per_sample_metrics_path}: {e}")

        # Priority 2: per-slice validation rows, reduced to subjects. The
        # one-row-per-pass validation CSV is never a sample (#2478).
        rows = self._slice_pass(exp)
        if rows is None:
            return metrics
        self._slice_passes[exp.name] = rows
        for mn in self.config.metrics:
            col, reason = self._resolve_metric_column(rows.metric_columns, mn)
            if col is None:
                logger.warning(
                    "Campaign metric %r not loaded from %s -- %s", mn, rows.files[0], reason
                )
                continue
            values, dropped = slice_values(rows, col)
            if dropped:
                logger.warning(
                    "%s: %d non-finite %r slice values dropped", exp.name, dropped, col
                )
            self._row_columns[(exp.name, mn)] = col
            self._nonfinite[(exp.name, mn)] = dropped
            metrics[mn] = subject_means(values)
        return metrics

    @staticmethod
    def _slice_pass(exp: ExperimentStatus) -> SlicePass | None:
        """The arm's per-slice rows at its early-stopping pass, if it wrote any."""
        files = [Path(p) for p in exp.slice_rows_paths if Path(p).is_file()]
        if not files and exp.output_dir:
            files = find_rows_files(exp.output_dir)
        if not files and exp.metrics_csv_path:
            files = find_rows_files(Path(exp.metrics_csv_path).parent)
        if not files:
            return None
        return read_pass(files, early_stopping_iteration(run_root(files[0])))

    # ── Leaderboard ──────────────────────────────────────────────

    def _build_leaderboard(self, all_metrics: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
        rows = []
        for exp_name, metrics in all_metrics.items():
            for mn, vals in metrics.items():
                if exp_name in self._slice_passes:
                    col = self._row_columns[(exp_name, mn)]
                    stats = summarise(vals, col)
                    stats["iteration"] = self._slice_passes[exp_name].iteration
                    stats["nonfinite_dropped"] = self._nonfinite[(exp_name, mn)]
                else:
                    stats = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "median": float(np.median(vals)),
                        "n": len(vals),
                        "unit": "samples",
                    }
                rows.append({"experiment": exp_name, "metric": mn, **stats})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Resolve rank direction from the metric SSOT rather than a hand-rolled
        # lower-better set. The old literal set {lpips,mse,nmse,loss,mae} missed
        # rmse / fid / hfen / brisque and any prefixed name (val_lpips, …), so
        # those lower-is-better metrics were ranked as if higher were better —
        # the leaderboard put the WORST arm at rank 1.
        from spectramr.core.metrics.metric_directions import metric_higher_is_better

        for mn in df["metric"].unique():
            mask = df["metric"] == mn
            # rank(ascending=True) → smallest gets rank 1, i.e. lower-is-better.
            ascending = not metric_higher_is_better(str(mn))
            df.loc[mask, "rank"] = df.loc[mask, "mean"].rank(ascending=ascending)
        return df.sort_values(["metric", "rank"]).reset_index(drop=True)

    # ── Pairwise Comparison ──────────────────────────────────────

    def _pairwise_vs_baseline(self, bl: str, all_m: dict, metric: str) -> list[PairwiseResult]:
        results: list[PairwiseResult] = []
        bl_vals = all_m.get(bl, {}).get(metric)
        if bl_vals is None:
            return results
        for name, m in all_m.items():
            if name == bl:
                continue
            v = m.get(metric)
            if v is None:
                continue
            results.append(self._compare_pair(bl, name, bl_vals, v, metric))
        return results

    def _compare_pair(
        self,
        a: str,
        b: str,
        va: np.ndarray,
        vb: np.ndarray,
        metric: str,
    ) -> PairwiseResult:
        r = PairwiseResult(experiment_a=a, experiment_b=b, metric=metric)
        if a in self._slice_passes and b in self._slice_passes:
            return self._compare_rows(r)
        n = min(len(va), len(vb))
        if n < 2:
            return r
        va, vb = va[:n], vb[:n]
        try:
            r.paired_ttest = StatisticalTests.paired_ttest(va, vb)
        except Exception as e:
            logger.debug(f"Paired t-test failed ({a} vs {b}): {e}")
        try:
            r.wilcoxon = StatisticalTests.wilcoxon_signed_rank(va, vb)
        except Exception as e:
            logger.debug(f"Wilcoxon failed ({a} vs {b}): {e}")
        try:
            r.effect_size = StatisticalTests.cohens_d(va, vb, paired=True)
        except Exception as e:
            logger.debug(f"Cohen's d failed ({a} vs {b}): {e}")
        try:
            r.bootstrap_ci = StatisticalTests.bootstrap_ci(
                va,
                vb,
                n_bootstrap=self.config.n_bootstrap,
                confidence=self.config.confidence,
            )
        except Exception as e:
            logger.debug(f"Bootstrap CI failed ({a} vs {b}): {e}")
        return r

    def _pair_rows(self, a: str, b: str, metric: str) -> PairedSubjects | None:
        """Two row-sourced arms joined on slice identity; ``None`` if a column is missing."""
        col_a = self._row_columns.get((a, metric))
        col_b = self._row_columns.get((b, metric))
        if col_a is None or col_b is None:
            return None
        return pair_on_slices(self._slice_passes[a], self._slice_passes[b], col_a, col_b)

    def _compare_rows(self, r: PairwiseResult) -> PairwiseResult:
        paired = self._pair_rows(r.experiment_a, r.experiment_b, r.metric)
        if paired is None:
            return r
        r.provenance = paired.describe()
        if paired.n_subjects < 2:
            logger.warning(
                "%s vs %s on %s: %d subject(s) scored by both; no paired test",
                r.experiment_a,
                r.experiment_b,
                r.metric,
                paired.n_subjects,
            )
            return r
        tests = compare_subjects(
            paired, n_bootstrap=self.config.n_bootstrap, confidence=self.config.confidence
        )
        r.paired_ttest = tests["paired_ttest"]
        r.wilcoxon = tests["wilcoxon"]
        r.effect_size = tests["effect_size"]
        r.bootstrap_ci = tests["bootstrap_ci"]
        return r

    # ── Correction ───────────────────────────────────────────────

    def _apply_correction(self, results: list[PairwiseResult]) -> dict[str, dict]:
        corrections: dict[str, dict] = {}
        by_metric: dict[str, list[PairwiseResult]] = {}
        for r in results:
            by_metric.setdefault(r.metric, []).append(r)
        for mn, group in by_metric.items():
            pvals = [r.p_value for r in group]
            if any(np.isnan(p) for p in pvals):
                continue
            # Use the canonical dispatch so holm_bonferroni / fdr_bh /
            # benjamini_hochberg are honoured — the old binary if/else silently
            # collapsed every non-"fdr" method (incl. Holm and BH) to the much
            # more conservative Bonferroni.
            corr = StatisticalTests.correct_p_values(
                pvals,
                method=self.config.correction_method,
                alpha=self.config.significance_level,
            )
            corrections[mn] = {
                "method": self.config.correction_method,
                "raw_p_values": pvals,
                **corr,
            }
        return corrections

    # ── Significance Matrix ──────────────────────────────────────

    def _build_significance_matrix(self, all_m: dict, metric: str) -> pd.DataFrame | None:
        names = sorted(all_m.keys())
        n = len(names)
        if n < 2:
            return None
        mat = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                vi = all_m[names[i]].get(metric)
                vj = all_m[names[j]].get(metric)
                if vi is None or vj is None:
                    continue
                if names[i] in self._slice_passes:
                    paired = self._pair_rows(names[i], names[j], metric)
                    if paired is None:
                        continue
                    vi, vj = paired.table["a"].to_numpy(), paired.table["b"].to_numpy()
                k = min(len(vi), len(vj))
                if k < 2:
                    continue
                try:
                    res = StatisticalTests.paired_ttest(vi[:k], vj[:k])
                    mat[i, j] = mat[j, i] = res.get("p_value", 1.0)
                except Exception:
                    logger.debug(f"Significance test failed: {names[i]} vs {names[j]}")
        return pd.DataFrame(mat, index=names, columns=names)

    # ── Summary ──────────────────────────────────────────────────

    def _generate_summary(self, report: CampaignReport, all_m: dict) -> str:
        lines = [
            f"Campaign: {report.campaign_name}",
            f"Experiments: {len(all_m)}",
        ]
        if report.leaderboard is not None and not report.leaderboard.empty:
            lines.append("\nBest per metric:")
            for mn in report.leaderboard["metric"].unique():
                best = report.leaderboard[report.leaderboard["metric"] == mn].iloc[0]
                lines.append(
                    f"  {mn}: {best['experiment']} ({best['mean']:.4f} ± {best['std']:.4f})"
                )
        sig = sum(1 for r in report.pairwise_results if r.is_significant)
        lines.append(f"\nSignificant differences: {sig}/{len(report.pairwise_results)}")

        # Uncertainty summary
        calibrated = [ur for ur in report.uncertainty_reports if ur.ece is not None]
        if calibrated:
            lines.append(f"\nUncertainty calibration ({len(calibrated)} experiments):")
            for ur in calibrated:
                lines.append(
                    f"  {ur.experiment_name}: ECE={ur.ece:.4f}, "
                    f"sharpness={ur.sharpness:.4f}, "
                    f"corr={ur.uncertainty_error_correlation:.4f}"
                )

        return "\n".join(lines)

    # ── LaTeX Export ──────────────────────────────────────────────

    @staticmethod
    def leaderboard_to_latex(
        lb: pd.DataFrame,
        caption: str = "Results",
        label: str = "tab:results",
    ) -> str:
        if lb.empty:
            return "% Empty"
        metrics = lb["metric"].unique()
        exps = lb["experiment"].unique()
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
        ]
        lines.append(f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(m.upper() for m in metrics) + r" \\")
        lines.append(r"\midrule")
        for exp in exps:
            cells = [exp.replace("_", r"\_")]
            for mn in metrics:
                row = lb[(lb["experiment"] == exp) & (lb["metric"] == mn)]
                if not row.empty:
                    mean_val = row.iloc[0]["mean"]
                    std_val = row.iloc[0]["std"]
                    # Bold best performer
                    best_row = lb[lb["metric"] == mn].iloc[0]
                    if exp == best_row["experiment"]:
                        cells.append(f"\\textbf{{{mean_val:.4f}}} ± {std_val:.4f}")
                    else:
                        cells.append(f"{mean_val:.4f} ± {std_val:.4f}")
                else:
                    cells.append("—")
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────


def _load_medical_image(path: str) -> torch.Tensor | None:
    """Load a medical image file as an unnormalized torch tensor.

    Thin compatibility shim around
    :func:`spectramr.data.builders.eval_dataset_builder.load_clean_reference`,
    which is the SSOT (CLAUDE.md #11) for evaluation-side reference
    tensor loading. Kept under the original name so existing call
    sites (private to this module) continue to work without churn.
    """
    try:
        return load_clean_reference(path)
    except Exception as e:
        logger.debug(f"Failed to load {path}: {e}")
        return None
