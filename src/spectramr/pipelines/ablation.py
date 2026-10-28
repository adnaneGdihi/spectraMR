"""Ablation Study Pipeline - Component Impact Analysis.

This pipeline enables systematic ablation studies to measure the impact
of individual components, loss functions, and training configurations.

Features:
- Component enable/disable via config
- Loss function variation testing
- Baseline vs ablation comparison
- Metrics collection and impact reporting
"""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.logging.log_policy import apply_logging_config
from spectramr.infrastructure.training.builders import DataBuilder

logger = logging.getLogger(__name__)


def _resolve_validation_metrics_csv(config: Any) -> Path:
    """Resolve the validation-metrics CSV path the training pipeline writes.

    ``run_training_pipeline`` derives ``run_dir`` from
    ``config.training.output_dir`` (default ``experiments/outputs``) and writes
    ``<run_dir>/logs/validation_metrics.csv``. The canonical location is tried
    first; if absent, fall back to the legacy ``config.logging.log_dir`` path
    for backward compatibility.
    """
    candidates: list[Path] = []

    output_dir = None
    training = getattr(config, "training", None)
    if training is not None:
        output_dir = getattr(training, "output_dir", None)
    output_dir = output_dir or "experiments/outputs"
    candidates.append(Path(output_dir) / "logs" / "validation_metrics.csv")

    logging_cfg = getattr(config, "logging", None)
    log_dir = (logging_cfg.sinks.dir if logging_cfg else None) if logging_cfg else None
    if log_dir:
        candidates.append(Path(log_dir) / "validation_metrics.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Nothing exists yet — return the canonical location for the warning path.
    return candidates[0]


def train_and_score(config: TrainingSettings, device: str | None = None) -> dict[str, float]:
    """Train one config to completion and return its best validation metrics.

    This is the default ``evaluation_fn`` for :func:`run_ablation_study`. The
    historical pipeline shipped *without* a training-backed evaluator, so
    ``run_ablation_study(..., evaluation_fn=None)`` silently produced empty
    metrics and an empty impact analysis. This wires the real path:

    1. ``run_training_pipeline(config, device)`` trains the variant (it returns
       only ``{success, final_loss}`` — validation metrics are streamed to
       ``validation_metrics.csv`` under the run's log dir, not returned).
    2. ``_summarise_best_metrics_from_csv`` reads that CSV back and returns the
       best value per metric column (max for psnr/ssim/…, min for losses).

    Args:
        config: Fully-resolved (and override-applied) ``TrainingSettings``.
        device: Device string forwarded to the training pipeline. ``None``
            means "not requested here" -- the config, then the accelerated-run
            resolver, decides (non-negotiable 9b).

    Returns:
        ``{<metric>_best: value, ..., final_loss: float, success: bool}``.
        On a failed run the dict carries ``{success: False, error: ...}`` so
        the caller's impact analysis degrades gracefully instead of crashing
        the whole study.
    """
    # Local imports: pulling these at module load would drag the training
    # stack (torch, accelerator) into every importer of the ablation module.
    from spectramr.pipelines.train import (
        _summarise_best_metrics_from_csv,
        run_training_pipeline,
    )

    result = run_training_pipeline(config, device=device)
    # Only an explicit True is success; an absent flag is unknown (#181).
    if not (isinstance(result, dict) and result.get("success") is True):
        error = result.get("error") if isinstance(result, dict) else None
        return {"success": False, "error": error or "training reported no success flag"}

    # Read validation metrics from where run_training_pipeline actually writes
    # them: <training.output_dir>/logs/validation_metrics.csv.
    csv_path = _resolve_validation_metrics_csv(config)
    metrics = _summarise_best_metrics_from_csv(str(csv_path))
    if not metrics:
        logger.warning(
            "[ablation] no validation metrics found at %s — the variant "
            "trained but produced no validation rows (too few iterations, or "
            "validation disabled). Impact analysis for this arm will be empty.",
            csv_path,
        )
    metrics["success"] = True  # type: ignore[assignment]
    # final_loss is None when the run recorded no g_total_loss; omit it like any missing metric.
    if isinstance(result, dict) and result.get("final_loss") is not None:
        metrics["final_loss"] = float(result["final_loss"])
    return metrics


def run_ablation_pipeline(config_path: Path) -> dict[str, Any]:
    """Run legacy ablation pipeline with dataset fraction loaders.

    Args:
        config_path: Path to training config YAML

    Returns:
        dict with per-fraction status and error details
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    settings = TrainingSettings.from_yaml(str(config_path))
    apply_logging_config(settings.logging)
    data_builder = DataBuilder(settings)

    results: dict[str, Any] = {
        "success": True,
        "fractions": {},
    }

    loaders = data_builder.build_ablation_subsets().build()
    for fraction_name, loader in loaders.items():
        try:
            for _ in loader:
                pass
            results["fractions"][fraction_name] = {"status": "completed"}
        except Exception as exc:
            results["success"] = False
            results["fractions"][fraction_name] = {
                "status": "failed",
                "error": str(exc),
            }

    return results


def create_ablation_variant(
    base_config: TrainingSettings,
    ablation_spec: dict[str, Any],
    output_dir: str | Path | None = None,
) -> TrainingSettings:
    """Create a config variant with ablation applied.

    Args:
        base_config: Base TrainingSettings
        ablation_spec: Dict with keys:
                      - enabled: List of features to enable
                      - disabled: List of features to disable
                      - overrides: Dict of config overrides
                      - removed: List of components to remove
        output_dir: When provided, repoints ``training.output_dir`` to this
            per-variant directory. Load-bearing: ``run_training_pipeline`` writes
            ``<training.output_dir>/logs/validation_metrics.csv`` and
            :func:`_resolve_validation_metrics_csv` reads it back from the SAME
            key, so *without* a distinct dir the baseline and every variant
            train to and read from one shared CSV — each run clobbers the last
            and the impact deltas are a meaningless mixture.

    Returns:
        Modified TrainingSettings
    """
    # exclude_unset: a full dump marks every schema default as explicitly set,
    # and fields_set-sensitive validators (the DeepSpeed declared-but-inert
    # knob checks, config/schemas/base.py:639/:724) reject such a dict on
    # reconstruction — only a set-ness-preserving dump round-trips.
    config_dict = base_config.model_dump(exclude_unset=True)

    # Isolate this arm's outputs BEFORE reconstruction so its
    # validation_metrics.csv cannot collide with sibling arms (see docstring).
    if output_dir is not None:
        training = config_dict.get("training")
        if not isinstance(training, dict):
            training = {}
            config_dict["training"] = training
        training["output_dir"] = str(output_dir)

    # Apply overrides (feature-specific changes)
    for override_path, value in ablation_spec.get("overrides", {}).items():
        parts = override_path.split(".")
        target = config_dict
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    # Apply enable/disable flags (boolean features)
    for feature in ablation_spec.get("enabled", []):
        parts = feature.split(".")
        target = config_dict
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        if isinstance(target, dict):
            target[parts[-1]] = True

    for feature in ablation_spec.get("disabled", []):
        parts = feature.split(".")
        target = config_dict
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        if isinstance(target, dict):
            target[parts[-1]] = False

    return TrainingSettings(**config_dict)


def run_ablation_study(
    config_path: Path,
    ablations: dict[str, dict[str, Any]],
    output_dir: Path,
    evaluation_fn: Callable | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run ablation study comparing baseline and variants.

    Args:
        config_path: Path to baseline training config YAML
        ablations: Dict mapping ablation names to specifications
                  e.g., {
                      'no_data_consistency': {
                          'overrides': {'physics.data_consistency.enabled': False}
                      },
                      'no_perceptual_loss': {
                          'disabled': ['objectives.reconstruction.perceptual_loss.enabled']
                      },
                      'lower_lr': {
                          'overrides': {'optimization.optimizer.learning_rate': 1e-4}
                      }
                  }
        output_dir: Directory to save results
        evaluation_fn: Function to evaluate each config
                      Signature: (config, device) -> metrics_dict
        device: Device for evaluation ('cuda'/'cpu'/'auto'), or ``None`` to
            let the config and the 9b resolver decide

    Returns:
        Results dict with baseline and ablation metrics/comparison
    """
    logger.info(f"Starting ablation study with {len(ablations)} variants")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    base_config = TrainingSettings.from_yaml(str(config_path))
    apply_logging_config(base_config.logging)
    logger.info(f"Loaded baseline config from {config_path}")

    # Evaluate baseline. Give the baseline its OWN output dir so it does not
    # share ``validation_metrics.csv`` with the variants (metrics collision).
    logger.info("\n=== Baseline Configuration ===")
    baseline_metrics = {}
    if evaluation_fn:
        try:
            baseline_config = create_ablation_variant(
                base_config, {}, output_dir=output_dir / "baseline"
            )
            baseline_metrics = evaluation_fn(baseline_config, device)
            logger.info(f"Baseline metrics: {baseline_metrics}")
        except Exception as e:
            logger.error(f"Baseline evaluation failed: {e}")
            baseline_metrics = {"error": str(e)}

    # Track results
    results = {
        "ablation_study": True,
        "base_config": str(config_path),
        "baseline_metrics": baseline_metrics,
        "ablations": {},
        "impact_analysis": {},
    }

    # Run ablations
    for ablation_name, ablation_spec in ablations.items():
        logger.info(f"\n=== Ablation: {ablation_name} ===")
        logger.info(f"Spec: {ablation_spec}")

        # Create variant with its own isolated output dir (avoid CSV collision).
        arm_dir = str(ablation_name).replace("/", "_").replace(os.sep, "_")
        variant_config = create_ablation_variant(
            base_config, ablation_spec, output_dir=output_dir / arm_dir
        )

        # Evaluate variant
        variant_metrics = {}
        if evaluation_fn:
            try:
                variant_metrics = evaluation_fn(variant_config, device)
                logger.info(f"Variant metrics: {variant_metrics}")
            except Exception as e:
                logger.error(f"Variant evaluation failed: {e}")
                variant_metrics = {"error": str(e)}

        # Compute impact (delta from baseline)
        impact = {}
        if baseline_metrics and variant_metrics:
            for key in baseline_metrics:
                if key in variant_metrics and isinstance(baseline_metrics[key], (int, float)):
                    delta = variant_metrics[key] - baseline_metrics[key]
                    pct_change = (
                        (delta / baseline_metrics[key] * 100) if baseline_metrics[key] != 0 else 0
                    )
                    impact[key] = {
                        "baseline": baseline_metrics[key],
                        "variant": variant_metrics[key],
                        "delta": delta,
                        "pct_change": pct_change,
                    }

        # Save ablation result
        ablation_result = {
            "name": ablation_name,
            "spec": ablation_spec,
            "metrics": variant_metrics,
            "impact": impact,
        }
        results["ablations"][ablation_name] = ablation_result
        results["impact_analysis"][ablation_name] = impact

        # Save variant config JSON
        variant_json_path = output_dir / f"ablation_{ablation_name}.json"
        with open(variant_json_path, "w") as f:
            json.dump(ablation_result, f, indent=2, default=str)

    # Generate comparison report
    logger.info("\n=== Impact Summary ===")
    for ablation_name, impact in results["impact_analysis"].items():
        logger.info(f"\n{ablation_name}:")
        for metric_name, metric_impact in impact.items():
            logger.info(
                f"  {metric_name}: {metric_impact['baseline']:.6f} → "
                f"{metric_impact['variant']:.6f} ({metric_impact['pct_change']:+.2f}%)"
            )

    # Save full results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results saved to {results_path}")

    return results


def run_loss_ablation(
    config_path: Path,
    loss_configs: dict[str, dict[str, Any]],
    output_dir: Path,
    evaluation_fn: Callable | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run ablation specifically for loss function weights and components.

    Args:
        config_path: Path to baseline training config YAML
        loss_configs: Dict mapping loss variant names to weight overrides
                     e.g., {
                         'l1_only': {'overrides': {
                             'objectives.reconstruction.lambda_l1': 1.0,
                             'objectives.reconstruction.lambda_perceptual': 0.0,
                             'objectives.gan.enabled': False
                         }},
                         'no_gan': {'disabled': ['objectives.gan.enabled']}
                     }
        output_dir: Directory to save results
        evaluation_fn: Function to evaluate each config
        device: Device for evaluation ('cuda'/'cpu'/'auto'), or ``None`` to
            let the config and the 9b resolver decide

    Returns:
        Results dict with loss ablation analysis
    """
    logger.info(f"Starting loss function ablation with {len(loss_configs)} variants")
    return run_ablation_study(
        config_path,
        ablations=loss_configs,
        output_dir=output_dir,
        evaluation_fn=evaluation_fn,
        device=device,
    )
