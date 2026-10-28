"""Hyperparameter Optimization Pipeline - Config-Driven Parameter Sweep.

This pipeline provides grid/random search over configuration parameter spaces
using TrainingSettings as the SSOT (Single Source of Truth), plus an
Optuna-compatible subprocess-based trial runner used by ``HPOCoordinator``.

Supports:
- Grid search over parameter combinations (``run_hpo_grid``)
- Random sampling of parameter values (``run_hpo_random``)
- Config override mechanism (``override_config_param``)
- Training loop integration via subprocess (``subprocess_training_objective``)
- Results aggregation and reporting

Trial isolation
---------------
``subprocess_training_objective`` spawns ``python -m spectramr.cli train`` as a
subprocess for each trial — this is critical because a single trial crash
(NaN, OOM, library mismatch) inside the same Python interpreter can poison
Optuna's TPE state by leaking partial reports into the model. Subprocesses +
explicit ``optuna.TrialPruned`` keep the search robust across hundreds of
trials over multi-day campaigns.
"""

import csv
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml

from spectramr.config.settings import TrainingSettings

logger = logging.getLogger(__name__)


def override_config_param(
    config: TrainingSettings, param_path: str, value: Any
) -> TrainingSettings:
    """Override a nested config parameter using dot notation.

    Args:
        config: TrainingSettings instance
        param_path: Dot-separated path (e.g., 'optimization.optimizer.learning_rate')
        value: New value for parameter

    Returns:
        New TrainingSettings with parameter overridden
    """
    # Convert to dict, override, rebuild
    config_dict = config.model_dump()

    # Navigate nested dict
    parts = param_path.split(".")
    target = config_dict
    for part in parts[:-1]:
        if part not in target:
            raise KeyError(f"Config path not found: {param_path}")
        target = target[part]

    # Set value
    if parts[-1] not in target:
        raise KeyError(f"Config parameter not found: {param_path}")
    target[parts[-1]] = value

    # Rebuild config with updated dict
    return TrainingSettings(**config_dict)


def run_hpo_grid(
    config_path: Path,
    param_grid: dict[str, list],
    output_dir: Path,
    training_fn: Callable | None = None,
    device: str = "cuda",
    metric: str = "final_loss",
    minimize: bool = True,
) -> dict[str, Any]:
    """Run grid search over hyperparameter combinations.

    Args:
        config_path: Path to base training config YAML
        param_grid: Dict mapping param paths to lists of values
                   e.g., {'optimization.optimizer.learning_rate': [1e-4, 1e-3]}
        output_dir: Directory to save results
        training_fn: Optional training function to evaluate each config
                    Signature: (config, device) -> metrics_dict
        device: Device for training ('cuda' or 'cpu')
        metric: Metric key to optimize. Defaults to ``final_loss`` — the key the
                standard evaluator (``ablation.train_and_score``) actually emits.
                Loose substring matching applies (see ``_resolve_metric``), so
                ``val_psnr_best`` etc. also work.
        minimize: Optimization direction. ``True`` for a loss (lower is better);
                  set ``False`` when ``metric`` is a quality score (PSNR/SSIM).

    Returns:
        Results dict with all trials and best config
    """
    logger.info(f"Starting grid search with {len(param_grid)} parameters")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    base_config = TrainingSettings.from_yaml(str(config_path))
    logger.info(f"Loaded base config from {config_path}")

    # Generate all combinations
    import itertools

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    logger.info(f"Grid search: {len(combinations)} combinations")

    trials = []
    best_trial = None
    best_metric = float("inf") if minimize else -float("inf")

    for trial_id, combo in enumerate(combinations, 1):
        logger.info(f"\n--- Trial {trial_id}/{len(combinations)} ---")

        # Override config
        config = base_config
        trial_params = dict(zip(param_names, combo, strict=False))
        for param, value in trial_params.items():
            logger.info(f"  {param} = {value}")
            config = override_config_param(config, param, value)

        # Run training if provided
        metrics = {}
        if training_fn:
            try:
                metrics = training_fn(config, device)
                logger.info(f"Trial {trial_id} metrics: {metrics}")

                # Track best trial (direction-aware; keyed on a metric the
                # evaluator actually emits).
                metric_value = _resolve_metric(metrics, metric)
                if _better_metric(metric_value, best_metric, minimize=minimize):
                    best_metric = float(metric_value)
                    best_trial = {
                        "trial_id": trial_id,
                        "params": trial_params,
                        "metrics": metrics,
                    }
            except Exception as e:
                logger.error(f"Trial {trial_id} failed: {e}")
                metrics = {"error": str(e)}

        # Save trial result
        trial_result = {
            "trial_id": trial_id,
            "params": trial_params,
            "metrics": metrics,
        }
        trials.append(trial_result)

        # Save individual trial config (Pydantic has no direct YAML export here;
        # the per-trial record is persisted as JSON).
        trial_json_path = output_dir / f"trial_{trial_id:03d}.json"
        with open(trial_json_path, "w") as f:
            json.dump(trial_result, f, indent=2, default=str)

    # Compile results
    results = {
        "hpo_type": "grid",
        "base_config": str(config_path),
        "param_grid": param_grid,
        "num_trials": len(combinations),
        "trials": trials,
        "best_trial": best_trial,
    }

    # Save results
    results_path = output_dir / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    if best_trial:
        logger.info(f"\nBest trial: #{best_trial['trial_id']}")
        logger.info(f"Best params: {best_trial['params']}")
        logger.info(f"Best metrics: {best_trial['metrics']}")

    return results


def run_hpo_random(
    config_path: Path,
    param_dist: dict[str, Any],
    num_trials: int,
    output_dir: Path,
    training_fn: Callable | None = None,
    device: str = "cuda",
    seed: int | None = None,
    metric: str = "final_loss",
    minimize: bool = True,
) -> dict[str, Any]:
    """Run random search over hyperparameter space.

    Args:
        config_path: Path to base training config YAML
        param_dist: Dict mapping param paths to distributions
                   e.g., {'optimization.optimizer.learning_rate': ('loguniform', 1e-5, 1e-2)}
        num_trials: Number of random samples to try
        output_dir: Directory to save results
        training_fn: Optional training function to evaluate each config
        device: Device for training ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        metric: Metric key to optimize (defaults to ``final_loss``; loose
                substring matching applies).
        minimize: Optimization direction (``True`` for a loss).

    Returns:
        Results dict with all trials and best config
    """
    import random

    import numpy as np

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    logger.info(f"Starting random search with {num_trials} trials")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    base_config = TrainingSettings.from_yaml(str(config_path))
    logger.info(f"Loaded base config from {config_path}")

    trials = []
    best_trial = None
    best_metric = float("inf") if minimize else -float("inf")

    for trial_id in range(1, num_trials + 1):
        logger.info(f"\n--- Trial {trial_id}/{num_trials} ---")

        # Sample random params
        config = base_config
        trial_params = {}

        for param_name, dist_spec in param_dist.items():
            if isinstance(dist_spec, (list, tuple)) and len(dist_spec) >= 2:
                dist_type = dist_spec[0]

                if dist_type == "uniform":
                    _, low, high = dist_spec
                    value = random.uniform(low, high)
                elif dist_type == "loguniform":
                    _, low, high = dist_spec
                    value = np.exp(random.uniform(np.log(low), np.log(high)))
                elif dist_type == "choice":
                    _, *choices = dist_spec
                    value = random.choice(choices)
                elif dist_type == "randint":
                    _, low, high = dist_spec
                    value = random.randint(low, high)
                else:
                    # No-silent-fallback (CLAUDE.md #9): an unknown distribution
                    # type must raise, not silently drop the hyperparameter from
                    # the sweep (which would run the base value and look green).
                    raise ValueError(
                        f"Unknown distribution type {dist_type!r} for param "
                        f"{param_name!r}; expected one of "
                        "uniform/loguniform/choice/randint."
                    )

                trial_params[param_name] = value
                logger.info(f"  {param_name} = {value}")
                config = override_config_param(config, param_name, value)

        # Run training if provided
        metrics = {}
        if training_fn:
            try:
                metrics = training_fn(config, device)
                logger.info(f"Trial {trial_id} metrics: {metrics}")

                # Track best trial (direction-aware; keyed on a metric the
                # evaluator actually emits).
                metric_value = _resolve_metric(metrics, metric)
                if _better_metric(metric_value, best_metric, minimize=minimize):
                    best_metric = float(metric_value)
                    best_trial = {
                        "trial_id": trial_id,
                        "params": trial_params,
                        "metrics": metrics,
                    }
            except Exception as e:
                logger.error(f"Trial {trial_id} failed: {e}")
                metrics = {"error": str(e)}

        # Save trial result
        trial_result = {
            "trial_id": trial_id,
            "params": trial_params,
            "metrics": metrics,
        }
        trials.append(trial_result)

        # Save trial config
        trial_json_path = output_dir / f"trial_{trial_id:03d}.json"
        with open(trial_json_path, "w") as f:
            json.dump(trial_result, f, indent=2, default=str)

    # Compile results
    results = {
        "hpo_type": "random",
        "base_config": str(config_path),
        "param_dist": {k: v for k, v in param_dist.items()},
        "num_trials": num_trials,
        "seed": seed,
        "trials": trials,
        "best_trial": best_trial,
    }

    # Save results
    results_path = output_dir / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    if best_trial:
        logger.info(f"\nBest trial: #{best_trial['trial_id']}")
        logger.info(f"Best params: {best_trial['params']}")
        logger.info(f"Best metrics: {best_trial['metrics']}")

    return results


# ---------------------------------------------------------------------------
# Subprocess-based trial runner for Optuna integration
# ---------------------------------------------------------------------------


def _read_latest_metrics(csv_path: Path) -> dict[str, float]:
    """Read the last row of a training metrics CSV; scalars only."""
    if not csv_path.exists():
        return {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        last = None
        for row in reader:
            last = row
        if last is None:
            return {}
    out: dict[str, float] = {}
    for k, v in last.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _better_metric(candidate: float | None, incumbent: float, *, minimize: bool) -> bool:
    """Direction-aware comparison for HPO trial selection.

    Returns True iff ``candidate`` improves on ``incumbent`` under the
    optimization direction. A trial that did not produce the metric
    (``candidate is None``) never wins — this prevents the pre-fix bug where a
    nonexistent ``loss`` key resolved to a sentinel that silently maximized the
    WRONG direction and left ``best_trial`` unselected.
    """
    if candidate is None:
        return False
    return candidate < incumbent if minimize else candidate > incumbent


def _resolve_metric(metrics: dict[str, float], metric_name: str) -> float | None:
    """Find ``metric_name`` in the CSV row, with loose matching for column
    naming variations (e.g. PSNR may be logged as ``val_psnr_4x``).
    """
    if metric_name in metrics:
        return metrics[metric_name]
    needle = metric_name.lower()
    for k, v in metrics.items():
        if k.lower() == needle:
            return v
    # Loose match: any column containing the metric name as a substring
    for k, v in metrics.items():
        if needle in k.lower():
            return v
    return None


def _build_trial_yaml(
    base_config: dict,
    trial_number: int,
    overrides: dict[str, Any],
    output_root: Path,
    max_iterations: int,
) -> Path:
    """Materialize one trial's YAML on disk, applying dotted-path overrides.

    Mirrors the layout the standalone Optuna script uses so trials are
    self-contained (own logs, checkpoints, metrics).
    """
    import copy

    cfg = copy.deepcopy(base_config)
    name = f"trial_{trial_number:04d}"
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)

    # Apply overrides via shared dotted-path utility (supports [name=foo]
    # selectors so search spaces can target list-of-dict fields like
    # losses.kspace_losses[name=log_spectral].weight).
    # Imported from its canonical home, not re-exported through
    # ``hpo_search_spaces``. That re-export existed, and `366e3e1a6`, a formatting commit,
    # removed it as an unused import -- unused in that MODULE, but the only path
    # this line and the tests had. The result was an ImportError on every HPO
    # trial that applies an override, i.e. every HPO run, unseen because it is a
    # function-local import (invisible to check_layering.sh's ^-anchored greps)
    # and because ruff's F401 cannot know a re-export has downstream consumers.
    from spectramr.infrastructure.hpo.dotted_override import apply_dotted_override

    for path, value in overrides.items():
        apply_dotted_override(cfg, path, value)

    # Repoint output paths so trials never collide
    cfg.setdefault("training", {})["output_dir"] = str(out)
    cfg["training"]["max_iterations"] = int(max_iterations)
    cfg.setdefault("logging", {})["log_dir"] = str(out / "logs")
    cfg["logging"]["experiment_name"] = name
    # `artifacts` has no reader (#698); a copied block would name the base arm's directory.
    cfg.pop("artifacts", None)
    cfg.setdefault("loss_logging", {})["csv_path"] = str(out / "logs" / "loss_log.csv")
    cfg["loss_logging"]["output_dir"] = str(out / "metrics")
    cfg.setdefault("metrics", {})["output_dir"] = str(out / "metrics")
    cfg.setdefault("metadata", {})["name"] = name

    yaml_path = out / "config.yaml"
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, indent=2, default_flow_style=False)
    return yaml_path


def subprocess_training_objective(
    base_config_path: str,
    output_root: Path,
    metric_name: str = "val_loss",
    max_iterations: int = 30000,
    poll_interval_seconds: float = 30.0,
    metric_step_milestones: tuple[int, ...] = (4000, 8000, 16000, 32000),
    overrides_fn: Callable[["optuna.trial.Trial"], dict[str, Any]] | None = None,  # noqa: F821
    minimize: bool | None = None,
) -> Callable:
    """Return an Optuna-compatible objective callable backed by a subprocess.

    Each trial:

    1. Calls ``overrides_fn(trial)`` to sample hyperparameters as a dotted-path
       override dict (e.g. ``{"optimization.optimizer.learning_rate": 1e-4}``). If not
       provided, the optimizer's default sampling is used (which means an
       empty override dict — only useful when the optimizer mutates the
       config another way; in practice you should always pass overrides_fn).
    2. Materializes the resulting YAML on disk under
       ``output_root/trial_NNNN/config.yaml``.
    3. Spawns ``python -m spectramr.cli train --config ...`` as a subprocess.
    4. Polls the resulting ``loss_log.csv`` every ``poll_interval_seconds``,
       reports the score via ``trial.report(score, step)``, and prunes the
       trial if Optuna's pruner says so (kills the subprocess on prune).
    5. Returns the final score.

    Args:
        base_config_path: Path to the YAML each trial mutates.
        output_root: Directory under which per-trial subdirectories live.
        metric_name: CSV column to optimize (looser substring match if the
            exact column isn't present — e.g. ``val_psnr`` matches
            ``val_psnr_4x``).
        max_iterations: Truncated training budget per trial.
        poll_interval_seconds: How often to read the metrics CSV.
        metric_step_milestones: Iterations at which to consult the pruner.
        overrides_fn: ``trial -> {dotted_path: value}`` sampler.
        minimize: If True, return -score so the optimizer (which maximizes
            by default) actually minimizes. Auto-detect for ``val_loss`` /
            ``loss`` / ``error`` style names if None.

    Returns:
        ``objective(trial) -> float`` callable suitable for
        ``study.optimize`` or ``EnhancedHPOptimizer.optimize``.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(base_config_path) as fh:
        base_config = yaml.safe_load(fh)

    if minimize is None:
        # Auto-detect: lower-is-better names.
        lower_is_better = any(
            kw in metric_name.lower() for kw in ("loss", "error", "mse", "nmse", "mae", "phase_mse")
        )
        minimize = lower_is_better

    def _objective(trial: "optuna.trial.Trial") -> float:  # noqa: F821
        try:
            import optuna
        except ImportError as e:
            raise ImportError("optuna is required for subprocess HPO") from e

        overrides = overrides_fn(trial) if overrides_fn else {}
        yaml_path = _build_trial_yaml(
            base_config=base_config,
            trial_number=trial.number,
            overrides=overrides,
            output_root=output_root,
            max_iterations=max_iterations,
        )

        # The CSV path was set inside _build_trial_yaml.
        actual_csv = yaml_path.parent / "logs" / "loss_log.csv"

        cmd = [sys.executable, "-m", "spectramr.cli", "train", "--config", str(yaml_path)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        last_score: float | None = None
        last_step = 0
        next_milestone_idx = 0

        try:
            while proc.poll() is None:
                time.sleep(poll_interval_seconds)
                metrics = _read_latest_metrics(actual_csv)
                if not metrics:
                    continue
                step = int(metrics.get("step", metrics.get("iteration", last_step)))
                raw = _resolve_metric(metrics, metric_name)
                if raw is None:
                    continue
                score = -raw if minimize else raw
                last_score = score
                last_step = step
                trial.report(score, step)
                while (
                    next_milestone_idx < len(metric_step_milestones)
                    and step >= metric_step_milestones[next_milestone_idx]
                ):
                    if trial.should_prune():
                        logger.info(
                            "Trial %d pruned at step %d (score=%.3f)",
                            trial.number,
                            step,
                            score,
                        )
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=30)
                        raise optuna.TrialPruned()
                    next_milestone_idx += 1
            proc.wait()
        except optuna.TrialPruned:
            raise
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise

        if proc.returncode != 0:
            logger.error("Trial %d trainer exited with code %d", trial.number, proc.returncode)
            # Prune rather than returning a sentinel float: sentinel values
            # feed OOM/crash points into the TPE surrogate, biasing it toward
            # failure regions. Pruned trials are excluded from the surrogate.
            raise optuna.TrialPruned()

        if last_score is None:
            return float("-inf") if not minimize else float("inf")
        return last_score

    return _objective
