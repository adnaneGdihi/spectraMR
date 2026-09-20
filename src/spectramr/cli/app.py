#!/usr/bin/env python
"""Unified CLI Entry Point for spectraMR System.

This provides a single, platform-independent entry point for all operations.

Usage:
    python -m spectramr.cli train --config experiments/active/dummy_gan.yaml
    python -m spectramr.cli predict --config run.yaml --model checkpoints/best.pt --input data/test/
    python -m spectramr.cli benchmark --suite standard
    python -m spectramr.cli list-features --module all --format markdown
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spectramr.core.compute_device import (
    AcceleratorRequiredError,
    resolve_torch_device,
)
from spectramr.cli.val_batches import (
    add_val_batches_argument,
    resolve_val_batches_overrides,
)

logger = logging.getLogger(__name__)


def _dispatch_training(args: argparse.Namespace, *, is_sanity_check: bool) -> int:
    """Dispatch to the ``main.py`` training bootstrap WITHOUT a ``sys.argv`` round-trip.

    Historically ``train``/``sanity_check`` rebuilt ``sys.argv`` as a string
    list and re-invoked ``spectramr.main:main()``, which re-parsed everything
    through a *second*, divergent argparse. That dual-parser seam dropped
    flags silently (``--device``/``--seed`` never reached training; ``--debug``
    errored because ``main.py``'s train parser doesn't define it) and forced
    the two parsers to be hand-synced.

    We now call ``main``'s command functions directly with the already-parsed
    ``args``. Importing ``spectramr.main`` still runs its load-bearing
    import-time setup (cache root, thread isolation, ``PYTORCH_CUDA_ALLOC_CONF``
    — all set *before* ``import torch``), so that ordering is preserved; we
    just skip the redundant re-parse. ``main.__common_train_setup`` reads
    ``args.{config, device, dry_run, override, seed, resume}``.
    """
    if getattr(args, "debug", False):
        logging.getLogger("spectramr").setLevel(logging.DEBUG)

    # Normalize the attribute surface the main.py bootstrap expects, so a
    # missing optional flag is a documented default, never an AttributeError.
    args.dry_run = bool(getattr(args, "dry_run", False))
    args.device = getattr(args, "device", None)
    args.seed = getattr(args, "seed", None)
    args.override = getattr(args, "override", None)
    args.resume = getattr(args, "resume", None)
    # `--val-batches` is sugar for an override, resolved here rather than in each
    # subparser so `train` and `sanity_check` cannot drift apart on it.
    args.override = resolve_val_batches_overrides(
        args.override, getattr(args, "val_batches", None)
    )

    from spectramr.main import sanity_check_command, train_command

    if is_sanity_check:
        sanity_check_command(args)
    else:
        train_command(args)
    # train_command / sanity_check_command call sys.exit(1) on pipeline
    # failure; reaching here means success.
    return 0


def train(args: argparse.Namespace) -> int:
    """Run training pipeline."""
    return _dispatch_training(args, is_sanity_check=False)


def sanity_check(args: argparse.Namespace) -> int:
    """Run sanity check pipeline."""
    return _dispatch_training(args, is_sanity_check=True)


def _parse_vary_specs(vary: list[str] | None) -> dict[str, dict[str, object]]:
    """Turn ``--vary key=value`` strings into a ``run_ablation_study`` spec dict.

    Each ``--vary model.model_kwargs.force_pure_kspace=false`` becomes one named
    variant ``{"force_pure_kspace_false": {"overrides": {<path>: <parsed>}}}``.
    The value is parsed with the same type coercion as ``--override`` so
    ``false``/``1e-4``/``none`` land as bool/float/None, not strings.

    Raises:
        ValueError: on a malformed spec (no ``=``) — no silent fallback.
    """
    from spectramr.config.overrides import _parse_value  # canonical value parser (SSOT)

    specs: dict[str, dict[str, object]] = {}
    for raw in vary or []:
        if "=" not in raw:
            raise ValueError(f"--vary expects 'dotted.path=value', got {raw!r} (no '=').")
        path, value = raw.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"--vary has an empty key in {raw!r}.")
        # Variant name: leaf key + value, sanitized — stable and human-readable.
        leaf = path.split(".")[-1]
        safe_val = str(value).strip().replace(".", "_").replace("/", "_")
        name = f"{leaf}_{safe_val}"
        specs[name] = {"overrides": {path: _parse_value(value)}}
    return specs


def ablation(args: argparse.Namespace) -> int:
    """Run an ablation study: baseline + one variant per ``--vary`` override.

    Trains the baseline config and each variant (a single dotted-path override),
    then writes ``ablation_results.json`` with the baseline→variant delta and
    %-change per validation metric.

    Example:
        spectramr ablation -c base.yaml \\
            --vary model.model_kwargs.force_pure_kspace=false \\
            --output-dir experiments/results/exp11_fpk_ablation --device cuda
    """
    from spectramr.pipelines.ablation import run_ablation_study, train_and_score

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    try:
        ablations = _parse_vary_specs(args.vary)
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    if not ablations:
        logger.error("ablation requires at least one --vary key=value spec.")
        return 2

    # Optional per-variant iteration cap for quick local sweeps: fold it into
    # every variant's override set so the baseline and variants are comparable.
    if getattr(args, "max_iterations", None):
        for spec in ablations.values():
            spec["overrides"]["training.max_iterations"] = int(args.max_iterations)

    output_dir = (
        Path(args.output_dir)
        if getattr(args, "output_dir", None)
        else config_path.parent / f"{config_path.stem}_ablation"
    )

    logger.info(
        f"[ablation] baseline={config_path.name}, "
        f"{len(ablations)} variant(s): {list(ablations)}, out={output_dir}"
    )
    if getattr(args, "max_iterations", None):
        logger.warning(
            "[ablation] baseline + %d variant(s) train SEQUENTIALLY in this "
            "process. For many arms / long runs use the cluster `campaign` "
            "command instead.",
            len(ablations),
        )

    results = run_ablation_study(
        config_path=config_path,
        ablations=ablations,
        output_dir=output_dir,
        evaluation_fn=train_and_score,
        # No ``or "cuda"``: an omitted --device must stay ``None`` so the
        # config's own device (and then the 9b resolver) decides. Pinning it
        # to "cuda" here made ``run.device`` unreachable for this verb AND
        # made the requirement unconditional, since resolve_compute_device
        # treats an explicit "cuda" as a hard requirement FORCE_CPU cannot
        # relax.
        device=getattr(args, "device", None),
    )

    # Surface the impact summary on the console.
    impact = results.get("impact_analysis", {})
    for name, metrics in impact.items():
        logger.info(f"[ablation] {name}:")
        for metric, d in metrics.items():
            logger.info(
                f"    {metric}: {d['baseline']:.6f} -> {d['variant']:.6f} ({d['pct_change']:+.2f}%)"
            )
    logger.info(f"[ablation] full results -> {output_dir / 'ablation_results.json'}")
    return 0


def infer(args: argparse.Namespace) -> int:
    """Run inference via the SSOT inference pipeline (delegates to main.py)."""
    args.output = getattr(args, "output", None)
    from spectramr.main import infer_command

    infer_command(args)
    return 0


def infer_dataset(args: argparse.Namespace) -> int:
    """[DEPRECATED] Dataset-loader inference; alias for ``infer``."""
    args.output = getattr(args, "output", None)
    args.batch_size = getattr(args, "batch_size", None)
    from spectramr.main import infer_dataset_command

    infer_dataset_command(args)
    return 0


def experiment(args: argparse.Namespace) -> int:
    """Run a managed experiment (directory mgmt + training) via main.py."""
    args.override = getattr(args, "override", None)
    args.seed = getattr(args, "seed", None)
    args.resume = getattr(args, "resume", None)
    from spectramr.main import experiment_command

    experiment_command(args)
    return 0


def train_distributed(args: argparse.Namespace) -> int:
    """DDP training entry point (launch via ``torchrun ... -m spectramr.cli``)."""
    # Importing main first runs its load-bearing import-time env setup
    # (cache root, thread isolation, CUDA alloc — before ``import torch``).
    import spectramr.main  # noqa: F401  (side-effect import)
    from spectramr.pipelines.distributed import run_distributed_training

    run_distributed_training(
        config_path=args.config,
        backend=args.backend,
        resume_path=args.resume,
        overrides=args.override,
    )
    return 0


def predict(args: argparse.Namespace) -> int:
    """Run inference via the SSOT ``run_inference_pipeline``.

    ``predict`` previously used the deprecated, config-less
    ``application.pipelines.inference_pipeline.InferencePipeline`` (emits a
    ``DeprecationWarning`` promoted to an error under ``spectramr.*`` tests, and
    slated for removal — see ``.claude/rules/data.md``). It reconstructed the
    architecture from a pickled ``nn.Module`` (``weights_only=False``), diverging
    from the config-as-SSOT principle (CLAUDE.md #1/#7).

    This routes to the SAME entry point as ``infer`` -- and, since PR "predict:
    canonical builder", through the same *preamble* as well
    (``main.begin_inference_run``: execution ledger, seed, determinism policy,
    accelerator resolution) under the same ``torch.no_grad()``. Previously the
    claim covered only the pipeline function: ``predict`` ran with no audit
    trail, no seed, no determinism policy, a live autograd graph, and a
    hardcoded ``or "cuda"`` device that bypassed the accelerated-run contract
    (non-negotiable 9b). That gap made every ``predict`` run non-reproducible
    while the docstring said otherwise (pitfall #16).

    The settings come from the run's own ``resolved_config.json`` beside
    ``--model`` when it exists (its declared block re-validates to what the
    training run resolved, #1379), else from ``--config``; ``--from-yaml``
    forces the YAML. Neither present: fail loud with guidance rather than fall
    back to the removed config-less path (pitfall #9).
    """
    config = getattr(args, "config", None)
    from_yaml = bool(getattr(args, "from_yaml", False))
    if config is None:
        from spectramr.infrastructure.validation.resolved_config_artifact import (
            resolved_config_beside,
        )

        if from_yaml or resolved_config_beside(getattr(args, "model", None)) is None:
            logger.error(
                "predict requires --config <training.yaml> (the SSOT that produced the "
                "checkpoint) unless the checkpoint's run directory holds resolved_config.json, "
                "which predict then rebuilds the settings from (#1379); the deprecated "
                "config-less InferencePipeline is no longer wired. E.g. `spectramr predict "
                "--config run.yaml --model best.pt --input data/`."
            )
            return 2

    # ``import spectramr.main`` also applies the pre-``import torch`` environment
    # setup (cache root, thread isolation, CUDA alloc) that every other verb in
    # this module relies on.
    import torch

    from spectramr.main import begin_inference_run
    from spectramr.pipelines.infer import run_inference_pipeline

    # No ``or "cuda"``: the device is whatever the 9b resolver returns for the
    # requested string (``None`` -> auto), and a heavy pipeline that cannot get
    # an accelerator raises instead of quietly scoring on CPU.
    settings, device = begin_inference_run(
        config,
        getattr(args, "device", None),
        pipeline="predict",
        checkpoint_path=args.model,
        from_yaml=from_yaml,
    )

    logger.info(f"Loading model from: {args.model}")
    logger.info(f"Input: {args.input}  Output: {args.output}  Config: {config}")

    try:
        with torch.no_grad():
            result = run_inference_pipeline(
                config_path=config,
                checkpoint_path=args.model,
                input_path=args.input,
                output_path=args.output,
                device=str(device),
                from_yaml=from_yaml,
                settings=settings,
            )
        logger.info(f"Inference complete: {result}")
        return 0
    except Exception as e:
        logger.exception("Prediction failed")
        logger.error(f"Error during inference: {e}")
        return 1


def benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks on standard datasets.

    Benchmarking is a heavy pipeline: a CPU "benchmark" measures nothing anyone
    wants to know, so no-accelerator RAISES instead of quietly producing numbers
    off the wrong device (which would then be compared against GPU baselines).
    """
    import time

    import torch

    logger.info(f"Running benchmark suite: {args.suite}")

    results = {}
    device = _resolve_device(getattr(args, "device", None), pipeline="benchmark")

    if args.suite in ["standard", "all"]:
        # Standard quality metrics benchmark
        logger.info("Running standard quality benchmarks...")
        from spectramr.core.metrics.evaluation_metrics import NMSE, PSNR, SSIM

        # Synthetic benchmark with random tensors
        x = torch.randn(10, 1, 256, 256, device=device)
        y = x + torch.randn_like(x) * 0.1

        ssim = SSIM()(x, y).item()
        psnr = PSNR()(x, y).item()
        nmse = NMSE()(x, y).item()

        results["ssim"] = ssim
        results["psnr"] = psnr
        results["nmse"] = nmse
        logger.info(f"  SSIM: {ssim:.4f}, PSNR: {psnr:.2f} dB, NMSE: {nmse:.6f}")

    if args.suite in ["throughput", "all"]:
        # Throughput benchmark
        logger.info("Running throughput benchmark...")
        x = torch.randn(1, 1, 256, 256, device=device)

        start = time.perf_counter()
        for _ in range(100):
            _ = x * 2 + 1  # Simple operation
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results["throughput_ops_per_sec"] = 100 / elapsed
        logger.info(f"  Throughput: {100 / elapsed:.2f} ops/sec")

    if args.suite in ["memory", "all"]:
        # Memory benchmark
        logger.info("Running memory benchmark...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(32, 64, 256, 256, device=device)
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            results["peak_memory_mb"] = peak_mb
            logger.info(f"  Peak GPU Memory: {peak_mb:.2f} MB")
            del x
        else:
            logger.info("  (GPU not available, skipping memory benchmark)")

    logger.info(f"\nBenchmark complete. Results: {results}")
    return 0


def export(args: argparse.Namespace) -> int:
    """Export model to ONNX/TorchScript."""
    from collections.abc import Mapping

    import torch

    logger.info(f"Exporting model: {args.model}")
    logger.info(f"Format: {args.format}")

    try:
        from spectramr.config.settings import TrainingSettings
        from spectramr.core.module_utils import resolve_state_dict
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder
        from spectramr.shared.utils.data_utils import get_sample_batch

        # ``--config`` is required, so the arm's architecture is always
        # knowable. That is what makes the branch below possible: a checkpoint
        # holding parameters rather than a pickled module used to be refused
        # outright --
        #
        #     "Error: Checkpoint contains state_dict, not model. Cannot export."
        #
        # -- which is every ordinary training checkpoint this repo writes, so
        # `export` did not work on the artifacts it exists to consume. Nothing
        # was missing except the architecture, and the config carries it.
        config = TrainingSettings.from_yaml(str(args.config))
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        pickled = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None

        if isinstance(pickled, torch.nn.Module):
            model = pickled
        else:
            model = (
                ModelBuilder(config, torch.device("cpu"))
                .build_generator()
                .validate()
                .build()["generator"]
            )
            # strict=True here, unlike the warm-start and campaign-evaluation
            # readers: an export is a frozen artifact handed to another runtime,
            # so a partially-loaded model would ship random weights inside a
            # file that looks authoritative. `resolve_state_dict` has already
            # raised on an unrecognised envelope; strict catches the rest.
            model.load_state_dict(
                resolve_state_dict(
                    checkpoint,
                    model.state_dict().keys(),
                    source=str(args.model),
                ),
                strict=True,
            )

        model.eval()

        # Resolve a sample input for tracing
        try:
            sample_input = get_sample_batch(config, device=torch.device("cpu"))
        except Exception as e:
            logger.error(f"Failed to load real data sample from dataloader: {e}")
            logger.error("Fail-fast strict architecture prohibits synthetic fallbacks.")
            return 1

        output_stem = args.model.stem

        if args.format in ["onnx", "both"]:
            onnx_path = args.model.parent / f"{output_stem}.onnx"
            logger.info(f"Exporting ONNX to: {onnx_path}")
            # Single ONNX export SSOT: ONNXExporter enforces the no-dummy guard
            # (we pass the REAL ``sample_input``), infers per-rank dynamic axes,
            # rejects complex inputs (ONNX has no complex dtype), and cleans up a
            # partial file on failure — replacing the inline torch.onnx.export.
            from spectramr.exports.onnx import ONNXExporter

            ONNXExporter(opset_version=17).export(
                model,
                onnx_path,
                input_sample=sample_input,
                input_names=["input"],
                output_names=["output"],
            )
            logger.info(f"  ONNX export successful: {onnx_path}")

        if args.format in ["torchscript", "both"]:
            ts_path = args.model.parent / f"{output_stem}.pt"
            logger.info(f"Exporting TorchScript to: {ts_path}")
            scripted = torch.jit.trace(model, sample_input)
            scripted.save(ts_path)
            logger.info(f"  TorchScript export successful: {ts_path}")

        return 0

    except Exception as e:
        logger.error(f"Error during export: {e}")
        return 1


def list_features(args: argparse.Namespace) -> int:
    """List available features (models, losses, metrics, strategies, physics, services)."""
    # ``tools/`` is a repo-root dev package, not part of the installed
    # ``spectramr`` distribution, so a bare ``from tools...`` only resolves when
    # CWD is the repo root. On a cluster job dir that ImportErrors. Locate the
    # repo root relative to this file and add it to sys.path so the command
    # works regardless of CWD.
    try:
        from tools.generate_feature_matrix import FeatureMatrixGenerator
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]  # …/cli/app.py → repo root
        tools_dir = repo_root / "tools"
        if not tools_dir.is_dir():
            logger.error(
                "list-features needs the repo-root 'tools/' package, which was "
                "not found (looked in %s). Run from a source checkout.",
                repo_root,
            )
            return 1
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.generate_feature_matrix import FeatureMatrixGenerator

    try:
        generator = FeatureMatrixGenerator()
        generator.generate()

        if args.module == "all":
            # Generate output for all modules
            if args.format == "markdown":
                output = generator.generate_markdown()
                if args.output:
                    output_path = Path(args.output)
                    generator.generate_markdown(output_path)
                    logger.info(f"Markdown saved to {output_path}")
                else:
                    print(output)
            elif args.format == "json":
                output = generator.generate_json()
                if args.output:
                    output_path = Path(args.output)
                    generator.generate_json(output_path)
                    logger.info(f"JSON saved to {output_path}")
                else:
                    print(output)
            elif args.format == "csv":
                output = generator.generate_csv()
                if args.output:
                    output_dir = Path(args.output)
                    output = generator.generate_csv(output_dir)
                logger.info(f"CSV files generated:\n{output}")
        else:
            # Generate for specific module
            features = generator.introspector.introspect_module(args.module)

            if args.format == "markdown":
                output = generator._create_markdown_table(
                    features, columns=["name", "description", "class_name"]
                )
                print(f"## {args.module.capitalize()} Features\n")
                print(output)
            elif args.format == "json":
                import json

                output = json.dumps([asdict(f) for f in features], indent=2)
                print(output)
            elif args.format == "csv":
                import csv

                writer = csv.DictWriter(
                    sys.stdout,
                    fieldnames=[
                        "name",
                        "class_name",
                        "module",
                        "description",
                    ],
                )
                writer.writeheader()
                for feature in features:
                    writer.writerow(
                        {
                            "name": feature.name,
                            "class_name": feature.class_name,
                            "module": feature.module,
                            "description": feature.description[:60],
                        }
                    )

        return 0

    except Exception as e:
        logger.error(f"Error listing features: {e}")
        import traceback

        traceback.print_exc()
        return 1


def audit(args: argparse.Namespace) -> int:
    """Tier 0+1 (+ optional Tier 2) audit of one experiment YAML or a folder.

    Single-file mode (positional is a file): full report — Spec Card,
    Tags, all checks, Recommendations.

    Bulk mode (positional is a directory): recursively globs ``*.yaml`` /
    ``*.yml`` (skipping ``archive/`` / ``.backup_*`` / ``inprogress_backup/``),
    runs the same audit on each, prints one compact line per file plus
    an aggregate summary at the end (totals, top-recurring AUDIT-Rxxx
    recommendation codes, tag distribution).

    Exit codes (single-file): 0 pass / 1 warnings / 2 errors. Bulk mode
    returns 2 if any file fails, 1 if any has warnings, 0 if all clean.

    Under ``--strict`` (the smoke-wrapper default), warnings → errors.
    """
    if args.config.is_dir():
        return _audit_bulk(args)
    return _audit_one(args, str(args.config))


def _gate_probe_acceleration(
    args: argparse.Namespace,
    config: Any,
    report: Any,
) -> Any:
    """Tier-2 acceleration gate: the probe must run on a real accelerator.

    The Tier-2 probe's headline value — catching CUDA OOM at the configured
    batch/patch size, and AMP / GradScaler double-unscale traps — **only exists
    on an accelerator**. A CPU probe exercises neither. So a green CPU probe
    certifies "did not crash on CPU", not "this arm will run on the GPU it was
    scheduled for": a facade in the sense of pitfall #16, at the hardware layer.
    (Historically ``--device`` defaulted to ``cpu`` here and the smoke wrapper
    never overrode it, so every Tier-2 probe ever run was the degraded kind.)

    Probe-device resolution: CLI ``--device`` > the config's own
    ``training.device`` > a **declared** ``run.device`` > ``auto``. Defaulting
    to the config's device means the probe exercises the device the *real run*
    will use, not an unrelated one.

    The ``run.device`` leg goes through :func:`spectramr.main._declared_device`
    rather than a plain attribute read, because ``run.device`` carries a schema
    default of ``"cuda"``. Reading the attribute made the chain resolve to
    ``"cuda"`` for *every* arm that declared no device, which left the ``auto``
    leg unreachable and made ``FORCE_CPU`` inert here: the documented
    "CPU, user dictated -> pass (degraded)" row could not be reached, because
    an explicit ``"cuda"`` is the one request ``FORCE_CPU`` may not relax. The
    former fourth leg read the top-level ``device`` attribute, which
    ``RENAMES`` retired with a raise posture -- it is not a schema field, so it
    was always ``None``.

    Returns the ``DeviceDecision``, or ``None`` when the probe is off or cannot
    be accelerated — in the latter case a failing check has been appended to
    ``report`` and the caller must skip the probe.
    """
    if not getattr(args, "probe", False):
        return None

    from spectramr.infrastructure.validation.config_health_checker import (
        HealthCheckResult,
    )
    from spectramr.main import _declared_device

    cli_device = getattr(args, "device", None)
    training_device = getattr(getattr(config, "training", None), "device", None)
    declared_device = _declared_device(config)
    requested = cli_device or training_device or declared_device
    if cli_device:
        source = "cli"
    elif training_device:
        source = "training.device"
    elif declared_device:
        source = "run.device"
    else:
        source = "auto"

    try:
        decision = resolve_torch_device(requested, pipeline="probe", source=source)
    except (AcceleratorRequiredError, ValueError) as exc:
        report.results.append(
            HealthCheckResult(
                passed=False,
                check_name="tier2_probe_accelerated",
                message=f"Tier-2 probe cannot run accelerated: {exc}",
                severity="error",
                category="device",
                yaml_keys=["device", "training.device"],
                fix_hint=(
                    "Run the audit on a GPU node — the probe must exercise the "
                    "device the arm will actually train on, or it cannot catch "
                    "OOM / AMP failures. To deliberately accept a degraded, "
                    "non-accelerated probe, pass `--device cpu` (or set "
                    "FORCE_CPU=true)."
                ),
            )
        )
        return None

    if decision.accelerated:
        message = f"Tier-2 probe runs accelerated on {decision.device} (source={decision.source})."
    else:
        # Reached only when the user explicitly dictated CPU. Deliberately NOT a
        # warning: warnings must be actionable, and "you asked for CPU" is not.
        # The degradation is still recorded so triage can tell the two kinds of
        # probe apart.
        message = (
            f"Tier-2 probe runs on CPU by explicit request "
            f"(source={decision.source}). OOM and AMP/GradScaler coverage are "
            "DISABLED — this probe cannot certify the arm will run on a GPU."
        )

    report.results.append(
        HealthCheckResult(
            passed=True,
            check_name="tier2_probe_accelerated",
            message=message,
            severity="info",
            category="device",
            yaml_keys=["device", "training.device"],
        )
    )
    return decision


def _audit_ledger_block(config: object, *, health_report: object = None) -> dict:
    """The ``_ledger`` block for the audit's own report.

    Produced by the same SSOT train uses, so the pre-flight and the run cannot
    describe the same config differently.
    """
    from spectramr.infrastructure.validation.resolved_config_artifact import (
        build_resolved_config_payload,
    )

    payload = build_resolved_config_payload(
        config, health_report=health_report, ledger_source="audit"
    )
    return payload["_ledger"]


def _audit_report(config_path: str, config: Any) -> Any:
    """The Tier-0/1 report both audit surfaces share: the witness ladder, bridged.

    The bulk run called ``validate_config_health`` directly and so never ran a
    witness that is not a health check (``schedule.*``, the budget, undersampling,
    validation-metric and strategy-dispatch witnesses): 647 arms reported clean on
    checks the single-arm audit fails them on (2026-09-03). One builder, two callers.
    """
    from spectramr.infrastructure.validation.audit_waivers import apply_audit_waivers
    from spectramr.infrastructure.validation.config_health_checker import (
        HealthCheckReport,
    )
    from spectramr.infrastructure.validation.witness import (
        Tier,
        WitnessSubject,
        run_witnesses,
    )
    from spectramr.infrastructure.validation.witness.checks.legacy_adapters import (
        verdict_to_health_result,
    )

    subject = WitnessSubject.for_audit(config_path=config_path, settings=config)
    verdicts = run_witnesses(subject, tiers=frozenset({Tier.T0, Tier.T1}))
    # An arm may acknowledge a warning its own physics makes unfixable, by name and
    # in writing, rather than the whole run reaching for --allow-warnings. Applied
    # here so both audit surfaces and the exit code see the same report.
    verdicts = apply_audit_waivers(verdicts, subject.raw_config)
    return HealthCheckReport(results=[verdict_to_health_result(v) for v in verdicts])


def _audit_one(args: argparse.Namespace, config_path: str) -> int:
    """Single-file audit — extracted from the old ``audit`` body."""
    import json

    from spectramr.config.settings import TrainingSettings

    # Arm the substitution ledger BEFORE the config is resolved. The audit
    # resolves through the same ``_finalize_from_dict`` path as training, so it
    # already computed every drop — it just had nowhere to put them, because the
    # audit's report goes to stdout and under ``--json`` to a pipe. The one
    # surface whose job is to catch problems before GPU time was the one leaving
    # no artifact.
    from spectramr.core.execution_ledger import ExecutionLedger

    ExecutionLedger.begin_run(source=config_path)

    try:
        config = TrainingSettings.from_yaml(config_path)
    except Exception as exc:
        record = {
            "tier": 0,
            "category": "schema",
            "passed": False,
            "severity": "error",
            "config_path": config_path,
            "message": str(exc),
            "fix_hint": "Fix the YAML so it validates against TrainingSettings (Pydantic v2).",
        }
        if args.json:
            print(json.dumps({"results": [record], "passed": False}, indent=2, default=str))
        else:
            from spectramr.cli._rendering import console

            console.print(f"[bold red]❌[/] [cyan]\\[schema][/] {exc}")
        return 2

    # One gate, both surfaces. This previously called `validate_config_health`
    # directly and never touched `ValidatorRegistry`, which only the train path
    # ran — so the audit could pass a config that `train` then rejected at
    # startup. Verdicts are bridged back to HealthCheckResult so the rendering,
    # --json payload and exit codes below are untouched.
    report = _audit_report(config_path, config)

    # Tier-1 compatibility matrix (cached-cascade WS-X): resolve the experiment
    # to its IR and run the pure compatibility rules; fold each message into the
    # report so it participates in pass/warn/error + rendering + exit code. This
    # is the Tier-1 wiring of the "components agree" guarantee — kept OUTSIDE the
    # Pydantic load (config-layer purity) and fail-soft so it never crashes audit.
    # The compatibility matrix now runs through the witness gate above
    # (legacy.compatibility_matrix); the inline loop that used to live here
    # would double-report every finding.

    # Spec Card: human-readable summary of what the experiment actually
    # operates on at each layer (data shape/domain, model capabilities,
    # losses, adapters). Phase 1 prints it; Phase 2 uses the same
    # derivation for hard cross-checks. Never blocks. See
    # docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md
    from spectramr.infrastructure.validation.spec_card import synthesize_spec_card

    try:
        spec_card_text = synthesize_spec_card(config)
    except Exception as _exc:  # never let card synthesis break audit
        spec_card_text = f"[SPEC] (card synthesis failed: {_exc})"

    # Phase 7: Tags + Recommendations. Tags describe the experiment;
    # recommendations are compiler-style design suggestions (AUDIT-Rxxx).
    # Both are info-level — they expand the report, never block.
    from spectramr.infrastructure.validation.inference import (
        collect_all_recommendations,
        derive_tags,
    )

    try:
        tags = derive_tags(config)
    except Exception as _exc:
        tags = []
    try:
        recommendations = collect_all_recommendations(config)
    except Exception as _exc:
        recommendations = []

    probe_record = None
    probe_decision = _gate_probe_acceleration(args, config, report)
    probe_device = probe_decision.device if probe_decision is not None else None

    if args.probe and probe_decision is not None:
        # The operator_id paradigm's conditioner returns a parameter dict, not
        # an image, so the generic image-shape probe does not apply — dispatch
        # to the paradigm-specific Tier-2 probe (adjoint identity, exp(Ω(0))=I,
        # finite operator action, non-zero NLL gradient). See
        # spectramr.infrastructure.validation.operator_id_probe.
        if getattr(getattr(config, "training", None), "equivariant_imaging", None) is not None:
            # Equivariant Imaging owns its objective (measurement consistency +
            # equivariance via context["transformed_recon"]); the generic loss
            # probe cannot supply that context. Dispatch to the paradigm-specific
            # mechanism-fires probe (numeric sensing_margin + non-zero equivariance
            # gradient on the configured model). See equivariant_imaging_probe.
            from spectramr.infrastructure.validation.equivariant_imaging_probe import (
                ei_forward_probe,
            )

            probe = ei_forward_probe(config, device=probe_device)
        elif getattr(getattr(config, "training", None), "operator_id", None) is not None:
            from spectramr.infrastructure.validation.operator_id_probe import (
                operator_id_forward_probe,
            )

            probe = operator_id_forward_probe(config, device=probe_device)
        elif (
            getattr(getattr(config, "training", None), "phys_residual_conformal", None) is not None
        ):
            # PR-CC is a post-hoc conformal certificate; the generic forward probe
            # cannot exercise the calibrate+certify mechanism. Dispatch to the
            # paradigm probe (phantom coverage guarantee + hallucination flagging +
            # param-map model contract). See phys_residual_conformal_probe.
            from spectramr.infrastructure.validation.phys_residual_conformal_probe import (
                phys_residual_conformal_probe,
            )

            probe = phys_residual_conformal_probe(config, device=probe_device)
        else:
            from spectramr.infrastructure.validation.forward_probe import (
                synthetic_forward_probe,
            )

            probe = synthetic_forward_probe(
                config,
                device=probe_device,
                backward=True,
                use_phantom=not args.noise,
                save_images_dir=(str(args.save_probe_images) if args.save_probe_images else None),
                arm_name=Path(config_path).stem,
            )
        probe_record = probe.to_dict()
        probe_record["tier"] = 2
        probe_record["config_path"] = config_path
        # Stamp the resolved device decision so the smoke/triage tooling can
        # tell a real (accelerated) probe from a degraded CPU one.
        probe_record["accelerated"] = probe_decision.accelerated
        probe_record["device_source"] = probe_decision.source

    if args.json:
        payload = {
            "config_path": config_path,
            "spec_card": spec_card_text,
            "tags": [
                {
                    "name": t.name,
                    "value": t.value,
                    "derived_from": list(t.derived_from),
                    "note": t.note,
                }
                for t in tags
            ],
            "recommendations": [
                {
                    "code": r.code,
                    "title": r.title,
                    "detail": r.detail,
                    "level": r.level.value,
                    "yaml_keys": list(r.yaml_keys),
                    "suggested_yaml": r.suggested_yaml,
                    "references": list(r.references),
                }
                for r in recommendations
            ],
            "tier_0_1": report.to_dict(),
            "tier_2": probe_record,
            # What the declaration lost on the way to the resolved settings.
            # Same block train stamps into resolved_config.json, produced by the
            # same SSOT so the two surfaces cannot describe it differently.
            "_ledger": _audit_ledger_block(config, health_report=report),
            "passed": report.passed and (probe_record is None or probe_record["passed"]),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        # Round-11 (2026-05-17): rich-console rendering. ANSI when TTY,
        # plain when piped — JSON path above is unaffected.
        from spectramr.cli._rendering import console

        console.print(spec_card_text)
        if tags:
            console.print()
            console.print("[bold]\\[TAGS][/]")
            for t in tags:
                src_anchor = f" ← {','.join(t.derived_from)}" if t.derived_from else ""
                console.print(f"  [cyan]{t.name:<24}[/] = {t.value}[dim]{src_anchor}[/]")
                if t.note:
                    console.print(f"  {'':<24}   [dim]{t.note}[/]")
        console.print()
        for r in report.results:
            console.print(r)  # uses HealthCheckResult.__rich__
        if probe_record is not None:
            if probe_record["passed"]:
                console.print(f"[green]✅[/] [cyan]\\[tier2_probe][/] {probe_record['message']}")
            else:
                console.print(f"[bold red]❌[/] [cyan]\\[tier2_probe][/] {probe_record['message']}")
                if probe_record.get("fix_hint"):
                    console.print(f"     [dim]fix:[/] [italic]{probe_record['fix_hint']}[/]")
        if recommendations:
            console.print()
            console.print(f"[bold]\\[RECOMMENDATIONS][/] {len(recommendations)} suggestion(s)")
            for r in recommendations:
                marker = {"hint": "💡", "suggestion": "📌", "lint": "🧹"}.get(r.level.value, "•")
                console.print(f"  {marker} [yellow]{r.code}[/]  [bold]{r.title}[/]")
                # Indent multi-line detail
                for line in r.detail.split("\n"):
                    console.print(f"      {line}")
                if r.yaml_keys:
                    console.print(f"      [dim]keys:[/] {', '.join(r.yaml_keys)}")
                if r.suggested_yaml:
                    console.print("      [dim]suggested YAML:[/]")
                    for line in r.suggested_yaml.rstrip().split("\n"):
                        console.print(f"        [italic]{line}[/]")
                if r.references:
                    console.print(f"      [dim]see:[/] {', '.join(r.references)}")

    # Deposit the artifact if asked. Unlike the training pipeline this does NOT
    # soft-fail: the audit exists to report, so an audit that cannot write its
    # own record has failed at its job and must say so rather than exit 0.
    if getattr(args, "write_resolved_config", None):
        from spectramr.infrastructure.validation.resolved_config_artifact import (
            write_resolved_config,
        )

        target = Path(args.write_resolved_config)
        target.mkdir(parents=True, exist_ok=True)
        written = write_resolved_config(target, config, health_report=report, ledger_source="audit")
        if not args.json:
            from spectramr.cli._rendering import console

            console.print(f"[dim]wrote {written}[/dim]")

    has_error = (not report.passed) or (probe_record is not None and not probe_record["passed"])
    # The probe half of the old disjunct was unreachable: it required
    # ``not probe_record["passed"]``, which already makes ``has_error`` true and
    # returns 2 below. A second, weaker owner of the same condition is what
    # non-negotiable 17 forbids -- and this one had never been audited as the
    # sole line of defence, because it never ran.
    has_warning = bool(report.warnings)
    if has_error:
        return 2
    if has_warning:
        return 2 if args.strict else 1
    return 0


def _bulk_counts(report: Any) -> tuple[int, int]:
    """(errors, warnings) of a health report, as the report itself defines them."""
    return len(getattr(report, "errors", []) or []), len(getattr(report, "warnings", []) or [])


def _excluded_by_patterns(rel_path: str, name: str, patterns: list[str]) -> bool:
    """Return True if ``rel_path`` or ``name`` matches any exclude pattern.

    Patterns use fnmatch glob syntax. A bare substring (no ``*``/``?``/``[``)
    is wrapped as ``*pattern*`` so ``--exclude ablation`` behaves intuitively.
    ``rel_path`` should use ``/`` separators (``Path.as_posix()``); matching it
    (not just the basename) is what lets ``*ablation*`` catch arms inside an
    ``ablations/`` subdirectory whose own filename has no "ablation" token.
    """
    import fnmatch

    for raw in patterns:
        pat = raw if any(c in raw for c in "*?[") else f"*{raw}*"
        if fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def _audit_bulk(args: argparse.Namespace) -> int:
    """Recursively audit every YAML under a directory.

    Compact per-file output line + aggregate summary at end. Skips
    backup / archive directories so the report stays focused on live
    experiments. ``--exclude PATTERN`` (repeatable) drops further YAMLs by
    path/filename glob — e.g. ``--exclude '*ablation*'`` to focus on training
    arms.
    """
    import json
    from collections import Counter
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.inference import (
        collect_all_recommendations,
        derive_tags,
    )

    root: Path = args.config
    # Skip directories whose YAMLs are not standalone runnable experiments:
    # archives, partial dataset/template snippets, manifest files. These
    # are reusable building blocks consumed by other YAMLs, not configs
    # that go through the full TrainingSettings schema.
    SKIP_PARTS = {
        "archive",
        "inprogress_backup",
        "templates",
        "datasets",
        "roadmap",
    }
    SKIP_PREFIXES = (".backup_", ".")

    excludes: list[str] = list(getattr(args, "exclude", None) or [])
    n_excluded = 0

    yamls: list[Path] = []
    for p in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if any(part.startswith(SKIP_PREFIXES) for part in p.parts):
            continue
        if excludes and _excluded_by_patterns(p.relative_to(root).as_posix(), p.name, excludes):
            n_excluded += 1
            continue
        yamls.append(p)

    if not yamls:
        print(f"No YAML files under {root}")
        return 0

    n_pass = n_warn = n_err = n_schema_err = 0
    rec_code_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    per_file: list[dict] = []
    rec_examples: dict[str, str] = {}  # code -> first file that triggered it

    # Round-11 (2026-05-17): rich-console rendering. ANSI when TTY,
    # plain when piped. JSON output path uses ``json.dumps`` directly
    # and is unaffected.
    from spectramr.cli._rendering import console

    console.print(f"[bold]# Bulk audit:[/] {len(yamls)} YAML(s) under {root}")
    if excludes:
        console.print(f"[dim]  --exclude {excludes} → skipped {n_excluded} YAML(s)[/]")
    console.print()

    for path in yamls:
        rel = str(path.relative_to(root))
        try:
            cfg = TrainingSettings.from_yaml(str(path))
        except Exception as exc:
            n_schema_err += 1
            n_err += 1
            console.print(f"[bold red]❌[/] {rel:<70}  [red]schema error:[/] {str(exc)[:80]}")
            per_file.append({"path": rel, "outcome": "schema_error", "error": str(exc)})
            continue

        try:
            report = _audit_report(str(path), cfg)
        except Exception as exc:
            n_schema_err += 1
            n_err += 1
            console.print(
                f"[bold red]❌[/] {rel:<70}  [red]audit report crashed:[/] {str(exc)[:80]}"
            )
            continue

        try:
            tags = derive_tags(cfg)
        except Exception:
            tags = []
        try:
            recs = collect_all_recommendations(cfg)
        except Exception:
            recs = []

        for t in tags:
            # Tag distribution counts: only summarize the fact that tag
            # is set, not all values (values are too varied to chart).
            if isinstance(t.value, bool) and t.value:
                tag_counts[t.name] += 1
            else:
                tag_counts[f"{t.name}={t.value}"] += 1
        for r in recs:
            rec_code_counts[r.code] += 1
            rec_examples.setdefault(r.code, rel)

        # The report owns the definition (a *failed* result of that severity):
        # counting severities here made a passed-with-warning advisory fail the
        # bulk run under --strict while the single-arm audit exited 0 (2026-09-03).
        n_e, n_w = _bulk_counts(report)
        outcome: str
        icon_markup: str
        outcome_style: str
        if n_e:
            n_err += 1
            outcome = "ERROR"
            icon_markup = "[bold red]❌[/]"
            outcome_style = "red"
        elif n_w:
            n_warn += 1
            outcome = "WARN" if not args.strict else "ERROR(strict)"
            if args.strict:
                icon_markup = "[bold red]❌[/]"
                outcome_style = "red"
                n_err += 1
            else:
                icon_markup = "[yellow]⚠[/] "
                outcome_style = "yellow"
        else:
            n_pass += 1
            outcome = "ok"
            icon_markup = "[green]✅[/]"
            outcome_style = "green"

        rec_marker = f"  [dim]+{len(recs)}rec[/]" if recs else ""
        console.print(f"{icon_markup} {rel:<70}  [{outcome_style}]{outcome:<13}[/]{rec_marker}")
        per_file.append(
            {
                "path": rel,
                "outcome": outcome,
                "n_errors": n_e,
                "n_warnings": n_w,
                "n_recommendations": len(recs),
                "tags": [{"name": t.name, "value": t.value} for t in tags],
                "recommendation_codes": [r.code for r in recs],
                "error_check_names": [
                    getattr(x, "check_name", "?")
                    for x in report.results
                    if getattr(x, "severity", "") == "error"
                ],
                "warning_check_names": [
                    getattr(x, "check_name", "?")
                    for x in report.results
                    if getattr(x, "severity", "") == "warning"
                ],
            }
        )

    # ── Aggregate summary ────────────────────────────────────────────
    console.print()
    console.print("[dim]" + ("─" * 80) + "[/]")
    console.print(
        f"[bold]SUMMARY[/]  total={len(yamls)}  "
        f"[green]✅ pass={n_pass}[/]  "
        f"[yellow]⚠ warn={n_warn}[/]  "
        f"[bold red]❌ err={n_err}[/]"
    )
    if n_schema_err:
        console.print(f"         [dim](of which schema errors: {n_schema_err})[/]")

    if tag_counts:
        console.print()
        console.print("[bold]\\[TAG DISTRIBUTION][/]  [dim](across all audited YAMLs)[/]")
        for name, count in tag_counts.most_common(20):
            console.print(f"  [cyan]{count:>4}[/]  {name}")

    if rec_code_counts:
        from spectramr.infrastructure.validation.recommendations import DIAGNOSTIC_CODES

        console.print()
        console.print(
            "[bold]\\[TOP RECOMMENDATIONS][/]  [dim](compiler-style design suggestions)[/]"
        )
        for code, count in rec_code_counts.most_common(15):
            title = DIAGNOSTIC_CODES.get(code, "(unknown code)")
            example = rec_examples.get(code, "")
            console.print(f"  [cyan]{count:>4}[/]  [yellow]{code}[/]  {title}")
            if example:
                console.print(f"        [dim]first seen in: {example}[/]")

    if args.json:
        payload = {
            "root": str(root),
            "total": len(yamls),
            "pass": n_pass,
            "warn": n_warn,
            "err": n_err,
            "schema_err": n_schema_err,
            "tag_counts": dict(tag_counts),
            "recommendation_counts": dict(rec_code_counts),
            "files": per_file,
        }
        # Print JSON below the human-readable summary so both modes
        # work for piping (`audit --dir X --json | jq ...`).
        print()
        print("[JSON]")
        print(json.dumps(payload, indent=2, default=str))

    if n_err:
        return 2
    if n_warn:
        return 2 if args.strict else 1
    return 0


def campaign_submit(args: argparse.Namespace) -> int:
    """Submit all experiments in a campaign."""
    from spectramr.infrastructure.orchestration.campaign_orchestrator import (
        CampaignOrchestrator,
    )

    base_dir = str(args.base_dir) if args.base_dir else "."
    # --only is a comma-separated list. --include / --exclude are repeatable
    # `key=value` selectors (key in {name, role, tag.<key>}).
    only = [s for raw in (args.only or []) for s in raw.split(",") if s]
    orchestrator = CampaignOrchestrator(
        base_dir=base_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        only=only,
        include=args.include or [],
        exclude=args.exclude or [],
        where=getattr(args, "where", "slurm") or "slurm",
    )
    try:
        state = orchestrator.submit_campaign(str(args.config))
        print(state.summary_table())
        return 0
    except Exception as e:
        logger.error(f"Campaign submission failed: {e}")
        return 1


def campaign_status(args: argparse.Namespace) -> int:
    """Check campaign progress."""
    from spectramr.infrastructure.orchestration.campaign_orchestrator import (
        CampaignOrchestrator,
    )

    base_dir = str(args.base_dir) if args.base_dir else "."
    orchestrator = CampaignOrchestrator(base_dir=base_dir, dry_run=args.dry_run)
    try:
        orchestrator.check_progress(str(args.campaign_dir))
        return 0
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return 1


def campaign_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a completed campaign."""
    from spectramr.config.schemas.campaign import (
        CampaignConfigSchema,
        EvaluationConfigSchema,
    )
    from spectramr.infrastructure.orchestration.campaign_evaluator import (
        CampaignEvaluator,
    )
    from spectramr.infrastructure.orchestration.campaign_report_generator import (
        CampaignReportGenerator,
    )
    from spectramr.infrastructure.orchestration.campaign_state import CampaignState

    try:
        state = CampaignState.load(Path(args.campaign_dir) / "campaign_state.json")

        # Load evaluation config from campaign YAML if available
        eval_config = EvaluationConfigSchema()
        if state.campaign_config_path:
            try:
                campaign_cfg = CampaignConfigSchema.from_yaml(state.campaign_config_path)
                eval_config = campaign_cfg.evaluation
            except Exception as e:
                logger.warning(f"Could not load campaign config for eval settings: {e}")

        evaluator = CampaignEvaluator(eval_config=eval_config)
        report = evaluator.evaluate(state)
        print(report.summary)

        # Generate plots if configured
        if eval_config.generate_plots:
            gen = CampaignReportGenerator(output_dir=args.campaign_dir)
            dashboard = gen.generate(report)
            print(f"\nDashboard: {dashboard}")

        # Export LaTeX if configured or --latex flag
        if args.latex or eval_config.export_latex:
            if report.leaderboard is not None:
                latex = CampaignEvaluator.leaderboard_to_latex(report.leaderboard)
                latex_path = Path(args.campaign_dir) / "leaderboard.tex"
                latex_path.write_text(latex)
                print(f"LaTeX table: {latex_path}")

        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def campaign_cancel(args: argparse.Namespace) -> int:
    """Cancel all active jobs in a campaign."""
    from spectramr.infrastructure.orchestration.campaign_orchestrator import (
        CampaignOrchestrator,
    )

    base_dir = str(args.base_dir) if args.base_dir else "."
    orchestrator = CampaignOrchestrator(base_dir=base_dir)
    try:
        orchestrator.cancel_campaign(str(args.campaign_dir))
        return 0
    except Exception as e:
        logger.error(f"Cancel failed: {e}")
        return 1


def campaign_watch(args: argparse.Namespace) -> int:
    """Watch campaign progress and auto-evaluate when complete.

    Polls SLURM at a configurable interval (default 300s / 5 min).
    When all experiments reach terminal state, automatically runs
    the full evaluation pipeline (statistics + plots + LaTeX).
    """
    import time as _time

    from spectramr.config.schemas.campaign import (
        CampaignConfigSchema,
        EvaluationConfigSchema,
    )
    from spectramr.infrastructure.orchestration.campaign_evaluator import (
        CampaignEvaluator,
    )
    from spectramr.infrastructure.orchestration.campaign_orchestrator import (
        CampaignOrchestrator,
    )
    from spectramr.infrastructure.orchestration.campaign_report_generator import (
        CampaignReportGenerator,
    )

    base_dir = str(args.base_dir) if args.base_dir else "."
    poll_interval = args.poll_interval
    orchestrator = CampaignOrchestrator(base_dir=base_dir)

    logger.info(f"👁️  Watching campaign: {args.campaign_dir}")
    logger.info(f"   Poll interval: {poll_interval}s")
    logger.info("   Press Ctrl+C to stop watching\n")

    try:
        while True:
            state = orchestrator.check_progress(str(args.campaign_dir))

            if state.all_terminal:
                n_completed = len(state.completed)
                n_failed = len(state.failed)
                logger.info(
                    f"\n🏁 All experiments terminal: {n_completed} completed, {n_failed} failed"
                )

                if n_completed < 2:
                    logger.warning(
                        "Fewer than 2 experiments completed — "
                        "skipping evaluation (need ≥2 for comparison)"
                    )
                    return 0

                # Auto-evaluate
                logger.info("\n📊 Starting automated evaluation...\n")
                eval_config = EvaluationConfigSchema()
                if state.campaign_config_path:
                    try:
                        campaign_cfg = CampaignConfigSchema.from_yaml(state.campaign_config_path)
                        eval_config = campaign_cfg.evaluation
                    except Exception as e:
                        logger.warning(f"Could not load campaign config for eval: {e}")

                evaluator = CampaignEvaluator(eval_config=eval_config)
                report = evaluator.evaluate(state)
                print(report.summary)

                if eval_config.generate_plots:
                    gen = CampaignReportGenerator(output_dir=args.campaign_dir)
                    dashboard = gen.generate(report)
                    print(f"\nDashboard: {dashboard}")

                if eval_config.export_latex and report.leaderboard is not None:
                    latex = CampaignEvaluator.leaderboard_to_latex(report.leaderboard)
                    latex_path = Path(args.campaign_dir) / "leaderboard.tex"
                    latex_path.write_text(latex)
                    print(f"LaTeX table: {latex_path}")

                return 0

            # Not all terminal — sleep and poll again
            progress = state.progress_fraction
            logger.info(f"  ⏳ Progress: {progress:.0%} — next poll in {poll_interval}s\n")
            _time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("\n👋 Watch cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Watch failed: {e}")
        return 1


def hpo_cmd(args: argparse.Namespace) -> int:
    """Run hyperparameter optimization via the HPOUseCase.

    Each trial spawns a separate ``python -m spectramr.cli train`` subprocess
    so a single trial crash (NaN, OOM, lib mismatch) doesn't poison the
    Optuna study state.

    Also handles three discovery modes that exit early without running HPO:
      --list-presets       → list built-in search space names
      --list-schema-paths  → list every dotted path TrainingSettings exposes
      --print-template     → emit a starter YAML for --search-space

    Returns 0 on success, 1 on failure.
    """
    from spectramr.application.use_cases.hpo_use_case import HPORequest, HPOUseCase
    from spectramr.infrastructure.logging import LoggingService
    from spectramr.pipelines.hpo_search_spaces import (
        enumerate_schema_paths,
        list_presets,
        make_search_space_template,
    )

    # Discovery / introspection modes — exit early without launching HPO.
    if args.list_presets:
        print("Available search-space presets:")
        for name in list_presets():
            print(f"  {name}")
        print("\nUsage: --search-preset <name>")
        return 0
    if args.list_schema_paths:
        print("Tunable dotted-path fields exposed by TrainingSettings:")
        print("(use any of these as keys in a --search-space YAML)\n")
        for p in enumerate_schema_paths():
            print(f"  {p}")
        return 0
    if args.print_template:
        print(make_search_space_template())
        return 0

    if args.search_preset and args.search_space:
        print(
            "ERROR: --search-preset and --search-space are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    # Resolve --search-preset (which is now repeatable) into either a single
    # preset name (back-compat with HPORequest.search_preset) or a merged
    # SearchSpace dict (when multiple presets given).
    preset_names = list(args.search_preset) if args.search_preset else []
    search_space_dict = None
    search_preset_name: str | None = None
    if len(preset_names) == 1:
        search_preset_name = preset_names[0]
    elif len(preset_names) > 1:
        # Merge here so the use case sees a single concrete spec rather than
        # a list — keeps HPORequest's contract simple.
        from spectramr.pipelines.hpo_search_spaces import load_presets

        try:
            merged = load_presets(preset_names, on_conflict=args.preset_merge_policy)
        except ValueError as e:
            print(f"ERROR: preset merge failed: {e}", file=sys.stderr)
            print(
                "Hint: pass --preset-merge-policy override (later wins) or "
                "keep (first wins) to resolve explicitly.",
                file=sys.stderr,
            )
            return 1
        # Convert SearchSpace -> dict so HPORequest.search_space_dict can carry it
        search_space_dict = {
            path: {
                "dist": dist.kind,
                **({"low": dist.low, "high": dist.high} if dist.kind != "categorical" else {}),
                **({"choices": dist.choices} if dist.kind == "categorical" else {}),
            }
            for path, dist in merged.params.items()
        }

    pruner = args.pruner if args.pruner != "none" else "nop"
    request = HPORequest(
        config_path=str(args.config),
        model_types=list(args.model_type),
        n_trials=args.n_trials,
        objective_metric=args.objective_metric,
        device=args.device,
        max_iter_per_trial=args.max_iter,
        sampler_type=args.sampler,
        pruner_type=pruner,
        storage_url=args.storage,
        output_dir=str(args.output_dir) if args.output_dir else "",
        cost_weight=args.cost_weight,
        enable_cost_optimization=args.multi_objective,
        search_preset=search_preset_name,
        search_space_path=str(args.search_space) if args.search_space else None,
        search_space_dict=search_space_dict,
    )

    logging_service = LoggingService()
    use_case = HPOUseCase(logging_service)
    response = use_case.execute(request)

    if not response.success:
        print("HPO run failed — see logs for details", file=sys.stderr)
        return 1

    print("\n=== HPO Results ===")
    for model_type, result in response.results.items():
        status = result.get("status", "?")
        print(f"\nModel: {model_type}  (status: {status})")
        if status != "completed":
            err = result.get("error", "(no error message)")
            print(f"  error: {err}")
            continue
        n_trials = result.get("n_trials", "?")
        best_value = result.get("best_value", "?")
        best_params = result.get("best_params", {})
        print(f"  n_trials  : {n_trials}")
        print(f"  best_value: {best_value}")
        if best_params:
            print("  best_params:")
            for k, v in best_params.items():
                print(f"    {k}: {v}")
    return 0


def report(args: argparse.Namespace) -> int:
    """Generate canonical figures + tables for an experiment output dir.

    Runs the same orchestrator the end-of-training hook uses, so the output is
    identical whether training triggers it or you invoke it manually on a
    downloaded run. Pass ``--config`` to reuse that YAML's ``reporting:`` block
    (style / formats / dpi / figures / tables / QC / HTML) verbatim — full
    parity with the training hook. Explicit CLI flags override the config.
    """
    from spectramr.infrastructure.reporting import generate_report

    exp_dir: Path = args.exp_dir
    if not exp_dir.exists():
        print(f"error: experiment dir does not exist: {exp_dir}")
        return 2
    cohort: dict | None = None
    if args.cohort_json is not None:
        import json

        cohort = json.loads(Path(args.cohort_json).read_text())

    # Base kwargs: from a --config reporting block when given (parity with the
    # training hook), else the plain CLI defaults.
    kwargs: dict = {
        "task": "default",
        "method_name": None,
        "out_subdir": "report",
        "seed": args.seed,
        "dataset_version": args.dataset_version,
    }
    if args.config is not None:
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings.from_yaml(str(args.config))
        rep = getattr(settings, "reporting", None)
        if rep is not None:
            _v = lambda x: getattr(x, "value", x)  # noqa: E731 — enum→str unwrap
            kwargs.update(
                task=_v(getattr(rep, "task", "default")),
                style=_v(getattr(rep, "style", "nature")),
                formats=tuple(getattr(rep, "formats", ["pdf", "png"])),
                dpi=getattr(rep, "dpi", 600),
                panel_labels=getattr(rep, "panel_labels", True),
                method_name=getattr(rep, "method_name", None),
                # NOTE: reporting.figures is deliberately NOT inherited here — the
                # `report` command is the plotting SSOT and renders ALL applicable
                # figures by default (data-less ones soft-skip). Use --figures to
                # opt into a subset. The end-of-training hook still honours
                # reporting.figures per-run (that is a deliberate per-run scope).
                tables_=getattr(rep, "tables", None),
                metrics=getattr(rep, "metrics", None),
                hyperparameters=getattr(rep, "hyperparameters", None),
                extra_runs=getattr(rep, "extra_runs", None),
                out_subdir=getattr(rep, "out_subdir", "report"),
                emit_manifest=getattr(rep, "emit_manifest", True),
                submission_bundle=getattr(rep, "submission_bundle", False),
                tikz=getattr(rep, "tikz", False),
                qc_figures=getattr(rep, "qc_figures", True),
                html_report=getattr(rep, "html_report", True),
                interactive=getattr(rep, "interactive", True),
                cohort=getattr(rep, "cohort", None),
            )
        if kwargs.get("seed") is None:
            # ``settings.seed`` is the LEGACY top-level spelling; ``RENAMES``
            # retired it to ``run.seed`` (2026-07-31) with a raise posture, so
            # this getattr could never resolve and the report's provenance
            # carried ``seed: null`` on every run that did not pass --seed
            # (pitfall #15). ``run.seed`` always resolves (schema default 42),
            # which is precisely the seed the run actually used.
            kwargs["seed"] = settings.run.seed

    # Explicit CLI overrides win over the config block.
    if args.task is not None:
        kwargs["task"] = args.task
    if args.out_subdir is not None:
        kwargs["out_subdir"] = args.out_subdir
    if args.method is not None:
        kwargs["method_name"] = args.method
    if cohort is not None:
        kwargs["cohort"] = cohort
    if args.html is not None:
        kwargs["html_report"] = args.html
    if args.qc is not None:
        kwargs["qc_figures"] = args.qc
    if getattr(args, "interactive", None) is not None:
        kwargs["interactive"] = args.interactive
    if getattr(args, "figures", None):
        kwargs["figures"] = [f.strip() for f in args.figures.split(",") if f.strip()]

    # Batch mode: --exp-dir is a cohort root; report every run under it and
    # write a linking report_index.html. Per-run failures don't abort the batch.
    if getattr(args, "recursive", False):
        from spectramr.infrastructure.reporting import generate_reports

        kwargs.pop("method_name", None)  # per-run name applied inside
        batch = generate_reports(exp_dir, **kwargs)
        if batch["n_runs"] == 0:
            print(f"No run directories found under {exp_dir}")
            return 0
        print(f"Generated {batch['n_ok']}/{batch['n_runs']} report(s) under {exp_dir}")
        if batch["index"] is not None:
            print(f"Index: {batch['index']}")
        return 0 if batch["n_ok"] == batch["n_runs"] else 1

    kwargs["method_name"] = kwargs.get("method_name") or exp_dir.name
    result = generate_report(exp_dir, **kwargs)
    n_fig = sum(1 for v in result["figures"].values() if v is not None)
    n_tab = sum(1 for v in result["tables"].values() if v is not None)
    print(f"Wrote {n_fig} figure(s) and {n_tab} table(s) to {result['out_dir']}")
    print(f"Summary: {result['summary']}")
    if result.get("html") is not None:
        print(f"HTML report: {result['html']}")
    return 0


def _synthesize_phantom_volumes(n: int, seed: int) -> "list[tuple[str, 'torch.Tensor']]":
    """Build deterministic phantom slices for smoke runs (no real data)."""
    import torch

    volumes: list[tuple[str, torch.Tensor]] = []
    gen = torch.Generator().manual_seed(seed)
    for i in range(n):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 64),
            torch.linspace(-1, 1, 64),
            indexing="ij",
        )
        r = torch.sqrt(xx**2 + yy**2)
        phantom = torch.exp(-(r**2) / 0.5) + 0.3 * torch.randn((64, 64), generator=gen).abs()
        volumes.append((f"phantom_{i}", phantom.float().unsqueeze(0)))
    return volumes


def _load_clean_volumes_from_pt(
    input_dir: Path,
    max_subjects: int,
    device: "torch.device | None" = None,
) -> "list[tuple[str, 'torch.Tensor']]":
    """Load `.pt` clean references.

    Delegates to :class:`spectramr.data.datasets.meta_eval_clean_refs.MetaEvalCleanRefsDataset`
    so the file → tensor transition lives in the data SSOT subtree
    (CLAUDE.md pitfall #11). Phase 4 of
    ``TODO/backlog_ssot_and_layering_cleanup.md``.
    """
    from spectramr.data.datasets.meta_eval_clean_refs import load_clean_volumes

    return load_clean_volumes(input_dir, max_subjects=max_subjects, device=device)


def _load_clean_volumes_from_h5(
    input_dir: Path,
    max_subjects: int,
    max_contrasts: int = 3,
    contrasts: tuple[str, ...] | None = None,
    device: "torch.device | None" = None,
    assets_out: "dict[str, object] | None" = None,
) -> "list[tuple[str, 'torch.Tensor']]":
    """Synthesize pseudo-GT magnitude volumes from M4Raw multi-rep H5 files.

    M4Raw stores files as ``<subject>_<contrast><rep>.h5`` (e.g.
    ``2022091411_FLAIR01.h5``). The same grouping convention the M4Raw
    training dataset uses — see
    :class:`spectramr.data.datasets.m4raw_dataset.M4RawRepetitionDataset` —
    strips the last 2 chars (rep index) so each group is one
    ``(subject, contrast)`` pair. This makes meta-evaluation
    multi-contrast by default (T1 / T2 / FLAIR), matching how the
    rest of the pipeline treats M4Raw.

    Per-group pseudo-GT is built with
    :func:`scripts.sim2rank.ground_truth.synthesize_pseudo_gt`
    (averages reps in k-space, SENSE-adjoint-combines) — the same
    path used by ``run_meta_eval_all_datasets`` and the sim2rank
    engine.

    Args:
        input_dir: Directory of M4Raw H5 files.
        max_subjects: Distinct subjects to load.
        max_contrasts: Contrasts per subject (defaults to 3 → T1, T2,
            FLAIR; matches sim2rank's CLI default).
        contrasts: Optional whitelist (e.g. ``("T1", "T2")``); applied
            against the parsed contrast token. ``None`` keeps all.

    Returns:
        ``[(id, tensor), ...]`` where ``id`` is ``"<subject>_<contrast>"``
        so downstream figures can stratify per contrast.
    """
    import torch

    from spectramr.data.datasets.m4raw_repetition_groups import (
        discover_repetition_groups,
    )
    from spectramr.infrastructure.physics.m4raw_pseudo_gt import synthesize_pseudo_gt

    rep_groups = discover_repetition_groups(
        input_dir,
        max_subjects=max_subjects,
        max_contrasts=max_contrasts,
    )

    contrast_whitelist = {c.upper() for c in contrasts} if contrasts is not None else None

    # ESPIRiT inside synthesize_pseudo_gt is the heavy step; running it on
    # CUDA saves the bulk of the per-subject cost. We pass ``device`` so the
    # pseudo-GT pipeline does the SVD on the GPU rather than copying around.
    pseudo_gt_device = device if device is not None else torch.device("cpu")

    volumes = []
    for group in rep_groups:
        # Group key: e.g. "2022091411_FLAIR01" → strip 2 → "2022091411_FLAIR".
        base = group[0].name.split(".")[0]
        tag = base[:-2] if len(base) >= 3 else base
        parts = tag.split("_")
        contrast = parts[1].upper() if len(parts) >= 2 else "UNKNOWN"
        if contrast_whitelist is not None and contrast not in contrast_whitelist:
            logger.debug("Skipping %s — contrast %r not in whitelist.", tag, contrast)
            continue
        try:
            coil_images, smaps, x_gt_mag, p99 = synthesize_pseudo_gt(group, device=pseudo_gt_device)
        except Exception as exc:
            logger.warning("Pseudo-GT failed for group %s: %s — skipping.", tag, exc)
            continue
        # x_gt_mag: (1, 1, H, W) → drop batch axis, keep (1, H, W).
        # Stay on ``device`` if requested — keeps the simulator/metric work
        # on the same accelerator without per-sample host↔device copies.
        clean = x_gt_mag.squeeze(0).float()
        if device is not None:
            clean = clean.to(device)
        else:
            clean = clean.cpu()
        volumes.append((tag, clean))
        # Prepare the REAL acquisition assets from the multi-coil reference the
        # pseudo-GT already produced, so the NR/physics metric battery is graded
        # on the true measurement rather than a synthetic birdcage stand-in.
        if assets_out is not None:
            try:
                from spectramr.infrastructure.physics.asset_preparation import (
                    prepare_metric_context,
                )

                assets_out[tag] = prepare_metric_context(
                    coil_images=coil_images, smaps=smaps, p99=p99
                )
            except Exception as exc:  # never let asset prep break the load
                logger.warning("Asset prep failed for %s: %s — synth fallback.", tag, exc)
    return volumes


def _load_clean_volumes(
    input_dir: Path,
    max_subjects: int,
    max_contrasts: int = 3,
    contrasts: tuple[str, ...] | None = None,
    device: "torch.device | None" = None,
    assets_out: "dict[str, object] | None" = None,
) -> "list[tuple[str, 'torch.Tensor']]":
    """Dispatch by file extension. .pt → direct load; .h5 → pseudo-GT.

    ``assets_out``, when provided, is populated with ``{content_id:
    MetricContext}`` of the real acquisition assets for .h5 (multi-coil)
    inputs — the .pt path carries no coil data, so it stays empty there.

    Raises:
        FileNotFoundError: ``input_dir`` does not exist.
        RuntimeError: directory exists but yields no usable volumes.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"--input directory not found: {input_dir}. "
            "Point at a directory of .pt clean references, a directory of "
            "M4Raw .h5 files (databases/m4raw/data/multicoil_train), or "
            "omit --input for the synthetic-phantom smoke mode."
        )
    pt_files = sorted(input_dir.glob("*.pt"))
    h5_files = sorted(input_dir.glob("*.h5")) or sorted(input_dir.rglob("*.h5"))
    if pt_files:
        volumes = _load_clean_volumes_from_pt(input_dir, max_subjects, device=device)
    elif h5_files:
        volumes = _load_clean_volumes_from_h5(
            input_dir,
            max_subjects=max_subjects,
            max_contrasts=max_contrasts,
            contrasts=contrasts,
            device=device,
            assets_out=assets_out,
        )
    else:
        raise RuntimeError(
            f"--input={input_dir} contains no .pt or .h5 files. "
            "Run `scripts/sim2rank/run_meta_eval_all_datasets.py` to materialize "
            "clean refs, or point --input at databases/m4raw/data/multicoil_train."
        )
    if not volumes:
        raise RuntimeError(
            f"--input={input_dir} matched files but produced zero usable "
            f"tensors (checked {len(pt_files)} .pt and {len(h5_files)} .h5). "
            "Refusing to silently fall back to synthetic phantoms — see "
            "CLAUDE.md pitfall #9."
        )
    return volumes


def _resolve_device(spec: str | None, *, pipeline: str = "evaluate") -> "torch.device":
    """Resolve the ``--device`` CLI string into a concrete ``torch.device``.

    Thin shell over the SSOT policy in :mod:`spectramr.core.compute_device`.
    ``auto`` picks CUDA when available; for a heavy ``pipeline`` it RAISES when
    no accelerator exists rather than downgrading to CPU, so an sbatch that lost
    its GPU allocation fails loudly instead of burning the wall-clock budget at
    ~100x slowdown (CLAUDE.md pitfalls #9, #10).
    """
    import torch

    decision = resolve_torch_device(spec, pipeline=pipeline, source="cli")
    return torch.device(decision.device)


def _meta_evaluate_full_harness(
    args: argparse.Namespace,
    clean_volumes: list,
    eval_mode: str,
) -> int:
    """Run the full §8 NR-metric validation harness and emit per-metric cards.

    Driven by ``meta-evaluate --tiers ...``. Delegates to
    :class:`NRMetricValidationUseCase.run_validation` (the application-layer
    orchestrator) and writes the cards / ranking / redundancy / aggregator block
    to ``--out``. Research-mode: this is HOW the NR battery is validated.
    """
    import json

    from spectramr.application.use_cases.nr_metric_validation_use_case import (
        DEFAULT_NR_BATTERY,
        HARNESS_TIERS,
        NRMetricValidationConfig,
        NRMetricValidationUseCase,
    )

    raw_tiers = list(args.tiers) if args.tiers else list(HARNESS_TIERS)
    if any(t == "all" for t in raw_tiers):
        tiers = tuple(HARNESS_TIERS)
    else:
        unknown = [t for t in raw_tiers if t not in HARNESS_TIERS]
        if unknown:
            logger.error(
                "unknown --tiers value(s) %s; valid: %s (or 'all')",
                unknown,
                list(HARNESS_TIERS),
            )
            return 1
        tiers = tuple(raw_tiers)

    metrics = tuple(args.metrics) if args.metrics else DEFAULT_NR_BATTERY
    config = NRMetricValidationConfig(
        metrics=metrics,
        n_severities_per_family=args.severities,
        seed=args.seed,
        eval_mode=eval_mode,
        tiers=tiers,
        register_aggregator=bool(getattr(args, "register_aggregator", False)),
    )
    logger.info(
        "NR validation harness: tiers=%s metrics=%d (RESEARCH-MODE, "
        "validation-pending — not for training)",
        list(tiers),
        len(metrics),
    )
    result = NRMetricValidationUseCase(config).run_validation(clean_volumes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2, default=float)
    logger.info("Wrote NR validation cards to %s", args.out)

    # §8.8 reporting: cards CSV + markdown (+ figures) alongside the JSON.
    if getattr(args, "tables", False) or getattr(args, "figures", False):
        from spectramr.application.use_cases.nr_validation.report import (
            write_nr_validation_report,
        )

        report_dir = args.tables_dir or args.figures_dir or args.out.parent
        report_paths = write_nr_validation_report(
            result, report_dir, figures=bool(getattr(args, "figures", False))
        )
        logger.info("Wrote %d NR validation report file(s) to %s", len(report_paths), report_dir)
        for p in report_paths:
            logger.info("  - %s", p)

    # Compact console card summary: metric -> per-tier pass/fail + overall.
    card_summary = {
        m: {
            "overall_pass": card["overall_pass"],
            "tiers": {t: v["passed"] for t, v in card["tiers"].items()},
        }
        for m, card in result["cards"].items()
    }
    print(
        json.dumps(
            {
                "research_mode": True,
                "tiers_run": result["tiers_run"],
                "n_samples": result["n_samples"],
                "cards": card_summary,
                "redundancy_kept": (result["redundancy"] or {}).get("kept"),
                "aggregator_backend": (result["aggregator"] or {}).get("backend"),
            },
            indent=2,
            default=float,
        )
    )
    return 0


def meta_evaluate(args: argparse.Namespace) -> int:
    """Run the five-method meta-evaluation against a chosen metric set."""
    import json

    from spectramr.core.metrics import list_available
    from spectramr.core.metrics.meta_evaluation import (
        MetaEvaluationPipeline,
        MetricSet,
        SimulatorConfig,
        resolve_eval_mode,
    )

    # Validate the evaluation dimensionality FIRST — '3d' must fail loud before
    # we spend any time loading volumes / estimating coil maps (pitfall #15).
    eval_mode = resolve_eval_mode(getattr(args, "eval_mode", "2d"))

    device = _resolve_device(getattr(args, "device", "auto"))
    logger.info("meta-evaluate runtime device: %s", device)

    # Strict loader contract:
    #   --input omitted          → deterministic synthetic phantoms (smoke).
    #   --input set & missing    → FileNotFoundError (don't mask the typo).
    #   --input set & empty      → RuntimeError       (don't mask missing data).
    # See CLAUDE.md pitfalls #9 (no silent fallbacks) and #10 (warnings ≠ ok).
    # Real acquisition assets, keyed by content_id, populated from multi-coil
    # .h5 inputs so the NR/physics battery is graded on the true measurement.
    real_assets: dict[str, object] = {}
    if args.input is not None:
        contrasts = tuple(args.contrasts) if args.contrasts else None
        clean_volumes = _load_clean_volumes(
            args.input,
            max_subjects=args.max_subjects,
            max_contrasts=args.max_contrasts,
            contrasts=contrasts,
            device=device,
            assets_out=real_assets,
        )
        logger.info(
            "Loaded %d clean reference volume(s) from %s (max_subjects=%d, max_contrasts=%d)",
            len(clean_volumes),
            args.input,
            args.max_subjects,
            args.max_contrasts,
        )
    else:
        logger.info(
            "--input omitted; synthesizing %d phantom slices (smoke mode).",
            args.synthetic_n,
        )
        clean_volumes = _synthesize_phantom_volumes(args.synthetic_n, args.seed)
        if device.type != "cpu":
            clean_volumes = [(tag, t.to(device)) for tag, t in clean_volumes]

    # ``--tiers`` routes to the FULL §8 validation harness (per-metric cards
    # across the label-free tiers + aggregator), not the bare ranking. It implies
    # NR-battery semantics and reuses the same loaded clean references.
    tiers_arg = getattr(args, "tiers", None)
    if tiers_arg is not None:
        return _meta_evaluate_full_harness(args, clean_volumes, eval_mode)

    # Resolve the metric set. ``--nr-battery`` defaults to the label-free
    # no-reference battery and drives the sweep with the audited digital-twin
    # degradations (spec §8.1); otherwise a curated short list.
    nr_battery = bool(getattr(args, "nr_battery", False))
    if args.metrics:
        requested = args.metrics
    elif nr_battery:
        from spectramr.application.use_cases.nr_metric_validation_use_case import (
            DEFAULT_NR_BATTERY,
        )

        requested = list(DEFAULT_NR_BATTERY)
    else:
        requested = ["psnr", "ssim", "rmse", "mae"]
    available = set(list_available())
    unknown = [n for n in requested if n not in available]
    if unknown:
        # No-silent-fallback (CLAUDE.md #9; mirrors the --tiers handling above):
        # a typo'd --metrics name must abort, not silently yield a smaller
        # ranking that looks like a successful run.
        raise ValueError(
            f"--metrics contains unregistered metric(s): {sorted(unknown)}; "
            f"available: {sorted(available)}"
        )
    valid = list(requested)

    # Wrap every metric with the safe-call adapter so registry-quirks
    # (4-D-only conv2d, multi-return tuples, dependency-missing crashes)
    # don't take down the run. Crashing metrics show up as NaN rows in
    # the CSVs / figures rather than aborting the pipeline.
    from spectramr.core.metrics.meta_evaluation.metric_adapter import (
        build_safe_metric_set,
    )

    metric_set = MetricSet(**build_safe_metric_set(valid))

    pipeline = MetaEvaluationPipeline.from_defaults()
    # Degradation SSOT (critique C2): production meta-evaluation injects the audited
    # digital-twin degradation registry (the physics SSOT) as the simulator operator
    # library via the dependency-inversion seam, so EVERY run — bare or --nr-battery —
    # is scored against the SAME physically-faithful, (theta, seed)-deterministic
    # corruptions. The dependency-free core ``DEGRADATION_LIBRARY`` (8 simpler ops) is
    # only used when ``--core-degradations`` is set (e.g. a physics-less environment),
    # never silently mixed with the physics ops. The resolved library is stamped into
    # the run summary (``degradation_provenance``).
    use_core = bool(getattr(args, "core_degradations", False))
    sim_families = None
    operator_library = None
    if not use_core:
        from spectramr.application.use_cases.nr_metric_validation_use_case import (
            DEFAULT_FAMILIES,
            build_physics_operator_library,
        )

        sim_families = list(DEFAULT_FAMILIES)
        operator_library = build_physics_operator_library(sim_families)
        logger.info("degradation SSOT: digital-twin families = %s", sim_families)
    else:
        logger.info("degradation SSOT: dependency-free core library (--core-degradations)")
    pipeline.simulator_config = SimulatorConfig(
        families=sim_families,
        n_severities_per_family=args.severities,
        seed=args.seed,
        operator_library=operator_library,
    )
    # Wire the run-shape knobs (each read → validated → stamped, pitfall #15).
    pipeline.eval_mode = eval_mode
    pipeline.use_betting = bool(getattr(args, "betting", False))
    pipeline.betting_alpha = float(getattr(args, "betting_alpha", 0.05))
    if not 0.0 < pipeline.betting_alpha < 1.0:
        raise ValueError(f"--betting-alpha must be in (0, 1), got {pipeline.betting_alpha}")
    # Cross-axis aggregator: "z" (Pareto-only z-weighted, default) or "kemeny"
    # (Condorcet-consistent L3⁺). Validated against the set the pipeline accepts so
    # a typo fails loud here, not deep in run() (pitfall #9).
    aggregator = str(getattr(args, "aggregator", "z"))
    if aggregator not in ("z", "kemeny"):
        raise ValueError(f"--aggregator must be 'z' or 'kemeny', got {aggregator!r}")
    pipeline.aggregator = aggregator
    # Pool hygiene: screen clones / broken metrics out of the voting pool.
    pipeline.screen_pool_enabled = bool(getattr(args, "screen_pool", False))
    logger.info(
        "meta-evaluate run shape: eval_mode=%s aggregator=%s screen_pool=%s betting=%s%s",
        pipeline.eval_mode,
        pipeline.aggregator,
        pipeline.screen_pool_enabled,
        pipeline.use_betting,
        f" (alpha={pipeline.betting_alpha})" if pipeline.use_betting else "",
    )
    output = pipeline.run(metric_set, clean_volumes, assets_by_content=real_assets or None)

    summary = output.to_summary()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Wrote meta-evaluation summary to %s", args.out)

    if getattr(args, "figures", False):
        from spectramr.core.metrics.meta_evaluation import render_figures

        fig_dir = args.figures_dir or args.out.parent / "figures"
        written = render_figures(output, fig_dir)
        logger.info("Rendered %d figure(s) to %s", len(written), fig_dir)
        for p in written:
            logger.info("  - %s", p)

    if getattr(args, "tables", False):
        from spectramr.core.metrics.meta_evaluation import write_tables

        tbl_dir = args.tables_dir or args.out.parent / "tables"
        written_tables = write_tables(output, tbl_dir)
        logger.info("Wrote %d CSV table(s) to %s", len(written_tables), tbl_dir)
        for p in written_tables:
            logger.info("  - %s", p)

    print(
        json.dumps(
            {
                "eval_mode": summary["eval_mode"],
                "betting": summary["betting"],
                "final_ranks": summary["final_ranks"],
                "final_scores": summary["final_scores"],
                "defensibility_flags": summary["defensibility_flags"],
            },
            indent=2,
            default=float,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``spectramr`` argument parser (every subcommand).

    Extracted from :func:`main` so the parser can be built and introspected in
    tests WITHOUT executing a command — e.g. asserting that every ``launch
    --pipeline`` verb actually accepts the ``--config`` flag the launcher
    injects (``spectramr launch <cfg> --pipeline <verb>`` would otherwise fail at
    argparse time for a verb whose config is positional / absent).

    Kept deliberately torch-free: only argparse + stdlib and the lightweight
    ``attach_subparsers`` shims run here, so ``--help`` and argument-error
    paths never import torch or the model catalogue. The colored
    console-logging handler (which imports torch transitively) is installed by
    :func:`main` only once a real subcommand is dispatched. This function is
    also the seam the startup-budget regression test exercises.
    """
    parser = argparse.ArgumentParser(
        prog="spectramr",
        description="spectraMR: Medical Image Super-Resolution and Reconstruction",
    )
    from spectramr.cli.diagnostics import _version as _spectramr_version

    parser.add_argument(
        "--version",
        action="version",
        version=f"spectramr {_spectramr_version()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full tracebacks on error (or set SPECTRAMR_DEBUG=1).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Doctor command — environment / wiring diagnostics (cluster pre-flight).
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Print environment diagnostics (torch/CUDA, devices, cache/data roots, env knobs).",
    )
    doctor_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Also validate this config loads against the schema.",
    )
    doctor_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if no CUDA device is visible (cluster pre-flight gate).",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON.")
    from spectramr.cli.diagnostics import run_doctor

    doctor_parser.set_defaults(func=run_doctor)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Path to config YAML"
    )
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    # Accept both spellings: --dry-run (argparse maps the hyphen to dry_run)
    # and the legacy --dry_run, so neither the docs nor old scripts break.
    train_parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Validate config without training",
    )
    train_parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Device (cuda/cpu/auto). If unset, uses the config's device (training.device, then run.device).",
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: from config)"
    )
    train_parser.add_argument(
        "--override",
        "-O",
        action="append",
        metavar="KEY=VALUE",
        help="Override config values (e.g., -O validation.schedule.interval_steps=100)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from checkpoint: a path, 'auto' (latest; raises if none), "
            "or 'if-present' (latest, else start fresh — the wall-clock chain spelling)."
        ),
    )
    train_parser.add_argument(
        "--allow-status",
        type=str,
        default=None,
        metavar="STATUS",
        help=(
            "Launch an arm whose metadata.status is needs_implementation, inert or "
            "blocked anyway, as a wiring / smoke exercise. Must name the exact status; "
            "it is stamped into provenance. Without it such an arm refuses to launch."
        ),
    )
    add_val_batches_argument(train_parser)
    train_parser.set_defaults(func=train)

    # Sanity check command
    sanity_parser = subparsers.add_parser(
        "sanity_check", help="Run sanity check (overfit single batch)"
    )
    sanity_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Path to config YAML"
    )
    sanity_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    sanity_parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Validate config without training",
    )
    sanity_parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Device (cuda/cpu/auto). If unset, uses the config's device (training.device, then run.device).",
    )
    sanity_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: from config)"
    )
    sanity_parser.add_argument(
        "--override",
        "-O",
        action="append",
        metavar="KEY=VALUE",
        help="Override config values (e.g., -O validation.schedule.interval_steps=100)",
    )
    sanity_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from checkpoint: a path, 'auto' (latest; raises if none), "
            "or 'if-present' (latest, else start fresh — the wall-clock chain spelling)."
        ),
    )
    add_val_batches_argument(sanity_parser)
    sanity_parser.set_defaults(func=sanity_check)

    # Ablation command — baseline + one variant per --vary override, with an
    # automated baseline→variant delta report (ablation_results.json).
    ablation_parser = subparsers.add_parser(
        "ablation",
        help=(
            "Run an ablation study: train the baseline config plus one variant "
            "per --vary override, then report per-metric deltas. Sequential / "
            "local; use `campaign` for large cluster sweeps."
        ),
    )
    ablation_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="Baseline config YAML"
    )
    ablation_parser.add_argument(
        "--vary",
        action="append",
        metavar="DOTTED.PATH=VALUE",
        help=(
            "Config override defining one variant (repeatable). Example: "
            "--vary model.model_kwargs.force_pure_kspace=false"
        ),
    )
    ablation_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write ablation_results.json (default: <config>_ablation/).",
    )
    ablation_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap training iterations per arm (quick local sweeps).",
    )
    ablation_parser.add_argument("--device", "-d", default=None, help="Device (cuda/cpu/auto).")
    ablation_parser.set_defaults(func=ablation)

    # Infer command (SSOT inference pipeline; config = training YAML)
    infer_parser = subparsers.add_parser(
        "infer", help="Run inference using a trained model (run_inference_pipeline)"
    )
    infer_parser.add_argument(
        "--checkpoint", "-ckpt", required=True, type=Path, help="Model checkpoint"
    )
    infer_parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input image or directory (omit only with --from-manifest-test-split)",
    )
    infer_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    infer_parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu/auto). If unset, uses the config's device (training.device, then run.device).",
    )
    infer_parser.add_argument(
        "--from-manifest-test-split",
        action="store_true",
        help=(
            "Take the input roster from the paired manifest's held-out test "
            "split (the unpaired-ULF cohort) instead of --input. Requires "
            "data.source.paired_manifest_path in the config."
        ),
    )
    infer_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help=(
            "Training YAML that produced the checkpoint. Optional when the checkpoint's "
            "run directory holds resolved_config.json, which wins unless --from-yaml is set."
        ),
    )
    infer_parser.add_argument(
        "--from-yaml",
        action="store_true",
        help="Read the settings from --config even when resolved_config.json exists beside the checkpoint.",
    )
    infer_parser.set_defaults(func=infer)

    # Infer-dataset command (deprecated alias for infer; kept for cluster jobs)
    infer_dataset_parser = subparsers.add_parser(
        "infer-dataset",
        help="[DEPRECATED] Dataset-loader inference (alias for `infer`)",
    )
    infer_dataset_parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Training config YAML"
    )
    infer_dataset_parser.add_argument(
        "--checkpoint", "-ckpt", required=True, type=Path, help="Model checkpoint"
    )
    infer_dataset_parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Input directory"
    )
    infer_dataset_parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output directory"
    )
    infer_dataset_parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu/auto). If unset, uses the config's device (training.device, then run.device).",
    )
    infer_dataset_parser.add_argument(
        "--batch-size", "-b", type=int, help="Override batch size from config"
    )
    infer_dataset_parser.set_defaults(func=infer_dataset)

    # Experiment command (directory management + training)
    experiment_parser = subparsers.add_parser(
        "experiment", help="Run complete experiment with directory management"
    )
    experiment_parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Path to config YAML"
    )
    experiment_parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        type=str,
        help="Unique experiment identifier (directory name)",
    )
    experiment_parser.add_argument(
        "--max-epochs", type=int, default=100, help="Max epochs (default: 100)"
    )
    experiment_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)",
    )
    experiment_parser.add_argument("--resume", type=Path, help="Resume from checkpoint (.pt path)")
    experiment_parser.add_argument("--device", "-d", default=None, help="Device (cuda/cpu/auto)")
    experiment_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    experiment_parser.add_argument(
        "--override",
        "-O",
        action="append",
        metavar="KEY=VALUE",
        help="Override config values",
    )
    experiment_parser.set_defaults(func=experiment)

    # Train-distributed command (DDP via torchrun)
    dist_parser = subparsers.add_parser(
        "train-distributed",
        help="DDP training (launch via: torchrun --nproc_per_node=N -m spectramr.cli train-distributed ...)",
    )
    dist_parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    dist_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (path, 'auto', or 'if-present')",
    )
    dist_parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["nccl", "gloo", "mpi"],
        help="Override parallel.backend. Default None (NOT 'nccl') on purpose: "
        "a concrete default made 'the user asked for nccl' indistinguishable "
        "from 'argparse filled it in', so the YAML value could never win.",
    )
    dist_parser.add_argument(
        "--override",
        "-O",
        action="append",
        metavar="KEY=VALUE",
        help="Override config values",
    )
    dist_parser.set_defaults(func=train_distributed)

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference via the SSOT pipeline (resolved_config.json beside --model, or --config)",
    )
    predict_parser.add_argument(
        "--model", "-m", type=Path, required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input directory or file"
    )
    predict_parser.add_argument(
        "--output", "-o", type=Path, default=Path("output/"), help="Output directory"
    )
    # Optional so the config-less verb still PARSES (parser tests), but predict()
    # fails loud at run time without it (config = inference SSOT). See predict().
    predict_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Training YAML that produced the checkpoint (SSOT for inference)",
    )
    predict_parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu/auto). If unset, uses the config's device (training.device, then run.device).",
    )
    predict_parser.add_argument(
        "--from-yaml",
        action="store_true",
        help="Read the settings from --config even when resolved_config.json exists beside --model.",
    )
    predict_parser.set_defaults(func=predict)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument(
        "--suite",
        "-s",
        choices=["standard", "memory", "throughput", "all"],
        default="standard",
        help="Benchmark suite to run",
    )
    bench_parser.add_argument(
        "--device",
        "-d",
        default=None,
        help=(
            "Device: auto (default) | cuda | cuda:<idx> | cpu. Benchmarking is a "
            "heavy pipeline — with no accelerator this RAISES rather than "
            "silently timing CPU kernels against GPU baselines. Pass `cpu` to "
            "deliberately benchmark on CPU."
        ),
    )
    bench_parser.set_defaults(func=benchmark)

    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument(
        "--model", "-m", type=Path, required=True, help="Path to model checkpoint"
    )
    export_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to training config YAML (required for real data fetching)",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["onnx", "torchscript", "both"],
        default="onnx",
        help="Export format",
    )
    export_parser.set_defaults(func=export)

    # List features command
    features_parser = subparsers.add_parser(
        "list-features", help="List available features (models, losses, metrics, etc.)"
    )
    features_parser.add_argument(
        "--module",
        "-m",
        choices=[
            "all",
            "models",
            "losses",
            "metrics",
            "strategies",
            "physics",
            "services",
        ],
        default="all",
        help="Module to list features for",
    )
    features_parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "csv"],
        default="markdown",
        help="Output format",
    )
    features_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (if not provided, prints to stdout)",
    )
    features_parser.set_defaults(func=list_features)

    # ── Audit command (Tier 0+1, optionally Tier 2) ──
    audit_parser = subparsers.add_parser(
        "audit",
        help=(
            "Audit an experiment YAML against the v1.0 schema (Tier 0), "
            "ConfigHealthChecker (Tier 1), and optionally a synthetic "
            "forward probe (Tier 2 with --probe). Exit code: 0 pass, "
            "1 warnings only, 2 errors."
        ),
    )
    audit_parser.add_argument("config", type=Path, help="Path to experiment YAML")
    audit_parser.add_argument(
        "--probe",
        action="store_true",
        help="Run the Tier-2 synthetic forward probe (~30 s, instantiates the model).",
    )
    audit_parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "Device for the Tier-2 probe: auto | cuda | cuda:<idx> | cpu. "
            "Default: follow the config's own device, so the probe exercises "
            "the device the real run will use. The probe MUST run on an "
            "accelerator — a CPU probe cannot catch CUDA OOM or AMP/GradScaler "
            "traps, which are its whole point — so --probe FAILS when no "
            "accelerator is available. Pass --device cpu (or FORCE_CPU=true) to "
            "deliberately accept a degraded, non-accelerated probe."
        ),
    )
    audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a structured JSON report instead of human-readable text.",
    )
    audit_parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Promote every warning to an error (exit 2 instead of 1). "
            "ON by default: CLAUDE.md non-negotiable 4 says a "
            "passed-with-warnings audit is not a pass, because that is the "
            "state that masks identity collapse, dropped losses and "
            "val-time OOM. Pass --no-strict to accept warnings with exit 1; "
            "per-arm opt-out belongs in the config "
            "(synthetic_forward_probe_skip), not on the command line."
        ),
    )
    audit_parser.add_argument(
        "--noise",
        action="store_true",
        help=(
            "Use white noise instead of a Shepp-Logan phantom for the "
            "Tier-2 probe input. Default is the phantom — its visible "
            "structure makes saved probe images inspectable."
        ),
    )
    audit_parser.add_argument(
        "--write-resolved-config",
        type=Path,
        metavar="DIR",
        default=None,
        help=(
            "Write resolved_config.json (config + _ledger) into DIR. The audit "
            "resolves the same config training will, so this records every "
            "dropped / defaulted / rewritten knob BEFORE any GPU time is spent."
        ),
    )
    audit_parser.add_argument(
        "--save-probe-images",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "If set, write the probe's input / output / target as PNG "
            "mosaics into DIR. One file per role per arm. Useful for "
            "eyeballing whether channels / k-space ↔ image conversions "
            "are handled correctly."
        ),
    )
    audit_parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help=(
            "Bulk mode only: skip YAMLs whose path (relative to the audited "
            "directory) or filename matches PATTERN. Glob syntax (fnmatch); a "
            "bare substring is wrapped as '*PATTERN*'. Repeatable. Canonical "
            "use: --exclude '*ablation*' to focus the audit on training arms "
            "(matches both ablations/ subdirs and *ablation* filenames)."
        ),
    )
    audit_parser.set_defaults(func=audit)

    # Campaign command (with sub-subcommands)
    campaign_parser = subparsers.add_parser(
        "campaign", help="Manage experiment campaigns (submit/status/evaluate/cancel)"
    )
    campaign_sub = campaign_parser.add_subparsers(dest="campaign_action", required=True)

    # campaign submit
    cs = campaign_sub.add_parser("submit", help="Submit a campaign to the cluster")
    cs.add_argument("config", type=Path, help="Path to campaign YAML")
    cs.add_argument("--base-dir", type=Path, default=None, help="Project root directory")
    cs.add_argument("--dry-run", action="store_true", help="Validate without submitting")
    cs.add_argument("--resume", action="store_true", help="Resume experiments from checkpoints")
    cs.add_argument(
        "--where",
        default="slurm",
        choices=["slurm", "docker", "apptainer"],
        help="Per-arm execution target: slurm (sbatch, default) or docker/"
        "apptainer (run each arm in a container, sequentially; parallel-mode only).",
    )
    cs.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME[,NAME...]",
        help="Run only the named arms (comma-separated; repeatable).",
    )
    cs.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="key=value",
        help=(
            "Include selector (repeatable). key is one of "
            "'name', 'role', or 'tag.<key>'. Multiple --include "
            "selectors combine via OR."
        ),
    )
    cs.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="key=value",
        help="Exclude selector (same syntax as --include; repeatable).",
    )
    cs.set_defaults(func=campaign_submit)

    # campaign status
    cst = campaign_sub.add_parser("status", help="Check campaign progress")
    cst.add_argument("campaign_dir", type=Path, help="Campaign output directory")
    cst.add_argument("--base-dir", type=Path, default=None)
    cst.add_argument("--dry-run", action="store_true")
    cst.set_defaults(func=campaign_status)

    # campaign evaluate
    ce = campaign_sub.add_parser("evaluate", help="Evaluate a completed campaign")
    ce.add_argument("campaign_dir", type=Path, help="Campaign output directory")
    ce.add_argument("--latex", action="store_true", help="Export LaTeX leaderboard table")
    ce.set_defaults(func=campaign_evaluate)

    # campaign cancel
    cc = campaign_sub.add_parser("cancel", help="Cancel all active jobs")
    cc.add_argument("campaign_dir", type=Path, help="Campaign output directory")
    cc.add_argument("--base-dir", type=Path, default=None)
    cc.set_defaults(func=campaign_cancel)

    # campaign watch
    cw = campaign_sub.add_parser("watch", help="Watch progress and auto-evaluate when complete")
    cw.add_argument("campaign_dir", type=Path, help="Campaign output directory")
    cw.add_argument("--base-dir", type=Path, default=None)
    cw.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Seconds between SLURM polls (default: 300 = 5 min)",
    )
    cw.set_defaults(func=campaign_watch)

    # ── hpo ─────────────────────────────────────────────────────────────
    hpo_parser = subparsers.add_parser(
        "hpo",
        help=(
            "Run hyperparameter optimization (Optuna-backed) over a base "
            "training YAML. Each trial spawns a separate trainer subprocess "
            "for crash isolation."
        ),
    )
    hpo_parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Base training YAML each trial will mutate.",
    )
    hpo_parser.add_argument(
        "--model-type",
        "-m",
        required=True,
        action="append",
        help=("Registered model_type to optimize (repeatable — one Optuna study per type)."),
    )
    hpo_parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials per model (default: 50).",
    )
    hpo_parser.add_argument(
        "--objective-metric",
        default="val_loss",
        help=(
            "CSV column to optimize. Auto-detects min vs max from name (loss/error/mse → minimize)."
        ),
    )
    hpo_parser.add_argument(
        "--max-iter",
        type=int,
        default=30000,
        help="Truncated training budget per trial (default: 30000).",
    )
    hpo_parser.add_argument(
        "--sampler",
        choices=["tpe", "cmaes", "nsga2"],
        default="tpe",
    )
    hpo_parser.add_argument(
        "--pruner",
        choices=["hyperband", "median", "successive_halving", "threshold", "none"],
        default="hyperband",
    )
    hpo_parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (e.g. sqlite:///experiments/hpo/study.db).",
    )
    hpo_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <config.training.output_dir>/hpo.",
    )
    hpo_parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Device (cuda/cpu/auto). If unset, uses the config's device (training.device, then run.device).",
    )
    hpo_parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="Multi-objective HPO (objective_metric, training_time_seconds).",
    )
    hpo_parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help="Cost-weight for multi-objective HPO (default: 0.0).",
    )
    # Search space — what HPO is allowed to vary per trial. Mutually
    # exclusive: pass at most one of --search-preset / --search-space.
    # If neither is given, every trial runs the base config unchanged
    # (the coordinator logs a loud warning).
    hpo_parser.add_argument(
        "--search-preset",
        default=None,
        action="append",
        help=(
            "Built-in search space preset name. Run `--list-presets` to see "
            "available choices. Repeatable — multiple presets are merged "
            "(conflict on overlapping paths raises an error). Mutually "
            "exclusive with --search-space."
        ),
    )
    hpo_parser.add_argument(
        "--preset-merge-policy",
        choices=["error", "override", "keep"],
        default="error",
        help=(
            "How to resolve conflicts when merging multiple --search-preset "
            "values: error (default, safest), override (later preset wins), "
            "keep (first preset wins)."
        ),
    )
    hpo_parser.add_argument(
        "--search-space",
        type=Path,
        default=None,
        help=(
            "Path to a YAML file describing the search space. Each top-level "
            "key is a dotted-path config key; each value is "
            "{dist: <kind>, low/high/choices: ...}. Mutually exclusive with "
            "--search-preset."
        ),
    )
    hpo_parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the names of all built-in search-space presets and exit.",
    )
    hpo_parser.add_argument(
        "--list-schema-paths",
        action="store_true",
        help=(
            "Print every dotted-path field exposed by TrainingSettings (so "
            "you can build your own --search-space YAML) and exit."
        ),
    )
    hpo_parser.add_argument(
        "--print-template",
        action="store_true",
        help=(
            "Print a starter --search-space YAML template (LR + WD + curriculum) "
            "and exit. Pipe to a file: --print-template > my_space.yaml"
        ),
    )
    hpo_parser.set_defaults(func=hpo_cmd)

    # ── report ──────────────────────────────────────────────────────────
    report_parser = subparsers.add_parser(
        "report",
        help=(
            "Generate canonical figures + tables for an experiment output dir "
            "(same pipeline as the end-of-training reporting hook)."
        ),
    )
    report_parser.add_argument(
        "--exp-dir",
        "-e",
        required=True,
        type=Path,
        help="Experiment output directory (containing logs/training_metrics.csv), "
        "or a cohort root when --recursive is given.",
    )
    report_parser.add_argument(
        "--recursive",
        "-r",
        "--all",
        dest="recursive",
        action="store_true",
        help="Treat --exp-dir as a cohort root: discover every run beneath it, "
        "report each, and write a linking report_index.html.",
    )
    report_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Optional config YAML: reuse its reporting: block (style/formats/dpi/"
        "figures/tables/qc_figures/html_report) — parity with the "
        "end-of-training hook. Explicit CLI flags override it.",
    )
    report_parser.add_argument(
        "--task",
        default=None,
        # Must equal sorted(TASK_PRESETS). Written out rather than derived
        # because importing `reporting.pipeline` costs ~3.3s and this runs while
        # building the parser for EVERY command, including `--help`; the rest of
        # this module imports reporting lazily inside handlers for the same
        # reason. `calibration` was missing here, so `TASK_PRESETS["calibration"]`
        # existed and was unreachable from the command line -- an advertised
        # preset nothing could select (non-negotiable 8). Agreement is enforced by
        # `test_report_task_choices_cover_every_preset`, which is where drift gets
        # caught instead of at 3.3s per invocation.
        choices=[
            "calibration",
            "default",
            "diffusion",
            "gan",
            "reconstruction",
            "super_resolution",
            "synthesis",
            "vae",
        ],
        help="Task preset that picks the default figure/table set "
        "(default: 'default', or the config's reporting.task).",
    )
    report_parser.add_argument("--method", default=None, help="Method label for plots.")
    report_parser.add_argument(
        "--figures",
        default=None,
        help="Comma-separated figure ids to render. Default: ALL applicable "
        "figures — the report command is the plotting SSOT and data-less "
        "figures soft-skip. e.g. --figures fig_1_2_learning_curves,qc_group_strip",
    )
    report_parser.add_argument(
        "--out-subdir",
        default=None,
        help="Subdirectory under exp-dir to write artifacts (default: 'report').",
    )
    report_parser.add_argument(
        "--seed", type=int, default=None, help="Seed stamped onto figure metadata."
    )
    report_parser.add_argument(
        "--dataset-version",
        default=None,
        help="Dataset version stamped onto figure metadata.",
    )
    report_parser.add_argument(
        "--cohort-json",
        type=Path,
        default=None,
        help="Path to a JSON file with cohort fields (train_n, val_n, test_n, age_mean, ...).",
    )
    report_parser.add_argument(
        "--html",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit (--html) or skip (--no-html) the self-contained QC "
        "HTML report. Default: on (or the config's reporting.html_report).",
    )
    report_parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit (--interactive) or skip (--no-interactive) the interactive "
        "plotly layer (2-D/3-D viewers, hoverable plots). Falls back to "
        "static PNGs when plotly is not installed. Overrides the config.",
    )
    report_parser.add_argument(
        "--qc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include (--qc) or skip (--no-qc) the QC figures. "
        "Default: on (or the config's reporting.qc_figures).",
    )
    report_parser.set_defaults(func=report)

    # Meta-evaluation command (CDSCR / LGDR / SFA / ESD / FSPD).
    meta_parser = subparsers.add_parser(
        "meta-evaluate",
        help="Rank a metric set with the five-method meta-evaluation framework.",
    )
    meta_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Directory of .pt clean references OR M4Raw multi-coil .h5 files "
        "(filename: <subject>_<contrast><rep>.h5). For H5 input, pseudo-GT "
        "is synthesized per (subject, contrast) group — see "
        "scripts/sim2rank/ground_truth.py. Omit for synthetic-phantom smoke mode.",
    )
    meta_parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric names from the registry; defaults to a short curated list.",
    )
    meta_parser.add_argument(
        "--severities",
        type=int,
        default=8,
        help="Severity grid size per degradation family.",
    )
    meta_parser.add_argument(
        "--max-subjects",
        type=int,
        default=8,
        dest="max_subjects",
        help="Cap on number of distinct M4Raw subjects (multi-contrast: see --max-contrasts).",
    )
    meta_parser.add_argument(
        "--max-contrasts",
        type=int,
        default=3,
        dest="max_contrasts",
        help="Contrasts per subject for the H5 loader (default 3 = T1, T2, FLAIR). "
        "Ignored for .pt input.",
    )
    meta_parser.add_argument(
        "--contrasts",
        nargs="+",
        default=None,
        help="Whitelist of contrasts to load (e.g. T1 T2). Default: all available.",
    )
    meta_parser.add_argument(
        "--synthetic-n",
        type=int,
        default=4,
        dest="synthetic_n",
        help="Number of synthetic phantoms when --input is omitted (smoke mode).",
    )
    meta_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto (default; CUDA if available else CPU), cpu, "
        "cuda, or cuda:<idx>. Explicit `cuda` raises if no GPU is visible "
        "rather than silently falling back to CPU.",
    )
    meta_parser.add_argument(
        "--eval-mode",
        type=str,
        default="2d",
        dest="eval_mode",
        choices=["2d", "3d"],
        help="Evaluation dimensionality. '2d' (default) is the per-slice path. "
        "'3d' (volumetric) is not implemented and raises at startup "
        "(fail-loud: an unimplemented option must not "
        "silently degrade). The resolved value is stamped "
        "into the summary JSON as 'eval_mode' so every bundle is "
        "self-describing about its dimensionality.",
    )
    meta_parser.add_argument(
        "--betting",
        action="store_true",
        dest="betting",
        help="Also emit the L2⁺ variance-adaptive, anytime-valid betting "
        "certification (Waudby-Smith–Ramdas confidence sequence; Lean "
        "Sim2Rank.Betting) alongside the always-on Hoeffding-powered gate. "
        "Omit for the 'real' powered-only implementation (the default).",
    )
    meta_parser.add_argument(
        "--betting-alpha",
        type=float,
        default=0.05,
        dest="betting_alpha",
        help="Mis-coverage budget α for the betting confidence sequence "
        "(only consumed when --betting is set). Must be in (0, 1).",
    )
    meta_parser.add_argument(
        "--aggregator",
        type=str,
        default="z",
        choices=["z", "kemeny"],
        help="Cross-axis consensus combiner. 'z' (default) is the Pareto-"
        "consistent z-weighted mean; 'kemeny' is the Condorcet-consistent "
        "L3⁺ Kemeny median (Lean Sim2Rank.Kemeny). NB: the two write "
        "different events into 'defensibility_flags' (breadth-of-agreement vs "
        "Condorcet winner) — the chosen semantics is stamped into the summary "
        "as 'defensibility_semantics'.",
    )
    meta_parser.add_argument(
        "--screen-pool",
        action="store_true",
        dest="screen_pool",
        help="Pool hygiene: screen clone / broken metrics out of the voting pool "
        "before ranking (§8 repair). Off by default (it changes the voting "
        "set). The screened pool is reported in the summary under 'screening'.",
    )
    meta_parser.add_argument(
        "--core-degradations",
        action="store_true",
        dest="core_degradations",
        help="Use the dependency-free core degradation library (8 simpler ops) "
        "instead of the audited digital-twin physics registry (the default "
        "SSOT). For physics-less environments / core-only smoke runs. The "
        "resolved library is recorded in the summary's 'degradation_provenance'.",
    )
    meta_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Simulator seed.",
    )
    meta_parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("experiments/results/meta_evaluation/summary.json"),
        help="Where to write the JSON summary.",
    )
    meta_parser.add_argument(
        "--figures",
        action="store_true",
        help="Render the diagnostic figures alongside the JSON summary.",
    )
    meta_parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        dest="figures_dir",
        help="Directory for rendered figures (defaults to <out>.parent/figures).",
    )
    meta_parser.add_argument(
        "--tables",
        action="store_true",
        help="Write the CSV exports (summary, per-family, correlation, raw).",
    )
    meta_parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        dest="tables_dir",
        help="Directory for CSV exports (defaults to <out>.parent/tables).",
    )
    meta_parser.add_argument(
        "--nr-battery",
        action="store_true",
        dest="nr_battery",
        help=(
            "Score the no-reference (label-free) metric battery: defaults "
            "--metrics to the NR battery and drives the sweep with the audited "
            "digital-twin degradations (physics-anchored concordance, spec §8.1)."
        ),
    )
    meta_parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        dest="tiers",
        help=(
            "Run the FULL §8 NR-metric validation harness instead of the bare "
            "ranking, emitting a per-metric pass/fail card across the requested "
            "label-free tiers. Choose any of: concordance (§8.1), fr_proxy "
            "(§8.2), cho (§8.3), construct_validity (§8.4); or 'all' for every "
            "tier. Implies --nr-battery semantics (NR battery + digital-twin "
            "sweep). The §8.7 aggregator (nr_quality_index) is fit alongside. "
            "RESEARCH-MODE: this is HOW the battery is validated — the metrics are "
            "not production-certified and never run during training."
        ),
    )
    meta_parser.add_argument(
        "--register-aggregator",
        action="store_true",
        dest="register_aggregator",
        help=(
            "When running --tiers, register the fitted nr_quality_index meta-metric "
            "into the global registry (off by default — research-mode runs do not "
            "pollute the registry)."
        ),
    )
    meta_parser.set_defaults(func=meta_evaluate)

    # ── audit_plan_novel.md subcommands (Ideas 2, 4, 7) ──
    # Wired here so ``--help`` shows audit-ksd / infer-protocol /
    # simulate-acquisition alongside the existing subcommands.
    try:
        from spectramr.cli.audit_plan_novel_cli import attach_subparsers as _attach_novel

        _attach_novel(subparsers)
    except ImportError as exc:  # pragma: no cover - import-time safety
        logger.warning("audit_plan_novel subcommands unavailable: %s", exc)
    # MRF §4 — design-mrf-sequence
    try:
        from spectramr.cli.design_mrf_sequence_cli import attach_subparsers as _attach_mrf

        _attach_mrf(subparsers)
    except ImportError as exc:  # pragma: no cover
        logger.warning("design-mrf-sequence subcommand unavailable: %s", exc)

    # ULF-PR-27 / V.4 — regulatory bundle CLI. Unlike the two attachments
    # above (which expose ``attach_subparsers(subparsers)``), regulatory.py
    # exposes ``add_arguments(parser)`` that mounts its own ``cmd`` sub-
    # subparser. So we first create the top-level ``regulatory`` parser
    # here and then hand it off.
    try:
        from spectramr.cli.regulatory import add_arguments as _attach_regulatory

        reg_parser = subparsers.add_parser(
            "regulatory",
            help="Regulatory bundle CLI (bundle / verify / status)",
        )
        _attach_regulatory(reg_parser)
    except ImportError as exc:  # pragma: no cover
        logger.warning("regulatory subcommand unavailable: %s", exc)

    # Unified launcher (WS-D): run any pipeline anywhere. Additive — the
    # dedicated commands (train / campaign submit / sbatch) keep working.
    from spectramr.cli.launch import (
        FANOUT_CHOICES,
        PIPELINE_VERBS,
        WHERE_CHOICES,
        launch,
    )

    launch_parser = subparsers.add_parser(
        "launch",
        help="Unified launcher: run any pipeline (train/infer/…) anywhere "
        "(local/docker/apptainer/slurm), single or campaign.",
    )
    launch_parser.add_argument(
        "config",
        type=Path,
        help="Training config YAML (or a campaign manifest with --fanout campaign).",
    )
    launch_parser.add_argument("--pipeline", default="train", choices=PIPELINE_VERBS)
    launch_parser.add_argument("--where", default="local", choices=WHERE_CHOICES)
    launch_parser.add_argument("--fanout", default="single", choices=FANOUT_CHOICES)
    launch_parser.add_argument("--account", default=None, help="SLURM account.")
    launch_parser.add_argument("--partition", default=None, help="SLURM partition.")
    launch_parser.add_argument("--mem", default=None, help="Memory (e.g. 64G).")
    launch_parser.add_argument("--gpus", type=int, default=None, help="GPUs to request.")
    launch_parser.add_argument("--time", default=None, help="Wall-clock limit (HH:MM:SS).")
    launch_parser.add_argument("--nodes", type=int, default=None, help="Nodes to request.")
    launch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command / sbatch script that would run, without executing.",
    )
    launch_parser.add_argument(
        "extra",
        nargs="*",
        help="Passthrough args for the verb after a '--' separator "
        "(e.g. ... -- --checkpoint best.pt --input d/ --output o/ for infer).",
    )
    launch_parser.set_defaults(func=launch)

    # ``spectramr profile`` — run a pipeline verb under Scalene. Attached
    # unconditionally (no try/except ImportError like the optional siblings
    # above): profile_cli imports cleanly WITHOUT scalene present, and checks
    # for it at run time so a missing profiler raises an actionable error
    # instead of the verb silently vanishing from --help.
    from spectramr.cli.profile_cli import attach_subparsers as _attach_profile

    _attach_profile(subparsers)

    return parser


# Subcommands whose first action imports PyTorch + the model registry
# (transitively monai / torchio) — tens of seconds on a cold process. Light
# verbs (``doctor``, ``campaign``, ``regulatory``, ``launch``) stay quiet because
# they don't pull the heavy graph. Names match the ``dest="command"`` subparser
# strings registered in ``build_parser``.
_HEAVY_STARTUP_COMMANDS = frozenset(
    {
        "train",
        "sanity_check",
        "ablation",
        "infer",
        "infer-dataset",
        "experiment",
        "train-distributed",
        "predict",
        "benchmark",
        "export",
        "list-features",
        "audit",
        "hpo",
        "report",
        "meta-evaluate",
        "profile",
    }
)


def _emit_startup_notice(command: str | None) -> bool:
    """Make the (unavoidable) first heavy import legible instead of a silent hang.

    The first ``train`` / ``audit`` / … in a fresh process must import PyTorch
    and the model registry; cold, that is tens of seconds during which the
    terminal looks frozen — the "after the import line it waits for a whole
    minute" report. We print ONE concise line to STDERR (never stdout, so
    ``audit --json | jq`` stays parseable) just before dispatch, so the wait is
    explained rather than mysterious.

    Quiet by design when the verb is lightweight, or when the batch-mode flags
    ``SPECTRAMR_QUIET`` / ``SPECTRAMR_SUPPRESS_CLINICAL_WARNING`` are set (the same
    switch that silences the clinical banner also silences this). Returns whether
    the notice was emitted so the gating is unit-testable.

    Also quiet on a non-zero rank. This is a bare ``print``, not a log record,
    so the root-logger clamp in ``quiet_secondary_ranks`` cannot reach it -- it
    needs its own gate, or a 4-GPU launch prints the same "importing PyTorch"
    line four times before anything else happens.
    """
    if command not in _HEAVY_STARTUP_COMMANDS:
        return False
    if os.environ.get("SPECTRAMR_QUIET") or os.environ.get("SPECTRAMR_SUPPRESS_CLINICAL_WARNING"):
        return False
    from spectramr.core import env as _env

    if _env.is_secondary_rank():
        return False
    print(
        f"⏳ spectramr {command}: importing PyTorch + model registry "
        f"(first call in a fresh process is slow, ~30-60 s)...",
        file=sys.stderr,
        flush=True,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``). Passed so the
            in-process ``LocalBackend`` (``spectramr launch --where local``) can
            drive any verb without spawning a subprocess.
    """
    # NB: ``bootstrap_console_logging()`` is deferred to AFTER ``parse_args`` +
    # the help guard below (importing the logging service pulls torch
    # transitively, so calling it here would defeat the torch-free ``--help`` /
    # usage-error path). See the startup-budget regression test.
    parser = build_parser()
    args = parser.parse_args(argv)
    # ``subparsers = ... required=True`` already enforces that a
    # subcommand is given, but a subcommand parser that forgets to call
    # ``set_defaults(func=...)`` would silently produce a Namespace
    # without ``args.func`` and crash with AttributeError. Fail loud
    # with a helpful exit code instead — see
    # ``TODO/audit/15_entry_points_bootstrap_cli.md`` F5.
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # F-LOG-COLOR round 12 (2026-05-17): install the colored console handler on
    # the root logger before the subcommand handler runs (idempotent; a no-op
    # when already present). Pre-fix, only ``hpo`` instantiated
    # ``LoggingService`` so train/audit/predict/sanity_check/campaign fell back
    # to Python's plain ``lastResort`` handler. Deferred to AFTER argument
    # parsing so ``--help`` / usage errors stay torch-free (the logging service
    # pulls torch transitively). Idempotent; real commands load torch anyway.
    from spectramr.infrastructure.logging import bootstrap_console_logging
    from spectramr.infrastructure.logging.rank_console import quiet_secondary_ranks

    bootstrap_console_logging()

    # torchrun runs N independent interpreters, so every INFO emitted before the
    # process group exists lands in the job log N times. ``setup_distributed``
    # applies this same clamp, but only after the config has loaded and the
    # backend resolved -- too late for the startup lines. Apply it here, off the
    # rank torchrun exported before the process started. No-op single-process.
    quiet_secondary_ranks()

    # The dispatched command will (for heavy verbs) now pull torch + the model
    # registry — tens of seconds cold. Signal that up front so the entry point
    # never looks frozen. Stderr-only + suppressible; see _emit_startup_notice.
    _emit_startup_notice(getattr(args, "command", None))

    # Top-level error boundary: a long cluster run should exit with a clean
    # message + meaningful code, not dump a raw traceback into the SLURM log.
    # Ctrl-C → 130 (conventional); handler ``sys.exit`` is preserved; any other
    # exception → concise error + exit 1 (full traceback only with -v / SPECTRAMR_DEBUG).
    verbose = bool(getattr(args, "verbose", False)) or bool(os.environ.get("SPECTRAMR_DEBUG"))
    cmd = getattr(args, "command", "?")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("'%s' interrupted by user (Ctrl-C).", cmd)
        return 130
    except SystemExit:
        raise  # a handler asked to exit with a specific code — honour it
    except Exception as exc:
        if verbose:
            logger.exception("Command '%s' failed", cmd)
        else:
            logger.error("Command '%s' failed: %s", cmd, exc)
            logger.error("Re-run with -v / --verbose (or SPECTRAMR_DEBUG=1) for a full traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
