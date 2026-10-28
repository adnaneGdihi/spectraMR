import argparse
import logging
import os
import sys
import warnings

# Suppress PyTorch internal cuda.cudart deprecation warning (not actionable by us)
warnings.filterwarnings(
    "ignore",
    message="The cuda.cudart module is deprecated",
    category=FutureWarning,
)

# Suppress PyTorch deterministic-mode warnings for ops that have no
# deterministic implementation.  These fire every training step and are
# non-actionable (we already set warn_only=True).
warnings.filterwarnings(
    "ignore",
    message=".*does not have a deterministic implementation.*",
    category=UserWarning,
)

# Suppress Triton cached-autotuning warning from mamba_ssm
warnings.filterwarnings(
    "ignore",
    message="Deterministic mode.*Triton",
    category=UserWarning,
)

# Early setup for Torch and Library Caches
# Use environment variables for portability: SPECTRAMR_CACHE_ROOT, SPECTRAMR_DATA_ROOT, SPECTRAMR_DEVICE
# Falls back to local ~/.cache if not set (local development)
# The cache layout, the clobber-vs-setdefault rule per variable, and the reason
# Triton and XDG_CACHE_HOME are load-bearing all live in ONE place now. They used
# to be inline here, which meant they applied to `spectramr train` and to nothing
# else: `torchrun -m spectramr.cli train-distributed` never imports this module, so
# the multi-GPU path ran with every one of these unset and DeepSpeed's import
# wrote into $HOME.
from spectramr.infrastructure.config.cpu_threads import resolve_cpu_threads  # noqa: E402
from spectramr.infrastructure.config.env_resolver import configure_cache_environment

configure_cache_environment()

# Threads per rank come from the scheduler's grant (SPECTRAMR_CPU_THREADS overrides);
# a run outside an allocation stays at 1.
_CPU_THREADS = resolve_cpu_threads()
os.environ["OMP_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_CPU_THREADS)

# CUDA memory configuration — MUST precede ``import torch``. PyTorch reads
# ``PYTORCH_CUDA_ALLOC_CONF`` exactly once, when the CUDA caching allocator is
# first initialised (at/just after import); an assignment placed *after* the
# import is silently inert. This block used to sit below ``import torch``, so
# ``expandable_segments:True`` never actually took effect and fragmentation OOMs
# were not mitigated despite the setting being present — the
# ``experiment_11_attention_wavelet_freq`` crash (2026-07-02): 2.88 GiB free on a
# 31 GiB GPU yet a 132 MiB allocation failed with 329 MiB "reserved but
# unallocated". Keep these three lines above ``import torch``.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"  # 2GB max (configurable in future)
os.environ["TORCH_CUDA_EAGER_CACHE_MANAGER"] = "1"
import torch

try:
    torch.set_num_threads(_CPU_THREADS)
except AttributeError:
    pass  # PyTorch build without OpenMP support
from spectramr.accelerator import initialize_accelerator

# ``apply_overrides`` / ``_parse_value`` now live in the innermost ``config/``
# layer (config/overrides.py) so ``pipelines/`` can import them rightward
# instead of leftward from ``main`` (CLAUDE.md #13). Re-exported here so
# ``from spectramr.main import apply_overrides`` / ``_parse_value`` keep working.
from spectramr.config.overrides import _parse_value, apply_overrides  # noqa: F401
from spectramr.config.settings import TrainingSettings

# NOTE: ``run_training_pipeline`` is imported LAZILY — inside the two functions
# that use it (``__common_train_setup`` / ``experiment_command``), never here at
# module top. The old top-level ``from spectramr.pipelines import
# run_training_pipeline`` pulled the ENTIRE pipeline graph (model registry →
# monai → torchio, tens of seconds cold) the instant *anything* imported this
# module — so ``from spectramr.main import _parse_value`` (used by the ``ablation``
# verb) and every other lightweight reach into ``spectramr.main`` paid the full
# cold-start cost up front. Deferring it keeps ``import spectramr.main`` cheap and
# lets a malformed config fail BEFORE the heavy import on the train path.

logger = logging.getLogger(__name__)


def _apply_log_level(settings: "TrainingSettings") -> None:
    """Apply the config's declared log level now, not only once the container builds."""
    from spectramr.infrastructure.logging.log_policy import apply_logging_config

    apply_logging_config(settings.logging)


def _resolved_determinism(settings) -> bool:
    """Resolve ``training.deterministic`` (reproducible-by-default).

    This is the single read point for the knob on the canonical entry path —
    it feeds ``initialize_accelerator(deterministic=...)``, which owns the
    cudnn/use_deterministic_algorithms policy. The hardcoded ``[AUDIT FIX]
    Enforce Full Determinism`` blocks that used to override the policy after
    every ``initialize_accelerator`` call (making this knob a pitfall-#15
    silent no-op) are gone. An absent knob resolves True, preserving the
    historically-forced deterministic behaviour.
    """
    training = getattr(settings, "training", None)
    value = getattr(training, "deterministic", None) if training is not None else None
    return True if value is None else bool(value)


def _declared_device(settings: "TrainingSettings") -> str | None:
    """Return the device the YAML *declared*, or ``None`` when it declared none.

    ``run.device`` is the canonical knob (``RENAMES``: top-level ``device`` ->
    ``run.device``, 2026-07-31) and it carries a schema default of ``"cuda"``.
    Reading the attribute therefore cannot tell "the user asked for CUDA" apart
    from "nobody said anything" -- and ``resolve_compute_device`` treats those
    two very differently: an explicit ``"cuda"`` is a hard requirement that
    ``FORCE_CPU`` does **not** relax, while ``None`` becomes ``"auto"`` and may
    legally land on CPU. ``model_fields_set`` is the only thing that separates
    them, so substituting the default for a declaration would turn every
    undeclared arm into a CUDA-mandated one (the no-silent-default-substitution
    rule, non-negotiable 3).

    This is the single owner of the question "what device did the config ask
    for?" (non-negotiable 17). Both the train family and the inference family
    call it; neither reads ``settings.run.device`` directly.
    """
    run = getattr(settings, "run", None)
    if run is None or "device" not in getattr(run, "model_fields_set", ()):
        return None
    return getattr(run, "device", None)


def begin_inference_run(
    config_path: "str | os.PathLike[str] | None",
    device_str: str | None,
    *,
    pipeline: str,
    checkpoint_path: "str | os.PathLike[str] | None" = None,
    from_yaml: bool = False,
) -> "tuple[TrainingSettings, torch.device]":
    """Arm the ledger, load the SSOT config, and resolve the accelerator.

    The single preamble for every inference-family verb. ``infer`` and
    ``infer-dataset`` each carried a verbatim copy of it; ``predict`` carried
    none of it -- it ran with no audit trail, no seed, no determinism policy,
    and a hardcoded ``or "cuda"`` that bypassed the accelerated-run contract
    (non-negotiable 9b). Extracting it is what makes ``predict``'s docstring
    claim -- "routes to the SAME entry point as ``infer``" -- true rather than
    a facade (pitfall #16).

    Returns the settings **and the resolved device**. Callers must forward that
    device to the pipeline instead of re-passing their raw ``--device`` string:
    ``"auto"``/``None`` otherwise resolves twice, independently, and the two
    answers are only incidentally equal.

    Exits 1 on a malformed config, matching the call sites this replaces.
    """
    # Arm the substitution ledger BEFORE the config resolves, so a knob the
    # schema drops is recorded on this surface too, not only on the train path.
    from spectramr.core.execution_ledger import ExecutionLedger

    ExecutionLedger.begin_run(source=str(config_path or checkpoint_path))

    try:
        if checkpoint_path is not None:
            # The run's own resolved_config.json beside the checkpoint wins over
            # the YAML unless ``from_yaml`` says otherwise (#1379); the resolver
            # logs which one it used and whether the two disagree.
            from spectramr.infrastructure.validation.inference_settings import (
                resolve_inference_settings,
            )

            settings, _source = resolve_inference_settings(
                config_path, checkpoint_path, from_yaml=from_yaml
            )
        else:
            settings = TrainingSettings.from_yaml(str(config_path))
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    _apply_log_level(settings)

    # Seed + determinism come from the training YAML that produced the
    # checkpoint. Hardcoding ``initialize_accelerator(device, 42)`` made
    # ``run.seed`` and ``training.deterministic`` silent no-ops (pitfall #15);
    # an absent knob still resolves deterministic=True, so reproducible output
    # stays the default for a scoring command.
    # ``--device`` wins; otherwise honour the device the YAML declared. Before
    # this, the inference family never consulted the config at all, so the CPU
    # opt-out that ``resolve_compute_device``'s own error message advertises
    # ("set ``run.device: cpu``") was unreachable from ``infer``/``predict``.
    return settings, initialize_accelerator(
        device_str or _declared_device(settings),
        settings.run.seed,
        deterministic=_resolved_determinism(settings),
        pipeline=pipeline,
    )


def __common_train_setup(args: argparse.Namespace, is_sanity_check: bool = False) -> None:
    """Shared pipeline bootstrap logic for train and sanity check."""
    # 0. Arm the substitution ledger BEFORE the config is loaded. Most silent
    #    drops happen during resolution itself (a key the schema ignores, a
    #    legacy bridge rewrite, a default standing in for an undeclared knob),
    #    so a ledger armed after ``from_yaml`` would miss exactly the class it
    #    exists to catch. Ends up in ``resolved_config.json`` under ``_ledger``.
    from spectramr.core.execution_ledger import ExecutionLedger

    ExecutionLedger.begin_run(source=str(args.config))

    # 1. Load Configuration first (v6.0: seed in config.training.seed)
    try:
        settings = TrainingSettings.from_yaml(str(args.config))
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    _apply_log_level(settings)

    # 2. Get seed: CLI override > run.seed (schema default 42).
    # `training.seed` is a RAISE-posture rename (phase 4b), so the old
    # getattr could never succeed and every path here silently used 42 --
    # including predict/infer_dataset, which never reach train.py's reader.
    seed = args.seed or settings.run.seed
    # Device priority: CLI > training.device > run.device (declared only).
    # The third leg used to read ``settings.device`` -- the *legacy* top-level
    # spelling, which ``RENAMES`` retired to ``run.device`` with a RAISE posture
    # on 2026-07-31. A config carrying it no longer validates, and one carrying
    # the canonical ``run.device`` was never read here, so this leg had been
    # permanently ``None`` and the canonical knob had no consumer on the train
    # path at all (pitfall #15).
    device = args.device or getattr(settings.training, "device", None) or _declared_device(settings)

    # 3. Initialize Accelerator (Torch setup) with resolved seed.
    # Determinism policy comes from the (now wired) training.deterministic
    # knob; initialize_accelerator owns the cudnn flags +
    # use_deterministic_algorithms(warn_only=True) — no post-hoc override.
    # Device: heavy pipeline → accelerated or raise (never a silent CPU run).
    pipeline_name = "sanity_check" if is_sanity_check else "train"
    initialize_accelerator(
        device,
        seed,
        deterministic=_resolved_determinism(settings),
        pipeline=pipeline_name,
    )

    # 2b. Apply overrides if provided
    if is_sanity_check:
        if not hasattr(args, "override") or args.override is None:
            args.override = []

        sanity_overrides = [
            # Data Pipeline: Force batch size 1 to prevent CUDA OOM
            "data.batch_size=1",
            # Data Pipeline: Batch isolation is handled in run_training_pipeline
            # Architecture & Regularization: Dropout/BatchNorm frozen via model.eval() in train loop
            "ema.enabled=False",
            # Optimizer & Hyperparameters
            "optimization.optimizer.learning_rate=1e-4",
            "optimization.warmup_steps=0",
            "optimization.lr_scheduler_strategy=step_lr",
            "optimization.lr_scheduler_kwargs.step_size=999999",
            "optimization.lr_scheduler_kwargs.gamma=1.0",
        ]

        # Inject without clobbering explicit user overrides
        existing_keys = [o.split("=")[0] for o in args.override]
        for so in sanity_overrides:
            k = so.split("=")[0]
            if k not in existing_keys:
                args.override.append(so)

        logger.info(
            f"🧪 [SANITY CHECK MODE] Injected Overrides for Overfitting: {sanity_overrides}"
        )

    if hasattr(args, "override") and args.override:
        settings = apply_overrides(settings, args.override)
        _apply_log_level(settings)

    if args.dry_run:
        # Dry run mode: just validate config
        from spectramr.bootstrap import build_container

        # ``device``, not ``args.device``: a dry run must validate the same
        # device the real run would resolve, or it green-lights a config whose
        # declared device the live path would reject.
        build_container(settings, device=device, pipeline=pipeline_name)
        logger.info("✅ Dry run completed. Configuration valid.")
        return

    # 4. Run Training Pipeline (lazy heavy import — see the module-top note).
    from spectramr.pipelines import run_training_pipeline

    resume_path = getattr(args, "resume", None)
    try:
        result = run_training_pipeline(
            settings,
            # ``device``, not ``args.device`` -- the same resolved value the
            # accelerator was initialized with above. Passing the raw CLI string
            # made ``build_container`` re-resolve the device independently
            # (train.py:534), so a config that declared ``run.device`` got an
            # accelerator on the declared device and a container on whatever
            # ``auto`` picked. Two owners of one question, agreeing only by
            # coincidence (non-negotiable 17); the dry-run branch above had
            # already been corrected and would otherwise validate a device the
            # live run never used.
            device=device,
            is_sanity_check=is_sanity_check,
            resume_path=resume_path,
            allow_status=getattr(args, "allow_status", None),
            validate_only=bool(getattr(args, "validate_only", False)),
        )
        mode_str = "Sanity Check" if is_sanity_check else "Training"
        logger.info(f"{mode_str} completed: {result}")

        # Fail-fast: pipeline may return success=False without raising. Only an
        # explicit True is success; an absent flag is unknown, not a pass (#181). The rule
        # lives in one place so ``train-distributed`` applies it too.
        from spectramr.pipelines.result_status import failure_reason

        error_msg = failure_reason(result)
        if error_msg is not None:
            logger.error(f"{mode_str} pipeline reported failure: {error_msg}")
            sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed")
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def train_command(args: argparse.Namespace) -> None:
    """Start the training pipeline."""
    __common_train_setup(args, is_sanity_check=False)


def sanity_check_command(args: argparse.Namespace) -> None:
    """Start the sanity check pipeline."""
    __common_train_setup(args, is_sanity_check=True)


def infer_command(args: argparse.Namespace) -> None:
    """Run inference.

    Calls ``run_inference_pipeline`` (the SSOT inference entry point).
    The previous call site passed three kwargs (``model_path``,
    ``override_config``, ``override_model_type``) that the function
    never accepted — every invocation crashed with ``TypeError``. See
    ``TODO/audit/00_implementation_tracker.md`` BB8.
    """
    from spectramr.pipelines import run_inference_pipeline

    settings, device = begin_inference_run(
        args.config,
        args.device,
        pipeline="infer",
        checkpoint_path=args.checkpoint,
        from_yaml=getattr(args, "from_yaml", False),
    )

    try:
        with torch.no_grad():
            result = run_inference_pipeline(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                input_path=args.input,
                output_path=args.output,
                device=str(device),
                from_manifest_test_split=getattr(args, "from_manifest_test_split", False),
                from_yaml=getattr(args, "from_yaml", False),
                settings=settings,
            )
        logger.info(f"Inference completed: {result}")
    except Exception as e:
        logger.exception("Inference failed")
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


def infer_dataset_command(args: argparse.Namespace) -> None:
    """[DEPRECATED] Alias for the ``infer`` subcommand.

    Per TODO/audit/01_pipelines.md F1, ``src/pipelines/infer_dataset.py``
    has been removed; this command now resolves to the same
    ``run_inference_pipeline`` that ``infer`` uses. The subcommand is
    kept for cluster-job compatibility and emits a deprecation
    notice. Migrate scripts to ``python -m spectramr.cli infer ...``.
    """
    logger.warning(
        "`infer-dataset` is a deprecated alias for `infer`. "
        "Update your CLI invocation to `python -m spectramr.cli infer ...`."
    )
    from spectramr.pipelines.infer import run_inference_pipeline

    _, device = begin_inference_run(args.config, args.device, pipeline="infer_dataset")

    try:
        with torch.no_grad():
            result = run_inference_pipeline(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                input_path=args.input,
                output_path=args.output,
                device=str(device),
                batch_size=args.batch_size if hasattr(args, "batch_size") else None,
            )
        logger.info(f"Inference completed: {result}")
    except Exception as e:
        logger.exception("Inference failed")
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


def experiment_command(args: argparse.Namespace) -> None:
    """Run a named experiment: a thin wrapper over the training pipeline.

    Translates ``--experiment`` / ``--max-epochs`` / ``--checkpoint-interval``
    into config overrides (``training.output_dir`` / ``training.epochs`` /
    ``checkpoint.save_interval``) and runs the canonical
    ``run_training_pipeline`` — the same path as ``train``.

    Example:
        python src/main.py experiment \\
            --config experiments/training/config.yaml \\
            --experiment exp_gan_v1 \\
            --max-epochs 100 \\
            --checkpoint-interval 10
    """
    # 1. Load configuration FIRST — the accelerator init below reads
    # ``training.seed`` / ``training.deterministic`` from it. The old order
    # (init at 42/deterministic=True, then load) made both knobs silent
    # no-ops on this *training* path (pitfall #15).
    # Arm the substitution ledger before the config resolves, so a knob
    # the schema drops is recorded on this surface too rather than only on
    # the training path.
    from spectramr.core.execution_ledger import ExecutionLedger

    ExecutionLedger.begin_run(source=str(args.config))

    try:
        settings = TrainingSettings.from_yaml(str(args.config))
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    # 2. Translate the experiment knobs into config overrides on the SSOT
    #    settings, then run the canonical training pipeline. (The old
    #    ExperimentDirector path was dead-on-arrival: its ``validate()`` required
    #    generator/loss config the CLI never supplied, so it raised before any
    #    training ran — the real work was always ``run_training_pipeline``.)
    experiment_overrides = [
        f"training.epochs={args.max_epochs}",
        f"checkpoint.save_interval={args.checkpoint_interval}",
        f"training.output_dir=experiments/results/{args.experiment}",
    ]
    user_overrides = list(args.override) if getattr(args, "override", None) else []
    settings = apply_overrides(settings, experiment_overrides + user_overrides)
    _apply_log_level(settings)

    # 3. Initialize accelerator with the post-override config-resolved policy,
    # mirroring ``__common_train_setup``: CLI --seed > config.training.seed >
    # 42; determinism from ``training.deterministic`` (absent → True). Runs
    # after ``apply_overrides`` so ``--override training.deterministic=...``
    # is honored too.
    seed = args.seed or settings.run.seed
    initialize_accelerator(
        args.device,
        seed,
        deterministic=_resolved_determinism(settings),
        pipeline="experiment",
    )

    output_dir = f"experiments/results/{args.experiment}"
    logger.info(f"✅ Experiment '{args.experiment}'")
    logger.info(f"   Output dir : {output_dir}")
    logger.info(f"   Max epochs : {args.max_epochs}")
    logger.info(f"   Ckpt every : {args.checkpoint_interval} epoch(s)")

    # 4. Run training pipeline (the live path used by ``train``; lazy heavy
    #    import — see the module-top note).
    from spectramr.pipelines import run_training_pipeline

    resume_path = str(args.resume) if getattr(args, "resume", None) else None
    try:
        result = run_training_pipeline(
            settings,
            device=args.device,
            resume_path=resume_path,
            allow_status=getattr(args, "allow_status", None),
            validate_only=bool(getattr(args, "validate_only", False)),
        )
        logger.info(f"Experiment completed: {result}")
        if not (isinstance(result, dict) and result.get("success") is True):
            reason = result.get("error", "unknown") if isinstance(result, dict) else result
            logger.error(f"Experiment pipeline reported failure: {reason}")
            sys.exit(1)
    except Exception as e:
        logger.exception("Experiment failed")
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


def main() -> None:
    """Deprecated entry point — delegates to :func:`spectramr.cli.app.main`.

    ``spectramr.main:main`` historically owned a SECOND argparse parser that
    diverged from the real console-script parser in ``spectramr.cli.app``. As of
    the 2026-05-29 entry-point unification, ``spectramr.cli.app`` is the single
    parser and defines every subcommand (incl. ``train-distributed``, ``infer``,
    ``infer-dataset``, ``experiment``). This function now just warns and
    delegates so old invocations (``python -m spectramr.main <cmd> ...``) keep
    working. The module's import-time environment setup (cache root, thread
    isolation, ``PYTORCH_CUDA_ALLOC_CONF`` — all before ``import torch``) still
    runs because importing this module triggers it.

    The per-command functions (``train_command``, ``sanity_check_command``,
    ``infer_command``, ``infer_dataset_command``, ``experiment_command``,
    ``apply_overrides``) remain importable; ``spectramr.cli.app`` calls them
    directly. The ``hpo`` verb is handled entirely by ``spectramr.cli.app``'s own
    ``hpo_cmd`` (which drives ``HPOUseCase`` directly), so no ``hpo_command``
    lives here.
    """
    warnings.warn(
        "`python -m spectramr.main` / `spectramr.main:main` is deprecated; use the "
        "`spectramr` console script (spectramr.cli.app:main). Delegating now.",
        DeprecationWarning,
        stacklevel=2,
    )
    from spectramr.cli.app import main as cli_main

    raise SystemExit(cli_main())


if __name__ == "__main__":
    main()
