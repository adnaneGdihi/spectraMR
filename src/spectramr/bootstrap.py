import logging
from pathlib import Path

from spectramr.config.settings import TrainingSettings
from spectramr.core.compute_device import resolve_torch_device
from spectramr.data.metadata.path_resolver import PathResolver

# Interfaces
from spectramr.domain.interfaces.checkpoint_service_interface import ICheckpointService

# Interfaces for Phase 4 Services
from spectramr.domain.interfaces.service_interfaces import (
    IDeviceService,
    ILoggingService,
    IMemoryOptimizationService,
    IMetricsService,
)
from spectramr.infrastructure.di.di_container import (
    DIContainer,
    init_container,
    register_service,
)
from spectramr.infrastructure.logging import (
    LoggingServiceFactory,
    MetricsTracker,
)

# Factories
from spectramr.infrastructure.checkpointing.facade import CheckpointStore
from spectramr.infrastructure.logging.sinks import resolve_run_dir
from spectramr.infrastructure.services.device_manager import DeviceManager

# Phase 4: Additional Services (Memory)
from spectramr.infrastructure.services.memory_optimization_service import (
    MemoryOptimizationService,
)
from spectramr.infrastructure.validation.config_validation import (
    validate_config_at_startup,
)

logger = logging.getLogger(__name__)


def _validate_data_availability_at_startup(config: TrainingSettings) -> None:
    """Phase 3c: Validate data is available before starting training.

    This is a fail-fast check that runs at startup, BEFORE any training setup.

    Args:
        config: TrainingSettings with data configuration

    Raises:
        ValueError: If data not available or configuration invalid
    """
    try:
        # v6.0: Infer dataset type from ``config.training.strategy_class``.
        # ``training`` is now schema-enforced; an absent ``strategy_class``
        # is a configuration error, not "use reconstruction silently"
        # (CLAUDE.md pitfall #9) — see
        # ``TODO/audit/15_entry_points_bootstrap_cli.md`` F7.
        if config.training is None or not config.training.strategy_class:
            raise ValueError(
                "[STARTUP] config.training.strategy_class is required to "
                "decide dataset_type for data-availability validation. "
                "Set training.strategy_class in the YAML (or set "
                "training.training_mode and let the schema route it)."
            )
        strategy_class = config.training.strategy_class
        if "gan" in strategy_class.lower():
            dataset_type = "gan"
        elif "diffusion" in strategy_class.lower():
            dataset_type = "diffusion"
        else:
            dataset_type = "reconstruction"

        requires_data = dataset_type not in ("synthetic", "debug")

        # Also honour the YAML-declared dataset_type — when the user
        # explicitly selects ``data.dataset_type: synthetic`` (e.g. the
        # MRF dictionary-generation arms) the data root is irrelevant.
        declared_dataset_type = getattr(getattr(config, "data", None), "dataset_type", None)
        if declared_dataset_type in ("synthetic", "debug"):
            requires_data = False

        if not requires_data:
            logger.info(f"[STARTUP] Skipping data validation for dataset_type={dataset_type}")
            return

        # Check data configuration exists
        if not hasattr(config, "data"):
            logger.warning("[STARTUP] No data configuration found")
            return

        data_config = config.data
        data_root = data_config.source.root

        if not data_root:
            logger.warning("[STARTUP] No data_root configured, skipping data check")
            return

        # Validate data availability. Resolve the data_root through PathResolver —
        # the SAME rule the dataset builder uses for a manifest's embedded data_root
        # (src/spectramr/data/builders/manifest_index.py::build_index_from_manifest) —
        # so the pre-flight and the loader agree. The old raw ``Path(data_root).exists()``
        # plus an ad-hoc ``SPECTRAMR_DATA_ROOT/data_root`` join diverged from that rule:
        # it raised "Data root not found" for a config whose data the loader would have
        # located via PathResolver's PROJECT_ROOT / SPECTRAMR_DATA_ROOT / legacy-prefix
        # handling (and its ``./databases`` default double-joined ``databases/...`` roots).
        resolved = PathResolver.resolve(data_root)
        logger.info(f"[STARTUP] Validating data at: {data_root} -> {resolved}")

        if not Path(resolved).exists():
            raise ValueError(
                f"Data root not found: {data_root} (resolved: {resolved})\n"
                f"Set SPECTRAMR_DATA_ROOT / PROJECT_ROOT to your data directory, or check "
                f"your data configuration / download the datasets."
            )

        logger.info("[STARTUP] ✅ Data validation passed")

    except Exception as e:
        # Fail-loud: log for the run archive, then propagate. (A dead
        # ImportError-skip arm was removed 2026-07-01 — no import remained
        # inside the try, and silently skipping validation is pitfall #9.)
        logger.error(f"[STARTUP] Data validation failed: {e}")
        raise


def build_container(
    config: TrainingSettings,
    device: str | None = None,
    *,
    pipeline: str = "train",
    run_dir: str | Path | None = None,
) -> DIContainer:
    """
    Bootstrap the Dependency Injection container with strict validation.

    **Validation**: Configuration is validated at startup to catch errors early.

    **Phase 3 Data Validation**: Checks data availability before training starts.

    **Device**: resolved via ``spectramr.core.compute_device.resolve_torch_device``.
    A heavy ``pipeline`` with no accelerator raises ``AcceleratorRequiredError``
    rather than degrading to CPU.
    """
    # Validate configuration FIRST before any service creation.
    #
    # This used to call ``validate_config_at_startup`` alone, which runs
    # ``ValidatorRegistry`` and nothing else — so a training run had NEVER been
    # through ``ConfigHealthChecker`` or the compatibility matrix, both of which
    # `spectramr audit` has always run. The audit and the run were checking
    # different things, in both directions. The witness gate runs the union, and
    # the registry adapter wraps ``ConfigValidator.validate`` so the previous
    # raise semantics (and the training-mode dispatchability check) are preserved.
    from spectramr.infrastructure.validation.witness import (
        Tier,
        WitnessSubject,
        run_witnesses,
    )

    validate_config_at_startup(config)

    # ...then run the witnesses this path was missing. Deliberately NOT blocking:
    # ConfigHealthChecker has never gated a training run, and promoting its
    # findings to a hard raise here would take arms offline as a side effect of
    # adding observability. This is the ratchet's first rung — see
    # docs/explanation/execution_ledger.md.
    _subject = WitnessSubject.for_audit(config_path=None, settings=config)
    for _v in run_witnesses(_subject, tiers=frozenset({Tier.T0, Tier.T1})):
        if not _v.passed:
            logger.warning("[witness] %s: %s", _v.witness_name, _v.message)

    # Populate Model Registry (Discovery)
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    # Phase 3c: Validate data availability at startup (fail-fast)
    _validate_data_availability_at_startup(config)

    container = init_container()
    container.clear()

    # 1. Device Manager
    # Device priority: CLI arg > config.run.device > config.training.device > "auto".
    # Resolution + the accelerated-run contract are delegated to the SSOT policy
    # (spectramr.core.compute_device): an unset device means "auto", and "auto"
    # with no accelerator RAISES for a heavy pipeline instead of degrading to
    # CPU. The two former fallbacks here ("no device anywhere -> cpu" and
    # "auto -> cpu") are what made DeviceManager's own CUDA guard unreachable:
    # bootstrap flattened "auto" to "cpu" before DeviceManager ever saw it, so a
    # GPU-less node produced a silent ~100x-slower run. See CLAUDE.md pitfall #9.
    device_arg = device or config.run.device
    if not device_arg and config.training:
        device_arg = config.training.device

    decision = resolve_torch_device(
        device_arg,
        pipeline=pipeline,
        source="cli" if device else "run.device",
    )
    logger.info(
        "[STARTUP] device=%s accelerated=%s (source=%s, pipeline=%s)",
        decision.device,
        decision.accelerated,
        decision.source,
        pipeline,
    )

    device_manager = DeviceManager(preferred_device=decision.device)
    register_service(IDeviceService, device_manager)
    register_service(DeviceManager, device_manager)

    # 2. Logging
    # Logging config is REQUIRED by schema
    if config.logging is None:
        raise ValueError("config.logging is required but not provided")

    run_dir = run_dir or resolve_run_dir(config, pipeline == "sanity_check")
    logging_service = LoggingServiceFactory.create(
        config.logging, model_type=config.model.model_type, run_dir=run_dir
    )
    register_service(ILoggingService, logging_service)

    # 3. Artifact Root & Checkpoints
    # config.logging.log_dir is standard Pydantic (required)
    artifact_root = Path(config.logging.sinks.dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # One checkpoint owner (infrastructure.checkpointing); the store is stateless, so
    # the root ``checkpoint:`` block being absent does not stop the container build.
    register_service(ICheckpointService, CheckpointStore())

    # 4. Metrics
    # Use MetricsTracker directly as per new architecture
    metrics_output_dir = str(artifact_root / "metrics")
    if hasattr(config, "metrics") and config.metrics and hasattr(config.metrics, "output_dir"):
        if config.metrics.output_dir:
            metrics_output_dir = config.metrics.output_dir

    # Resolve model type from canonical location
    # config.model.model_type is REQUIRED by schema
    if config.model is None:
        raise ValueError("config.model is required but not provided")
    if config.model.model_type is None:
        raise ValueError("config.model.model_type is required but not provided")

    model_type = config.model.model_type

    # Metrics service initialization (simplified constructor)
    # MetricsTracker delegates metric computation to spectramr.core.metrics
    save_images = False
    if hasattr(config, "validation") and config.validation:
        save_images = config.validation.visualization.enabled

    # Also check logging config for save_validation_images flag
    if hasattr(config, "logging") and config.logging:
        save_images = save_images or config.logging.images.save_validation

    # Final OR with log_validation_images (backward compat)
    save_images = save_images or config.logging.images.log_validation

    logging_service.log_info(
        f"[Bootstrap] Metrics image saving: save_images={save_images} "
        f"(validation.enable_visualization={config.validation.visualization.enabled if config.validation else False}, "
        f"logging.save_validation_images={config.logging.images.save_validation}, "
        f"logging.log_validation_images={config.logging.images.log_validation})"
    )

    metrics_service = MetricsTracker(
        output_dir=metrics_output_dir,
        save_images=save_images,
        image_format="png",
        device=device_arg,
        model_type=model_type,
    )
    # Register as IMetricsService
    # MetricsTracker now properly implements IMetricsService interface
    register_service(IMetricsService, metrics_service)
    register_service(MetricsTracker, metrics_service)
    logging_service.attach_artifact_sink("metrics_csv", lambda: metrics_service.validation_csv)
    logging_service.attach_artifact_sink("loss_csv", lambda: metrics_service.training_csv)

    # ========== PHASE 4: Additional Services Registration ==========

    # Memory Optimization Service
    memory_service = MemoryOptimizationService()
    memory_service.setup_memory_optimization()  # Setup Triton cache and memory settings
    register_service(IMemoryOptimizationService, memory_service)
    register_service(MemoryOptimizationService, memory_service)

    # NB: ``IManifestService``, ``IErrorHandlingService``,
    # ``IModelManagementService``, ``IModelCompilationService`` and
    # ``IProfilingService`` used to be registered here as "available
    # but optional". Audit 15 F2 found that nothing in the training
    # path actually resolves them — they were CLAUDE.md pitfall #9
    # surface (advertised contract, no consumer). They have been
    # removed. If a real consumer appears, register the service back
    # **and** call ``container.resolve(IService)`` in that consumer so
    # the wiring is testable end-to-end.

    # ========== End PHASE 4 Services ==========

    # Register Config itself
    register_service(TrainingSettings, config)

    # ========== PHASE 4.5: Service Audit at Startup ==========
    # Verify all required services are properly registered (fail-fast validation)
    from spectramr.infrastructure.di.service_audit import verify_startup_services

    verify_startup_services(container, fail_on_missing=True)

    # ========== PHASE 5: Domain Validation at Startup ==========
    # Schema-only check that loss functions match input domain (k-space,
    # image, or latent). Cheap to run; must precede loss instantiation
    # so a domain-mismatch failure doesn't pay the loss-construction
    # cost first — see ``TODO/audit/15_entry_points_bootstrap_cli.md``
    # F11.
    from spectramr.infrastructure.domain_validation import verify_startup_loss_domains

    input_domain = config.model.input_type
    verify_startup_loss_domains(config.losses, input_domain, fail_on_error=True)

    # ========== PHASE 5.5: Loss Audit at Startup ==========
    # Verify every configured loss name resolves to a registered loss
    # (fail-fast). This is a registry-membership pre-check that constructs
    # nothing — so the INFO-logging suppression that used to guard it (when
    # the audit built losses itself, duplicating the director's "Creating
    # Loss:" messages) is no longer needed.
    from spectramr.infrastructure.loss_audit import verify_startup_losses

    verify_startup_losses(config.losses)

    return container
