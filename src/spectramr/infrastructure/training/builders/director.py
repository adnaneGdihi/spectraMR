"""Training Environment Director

Orchestrates all builders to create a complete training environment.
This is the high-level entry point for building training components.
"""

import logging

import torch

from spectramr.core.compute_device import resolve_torch_device
from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)
from spectramr.infrastructure.distributed.strategy_registry import (
    ParallelContext,
    ParallelRuntime,
    resolve_parallel_strategy,
)

from .compile_apply import CompileRecord, apply_compile, compile_is_enabled
from .compile_placement import resolve_compile_placement
from .data_builder import DataBuilder
from .environment import TrainingEnvironment
from .infrastructure_builder import InfrastructureBuilder
from .loss_builder import LossBuilder
from .model_builder import ModelBuilder
from .optimization_builder import OptimizationBuilder
from .physics_builder import PhysicsBuilder

logger = logging.getLogger(__name__)


class TrainingEnvironmentDirector:
    """High-level orchestrator that builds complete training environment.

    Coordinates all specialized builders in the correct order to create
    an immutable TrainingEnvironment with all components ready for training.

    Attributes:
        _config: Training configuration
        _device: Resolved device for training

    Example:
        >>> director = TrainingEnvironmentDirector(config)
        >>> env = director.build_environment()
        >>> # Use environment in strategy
        >>> strategy = GANTrainingStrategy(env=env)
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize TrainingEnvironmentDirector.

        Args:
            config: Immutable training configuration
        """
        config = ctx.config
        self._config = config
        self._device = self._resolve_device()
        logger.info(f"TrainingEnvironmentDirector initialized on {self._device}")

    def _resolve_parallel(self):
        """Resolve the strategy plugin and its context, or ``(None, None)``.

        ``None`` for single-process so the two hook sites stay no-ops rather than
        paying for a plugin that does nothing -- and so an arm with no
        ``parallel:`` block behaves exactly as it did before this existed.
        """
        parallel = getattr(self._config, "parallel", None)
        if parallel is None or parallel.strategy == "none":
            return None, None
        plugin = resolve_parallel_strategy(parallel.strategy)
        ctx = ParallelContext(config=self._config, device=self._device, parallel=parallel)
        logger.info("[Parallel] strategy=%s", plugin.name)
        return plugin, ctx

    def _resolve_compile_placement(self):
        """Where compilation goes for this arm's declared strategy.

        Reads ``parallel.strategy`` and, for DeepSpeed, ``zero_stage`` -- the
        placement depends on the stage, so it cannot be a class attribute on the
        plugin.
        """
        parallel = getattr(self._config, "parallel", None)
        deepspeed = getattr(parallel, "deepspeed", None)
        return resolve_compile_placement(
            getattr(parallel, "strategy", None),
            zero_stage=getattr(deepspeed, "zero_stage", None),
        )

    def build_environment(self) -> TrainingEnvironment:
        """Build complete training environment in correct order.

        Orchestrates all builders to create models, optimizers, losses,
        physics operators, data loaders, and metrics.

        Returns:
            TrainingEnvironment: Immutable container with all components

        Raises:
            ValueError: If any builder fails validation
        """
        logger.info("Building training environment...")

        # Resolved FIRST so a refused combination fails here rather than after
        # the models, optimizers and dataloaders have been built on a cluster
        # node. The placement is a fact about the strategy, not a knob -- see
        # compile_placement for the measurements behind each row.
        placement = self._resolve_compile_placement()
        wants_compile = compile_is_enabled(self._config)
        if wants_compile and placement.refused:
            raise RuntimeError(placement.refusal)
        if wants_compile and placement.advisory:
            logger.warning(placement.advisory)
        compile_record = CompileRecord()

        # 1. Models (must come first - needed by optimizers)
        logger.info("Step 1/6: Building models...")
        model_builder = (
            ModelBuilder(self._config, self._device)
            .build_generator()
            .build_discriminator()
            .build_encoder_decoder()
            .validate()
            # No .compile() here. Compilation is placed per strategy now
            # (compile_placement); at this point the wraps have not happened,
            # which is the wrong point for every strategy but DeepSpeed.
            # It also means build_ema deep-copies the BARE model, so the
            # shadow tracks weights rather than an OptimizedModule.
            .build_ema()
        )
        models = model_builder.build()
        ema_model = model_builder.ema
        logger.info(f"Models created: {list(models.keys())}")

        # 1b. Parallelism, Stage A — BEFORE optimizers exist.
        #
        # FSDP must wrap here: FSDP.__init__ flattens parameters into a shard and
        # re-points their storage, so an optimizer built first would receive
        # shard-shaped gradients against full-shape moment buffers (a shape error
        # on the first step, or a silently wrong update under `foreach`) and
        # would allocate full-size state, defeating the point of sharding.
        # SyncBatchNorm also belongs here, since it rebuilds BatchNorm modules.
        parallel_plugin, parallel_ctx = self._resolve_parallel()
        if parallel_plugin is not None:
            models = parallel_plugin.prepare_models(models, parallel_ctx)

        # FSDP compiles here -- torch.compile(FSDP(m)), the recommended order --
        # and DeepSpeed compiles here because initialize() must receive an
        # already-compiled module (Stage A is a no-op for it).
        if placement.stage == "after_stage_a":
            models, compile_record = apply_compile(
                models, self._config, stage="after_stage_a", device=self._device
            )

        # 2. Optimization (depends on models)
        logger.info("Step 2/6: Building optimization components...")
        optimizers, schedulers, scaler = (
            OptimizationBuilder(self._config, models=models)
            .build_optimizers()
            .build_schedulers()
            .build_grad_scaler()
            .validate()
            .build()
        )
        logger.info(f"Optimizers created: {list(optimizers.keys())}")

        # 2b. Parallelism, Stage B — AFTER optimizers exist.
        #
        # DP/DDP wrap here on purpose: neither changes parameter identity, so the
        # optimizers stay valid, and wrapping earlier would make every downstream
        # consumer that expects a bare module (count_parameters, the optimizer
        # builder's model selection) see a DistributedDataParallel instead.
        parallel_runtime = None
        if parallel_plugin is not None:
            result = parallel_plugin.adopt(models, optimizers, schedulers, parallel_ctx)
            models, optimizers, schedulers = (
                result.models,
                result.optimizers,
                result.schedulers,
            )
            parallel_runtime = ParallelRuntime(
                strategy=parallel_plugin.name,
                step_policy=result.step_policy,
                checkpoint_adapter=parallel_plugin.checkpoint_adapter(parallel_ctx),
                provenance=result.provenance,
            )

        # Compile last for the strategies that wrap late. For ddp this is what
        # lets dynamo see the wrapper and engage DDPOptimizer, which splits the
        # graph at allreduce bucket boundaries; for the single-device path it
        # additionally keeps the optimizer, built above, looking at a bare
        # module (#2174).
        if placement.stage == "after_stage_b":
            models, compile_record = apply_compile(
                models, self._config, stage="after_stage_b", device=self._device
            )

        # 3. Losses (independent)
        logger.info("Step 3/6: Building loss functions...")
        losses = (
            LossBuilder(self._config, self._device)
            .build_reconstruction_losses()
            .build_adversarial_losses()
            .build_physics_losses()
            .build_regularization_losses()
            .validate()
            .build()
        )
        logger.info(f"Losses created: {list(losses.keys())}")

        # 4. Physics (independent)
        logger.info("Step 4/6: Building physics operators...")
        physics = (
            PhysicsBuilder(self._config, self._device)
            .build_fft_transformer()
            .build_data_consistency()
            .build_coil_sensitivity()
            .validate()
            .build()
        )
        logger.info(f"Physics components created: {list(physics.keys())}")

        # 5. Data loaders (independent)
        logger.info("Step 5/6: Building data loaders...")
        data_loaders = (
            DataBuilder(self._config)
            .build_train_val_loaders()
            .build_test_loader()
            .validate()
            .build()
        )
        logger.info(f"Data loaders created: {list(data_loaders.keys())}")

        # Sampler substitution (DistributedSampler) belongs to the strategy:
        # only the process-group-backed ones need it, and only they know it.
        if parallel_plugin is not None:
            data_loaders = parallel_plugin.prepare_data_loaders(data_loaders, parallel_ctx)

        # 6. Metrics (independent)
        logger.info("Step 6/6: Building metrics...")
        metrics = (
            InfrastructureBuilder(self._config, self._device).build_metrics().validate().build()
        )
        logger.info(f"Metrics created: {list(metrics.keys())}")

        # Return immutable environment
        env = TrainingEnvironment(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            losses=losses,
            physics=physics,
            data_loaders=data_loaders,
            parallel=parallel_runtime,
            compile=compile_record,
            metrics=metrics,
            scaler=scaler,
            device=self._device,
            config=self._config,
            ema=ema_model,
        )

        logger.info("Training environment built successfully!")
        return env

    def _resolve_device(self) -> torch.device:
        """Get device from DI container or config.

        Returns:
            torch.device: Device for training (cuda/cpu)
        """
        try:
            from spectramr.domain.interfaces.service_interfaces import (
                IDeviceManager as IDeviceService,
            )
            from spectramr.infrastructure.di import resolve_service

            device_service = resolve_service(IDeviceService)
            device = device_service.get_device()
            logger.info(f"Device resolved from DI container: {device}")
            return device
        except Exception as e:
            # Fallback to config or default
            logger.warning(
                f"Failed to resolve device from DI container: {e}. Using config/default."
            )
            # [SSOT] Prioritize top-level device config (canonical source),
            # then training.device. Resolution + the accelerated-run contract
            # belong to spectramr.core.compute_device: training is a heavy
            # pipeline, so no-accelerator RAISES rather than degrading to CPU.
            # The removed "CUDA requested but not available. Falling back to
            # CPU." branch was one of eight silent downgrades that let a
            # GPU-less node run ~100x slower and still report success.
            device_str = self._config.run.device
            if not device_str or device_str == "auto":
                device_str = (
                    self._config.training.device
                    if self._config.training and self._config.training.device
                    else "auto"
                )

            decision = resolve_torch_device(device_str, pipeline="train", source="run.device")
            return torch.device(decision.device)
