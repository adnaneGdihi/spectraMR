"""The built-in parallel-strategy plugins.

Ported from ``pipelines.parallel.apply_parallelism``'s ``if/elif`` chain, with
three behaviour fixes carried in:

* ``strategy: 'fsdp'`` **dispatches** instead of raising ``ValueError`` after the
  whole training environment was already built.
* ``sync_batch_norm`` is honoured only for process-group-backed strategies. The
  old code converted BatchNorm for ``dp`` as well, producing ``SyncBatchNorm``
  modules that cannot run under ``DataParallel`` (no process group). The schema
  now rejects that pairing outright, and this is the second line of defence.
* FSDP no longer degrades to an unwrapped model with a warning when the process
  group is missing -- it raises, matching what DDP has always done for the same
  misconfiguration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.distributed.strategy_registry import (
    ParallelContext,
    ParallelizationResult,
    register_parallel_strategy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DDPStrategy",
    "DataParallelStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    "NoParallelStrategy",
]


def _require_process_group(strategy: str) -> None:
    """Raise unless a process group exists.

    No silent fallback: a user who forgot ``torchrun`` otherwise gets a
    full-length single-GPU run that reports success and claims, in its own
    provenance, to have been distributed.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError(
            f"parallel.strategy={strategy!r} requires an initialised process "
            "group. Launch with:\n"
            "    torchrun --nproc_per_node=N -m spectramr.cli train-distributed "
            "--config <arm>.yaml\n"
            "Refusing to fall back to single-process: the run would report "
            "success while training something the config did not ask for."
        )


def _convert_sync_batchnorm(models: dict[str, Any]) -> None:
    """In-place SyncBatchNorm conversion.

    Safe to do before optimizers exist -- and it must be, because
    ``convert_sync_batchnorm`` rebuilds BatchNorm submodules. (It happens to
    reuse the same ``Parameter`` objects, so an optimizer built earlier would
    survive; relying on that is not worth it when Stage A is available.)
    """
    for name in list(models):
        module = models[name]
        if module is not None:
            models[name] = nn.SyncBatchNorm.convert_sync_batchnorm(module)
            logger.info("[Parallel] Converted %s BatchNorm -> SyncBatchNorm", name)


def _wrap_with_distributed_samplers(loaders: dict[str, Any]) -> dict[str, Any]:
    """Give every loader a DistributedSampler.

    Delegates the DataLoader reconstruction to the data layer so this module
    never constructs one directly (pitfall #11).
    """
    from spectramr.data.builders.data_loader_builder import wrap_with_distributed_sampler

    out = dict(loaders)
    for split, loader in loaders.items():
        if loader is None:
            continue
        out[split] = wrap_with_distributed_sampler(loader, shuffle=(split == "train"))
        logger.info("[Parallel] Wrapped %r DataLoader with DistributedSampler", split)
    return out


class _BaseStrategy:
    """Defaults: do nothing, and hand back an ordinary AMP step policy."""

    name = "none"

    #: Does this backend need an initialized ``torch.distributed`` process group?
    #:
    #: The flag exists so a *caller* can ask the question before paying for a
    #: run, rather than restating the answer as ``strategy != "none"`` -- which
    #: would be wrong for ``dp`` (``nn.DataParallel`` is genuinely single-process)
    #: and would make this invariant have two owners (non-negotiable 17). It is
    #: pinned to the ``_require_process_group`` call sites by an AST test, so a
    #: backend that calls the helper without setting the flag -- or sets it
    #: without calling -- fails in the same commit that introduces the drift.
    requires_process_group = False

    def prepare_models(self, models: dict[str, Any], ctx: ParallelContext) -> dict[str, Any]:
        return models

    def adopt(
        self,
        models: dict[str, Any],
        optimizers: dict[str, Any],
        schedulers: dict[str, Any],
        ctx: ParallelContext,
    ) -> ParallelizationResult:
        return ParallelizationResult(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            provenance=self.provenance(ctx),
        )

    def prepare_ema(self, ema: Any, models: dict[str, Any], ctx: ParallelContext) -> None:
        """Stage A, after :meth:`prepare_models`: adapt the EMA shadow. Default: nothing."""

    def prepare_data_loaders(self, loaders: dict[str, Any], ctx: ParallelContext) -> dict[str, Any]:
        return loaders

    def checkpoint_adapter(self, ctx: ParallelContext) -> Any:
        """Default: ``state_dict()`` is already complete and rank 0 alone writes.

        True for ``none``, and true for ``dp``/``ddp`` because both *replicate*
        rather than shard -- rank 0's copy IS the model. Only the sharding
        strategies override this.
        """
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            DefaultCheckpointAdapter,
        )

        return DefaultCheckpointAdapter()

    def provenance(self, ctx: ParallelContext) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "world_size": ctx.world_size,
            "timeout_seconds": ctx.parallel.timeout_seconds,
        }


class NoParallelStrategy(_BaseStrategy):
    """Single process, single device."""

    name = "none"


class DataParallelStrategy(_BaseStrategy):
    """``nn.DataParallel`` — single process, N GPUs, no process group."""

    name = "dp"

    def adopt(self, models, optimizers, schedulers, ctx):
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            raise RuntimeError(
                f"parallel.strategy='dp' but {gpu_count} GPU(s) are visible. "
                "DataParallel over one device is a no-op that still pays its "
                "scatter/gather cost; declare strategy: 'none' instead. "
                "(This used to warn and continue, so the arm's provenance "
                "claimed multi-GPU while it ran on one.)"
            )
        wrapped = dict(models)
        for name, module in models.items():
            if module is not None:
                wrapped[name] = nn.DataParallel(
                    module,
                    device_ids=ctx.parallel.device_ids,
                    output_device=ctx.parallel.output_device,
                )
                logger.info("[Parallel] Wrapped %s with DataParallel", name)
        return ParallelizationResult(
            models=wrapped,
            optimizers=optimizers,
            schedulers=schedulers,
            provenance=self.provenance(ctx),
        )


def wrap_with_ddp(module: nn.Module, parallel: Any, device: Any) -> nn.Module:
    """The one ``DistributedDataParallel`` construction, shared with strategy-owned modules.

    Construction broadcasts rank 0's parameters, so every rank must call it at the same
    point in the run (#2629).
    """
    from torch.nn.parallel import DistributedDataParallel

    _require_process_group("ddp")
    device_ids = (
        [device.index]
        if getattr(device, "type", None) == "cuda" and device.index is not None
        else None
    )
    return DistributedDataParallel(
        module,
        device_ids=device_ids,
        find_unused_parameters=parallel.find_unused_parameters,
        gradient_as_bucket_view=parallel.gradient_as_bucket_view,
        static_graph=parallel.static_graph,
    )


class DDPStrategy(_BaseStrategy):
    """``DistributedDataParallel`` — one process per device."""

    name = "ddp"
    requires_process_group = True

    def prepare_models(self, models, ctx):
        if ctx.parallel.sync_batch_norm:
            _convert_sync_batchnorm(models)
        return models

    def adopt(self, models, optimizers, schedulers, ctx):
        _require_process_group(self.name)
        wrapped = dict(models)
        for name, module in models.items():
            if module is None:
                continue
            wrapped[name] = wrap_with_ddp(module, ctx.parallel, ctx.device)
            logger.info("[Parallel] Wrapped %s with DDP", name)
        return ParallelizationResult(
            models=wrapped,
            optimizers=optimizers,
            schedulers=schedulers,
            provenance=self.provenance(ctx),
        )

    def prepare_data_loaders(self, loaders, ctx):
        return _wrap_with_distributed_samplers(loaders)

    def provenance(self, ctx):
        record = super().provenance(ctx)
        record.update(
            {
                "find_unused_parameters": ctx.parallel.find_unused_parameters,
                "gradient_as_bucket_view": ctx.parallel.gradient_as_bucket_view,
                "static_graph": ctx.parallel.static_graph,
                "sync_batch_norm": ctx.parallel.sync_batch_norm,
            }
        )
        return record


class FSDPStrategy(_BaseStrategy):
    """FSDP2 ``fully_shard`` — shards in **Stage A**, before optimizers.

    Parameters become DTensor shards in place. An optimizer built beforehand
    would hold full-size moment buffers against shard-shaped parameters, so the
    shard must precede it. No step policy is supplied: ``clip_grad_norm_`` over
    DTensor gradients already reduces the norm across shards.
    """

    name = "fsdp"
    requires_process_group = True

    def prepare_models(self, models, ctx):
        _require_process_group(self.name)
        from spectramr.infrastructure.distributed.fsdp_wrap import maybe_wrap_with_fsdp

        if ctx.parallel.sync_batch_norm:
            _convert_sync_batchnorm(models)
        for name in list(models):
            if models[name] is not None:
                models[name] = maybe_wrap_with_fsdp(
                    models[name], ctx.parallel, ctx.config.optimization.precision, ctx.device
                )
                logger.info("[Parallel] Sharded %s with FSDP2", name)
        return models

    def prepare_ema(self, ema, models, ctx):
        # The shadow is a pre-shard deepcopy; sharded alike, the blend and the
        # validation swap stay shard-local (a plain shadow cannot blend DTensors).
        from spectramr.infrastructure.distributed.fsdp_wrap import shard_like

        if ema is not None and models.get("generator") is not None:
            shard_like(ema.module, models["generator"])
            logger.info("[Parallel] Sharded the EMA shadow like the generator")

    def adopt(self, models, optimizers, schedulers, ctx):
        return ParallelizationResult(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            provenance=self.provenance(ctx),
        )

    def prepare_data_loaders(self, loaders, ctx):
        return _wrap_with_distributed_samplers(loaders)

    def checkpoint_adapter(self, ctx):
        # state_dict() returns THIS RANK'S SHARD by default -- a file with
        # plausible key names and 1/N of the weights.
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            FSDPCheckpointAdapter,
        )

        return FSDPCheckpointAdapter()

    def provenance(self, ctx):
        from spectramr.infrastructure.distributed.fsdp_wrap import resolve_fsdp_precision

        record = super().provenance(ctx)
        fsdp = ctx.parallel.fsdp
        precision = resolve_fsdp_precision(fsdp, ctx.config.optimization.precision)
        record["fsdp"] = {
            "sharding_strategy": fsdp.sharding_strategy,
            "auto_wrap_policy": fsdp.auto_wrap_policy,
            "transformer_layer_cls": list(fsdp.transformer_layer_cls),
            "min_num_params": fsdp.min_num_params,
            "cpu_offload": fsdp.cpu_offload,
            "mixed_precision": precision,
            # The class the training loop builds for this precision (mixed_precision.py).
            "grad_scaler": "ComplexGradScaler" if precision == "fp16" else None,
            "reduce_dtype": fsdp.reduce_dtype,
            "implementation": "fsdp2",
        }
        return record


#: Optimizer keys the multi-engine path has been reasoned through for: the GAN pair.
#: Any other multi-optimizer set is not covered by the freeze contract.
_MULTI_ENGINE_KEYS = frozenset({"opt_g", "opt_d"})


def _require_multi_engine_admissible(ds_cfg: Any, optimizers: Mapping[str, Any]) -> None:
    """Refuse a multi-engine arm outside what the two-engine GPU test covers (#2631).

    Config-only, so every rank raises identically before ``initialize``. The D
    cadence is iteration-based (``_resolve_disc_updates``), hence rank-uniform;
    the critic freeze keeps the generator's objective out of D's reduce buckets.
    ``allow_multi_engine`` stays the gate until the GPU test has passed.
    """
    if not ds_cfg.allow_multi_engine:
        raise RuntimeError(
            f"parallel.strategy='deepspeed' with {len(optimizers)} optimizers "
            f"({sorted(optimizers)}). deepspeed.initialize returns ONE engine per "
            "optimizer, and engine.step() issues collectives -- an arm whose "
            "discriminator steps on a different cadence than the generator will "
            "DEADLOCK rather than error. Set parallel.deepspeed.allow_multi_engine: "
            "true only after tests/unit/infrastructure/distributed/"
            "test_deepspeed_multi_engine_gpu.py (#2631) has passed for your zero_stage."
        )
    if ds_cfg.zero_stage >= 3:
        raise RuntimeError(
            "parallel.deepspeed.zero_stage=3 with more than one engine is refused: "
            "ZeRO-3 gathers parameters during forward, which the discriminator "
            "freeze does not cover, and no stage-3 run has passed the two-engine "
            "test (#2631). Use zero_stage 0-2."
        )
    unexpected = set(optimizers) - _MULTI_ENGINE_KEYS
    if unexpected:
        raise RuntimeError(
            f"parallel.deepspeed with multiple engines is only admitted for the GAN "
            f"pair {sorted(_MULTI_ENGINE_KEYS)}; got {sorted(unexpected)} as well. "
            "Their step cadence and critic freeze are not covered by the test (#2631)."
        )


class DeepSpeedStrategy(_BaseStrategy):
    """DeepSpeed ZeRO — the engine adopts the optimizer in **Stage B**.

    ``deepspeed.initialize`` *consumes* an already-built optimizer, which is
    exactly what Stage B provides. Handing it the optimizer (rather than
    declaring one in the ds_config) keeps TTUR, the LR multipliers and
    ``resolve_scheduler_spec`` describing what actually ran.
    """

    name = "deepspeed"
    requires_process_group = True

    def prepare_models(self, models, ctx):
        # Nothing in Stage A: unlike FSDP, DeepSpeed does not re-point parameter
        # storage until initialize(), and initialize() needs the optimizer.
        return models

    def adopt(self, models, optimizers, schedulers, ctx):
        _require_process_group(self.name)
        from spectramr.infrastructure.distributed.deepspeed_backend import (
            DeepSpeedStepPolicy,
            build_deepspeed_config,
            initialize_deepspeed_engine,
        )
        from spectramr.infrastructure.distributed.deepspeed_backend.cpu_adam import (
            prepare_client_optimizer,
        )

        ds_cfg = ctx.parallel.deepspeed
        if len(optimizers) > 1:
            _require_multi_engine_admissible(ds_cfg, optimizers)

        config = build_deepspeed_config(ctx.config)
        policy = DeepSpeedStepPolicy()
        new_models = dict(models)
        new_optimizers = dict(optimizers)
        new_schedulers = dict(schedulers)
        optimizer_records: dict[str, dict[str, Any]] = {}

        for opt_key, optimizer in optimizers.items():
            model_key = "discriminator" if opt_key == "opt_d" else "generator"
            model = models.get(model_key)
            if model is None or optimizer is None:
                continue
            client, client_scheduler, record = prepare_client_optimizer(
                optimizer,
                schedulers.get(opt_key),
                config=ctx.config,
                zero_stage=ds_cfg.zero_stage,
                offload_optimizer=ds_cfg.offload_optimizer,
                zenflow=ds_cfg.zenflow.enabled,
            )
            engine, ds_optimizer, ds_scheduler = initialize_deepspeed_engine(
                model=model,
                optimizer=client,
                lr_scheduler=client_scheduler,
                config=config,
            )
            record["engine"] = type(ds_optimizer).__name__
            optimizer_records[opt_key] = record
            new_models[model_key] = engine
            new_optimizers[opt_key] = ds_optimizer
            if ds_scheduler is not None:
                new_schedulers[opt_key] = ds_scheduler
            policy.register(ds_optimizer, engine)

        provenance = self.provenance(ctx)
        # Which optimizer class ran: the client handed to DeepSpeed and the wrapper it returned.
        provenance["optimizers"] = optimizer_records
        return ParallelizationResult(
            models=new_models,
            optimizers=new_optimizers,
            schedulers=new_schedulers,
            step_policy=policy,
            provenance=provenance,
        )

    def prepare_data_loaders(self, loaders, ctx):
        return _wrap_with_distributed_samplers(loaders)

    def checkpoint_adapter(self, ctx):
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            DeepSpeedCheckpointAdapter,
        )

        return DeepSpeedCheckpointAdapter()

    def provenance(self, ctx):
        from spectramr.infrastructure.distributed.deepspeed_backend import (
            build_deepspeed_config,
        )

        record = super().provenance(ctx)
        # The rendered dict, not the schema block: what DeepSpeed was actually
        # given is the thing worth being able to reconstruct.
        record["deepspeed"] = build_deepspeed_config(ctx.config)
        return record


for _plugin in (
    NoParallelStrategy(),
    DataParallelStrategy(),
    DDPStrategy(),
    FSDPStrategy(),
    DeepSpeedStrategy(),
):
    register_parallel_strategy(_plugin)
