"""Stage A of a pipelined run: refuse what is unverified, then split the generator.

Called by the ddp and fsdp plugins when ``parallel.mesh.pipeline > 1``. The full
model is built on every rank first (the hook needs it to partition), then each rank
keeps its own stage and drops the rest.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch
import torch.distributed as dist

from spectramr.core.parallel_groups import dp_group, dp_src_rank, dp_world_size
from spectramr.infrastructure.distributed.pipeline_parallel.module import PipelinedGenerator

__all__ = [
    "has_pipeline_hook",
    "pipeline_provenance",
    "pipeline_size",
    "split_generator_for_pipeline",
]

logger = logging.getLogger(__name__)


def pipeline_size(ctx: Any) -> int:
    mesh = getattr(ctx, "mesh", None)
    return int(mesh.shape.get("pp", 1)) if mesh is not None else 1


def _refusals(ctx: Any) -> list[str]:
    from spectramr.infrastructure.training.builders.compile_apply import compile_is_enabled

    config, parallel = ctx.config, ctx.parallel
    refused = []
    if compile_is_enabled(config):
        refused.append("optimization.compile")
    if int(config.optimization.gradient.accumulation_steps) > 1:
        refused.append("optimization.gradient.accumulation_steps > 1")
    if getattr(getattr(config, "ema", None), "enabled", False):
        refused.append("ema.enabled")
    if parallel.strategy == "fsdp" and parallel.fsdp.cpu_offload:
        refused.append("fsdp.cpu_offload")
    if getattr(parallel, "sync_batch_norm", False):
        refused.append("sync_batch_norm")
    # The split generator has no forward_with_intermediates, so the term would never run.
    if float(getattr(getattr(config, "losses", None), "lambda_deep_supervision", 0) or 0) > 0:
        refused.append("losses.lambda_deep_supervision > 0")
    # No DDP wrapper under a pipeline: its knobs would be stamped and never read.
    ddp_knobs = ("find_unused_parameters", "gradient_as_bucket_view", "static_graph")
    refused += [f"parallel.{k}" for k in ddp_knobs if getattr(parallel, k, False)]
    return refused


def has_pipeline_hook(model: Any) -> bool:
    """Whether a model (class or instance) can be split by ``pipeline_stages(n)``."""
    return callable(getattr(model, "pipeline_stages", None))


def _models_with_the_hook() -> list[str]:
    from spectramr.models.registry import list_models

    return sorted(
        name for name, entry in list_models().items() if has_pipeline_hook(entry.get("class"))
    )


def _meta_group(device_mesh: Any) -> Any:
    """This rank's host-side (gloo) copy of its pipeline group, for headers and host entries.

    ``new_subgroups_by_enumeration`` is collective: every rank builds every pipeline
    group's copy in the same order, then keeps its own.
    """
    pp_dim = list(device_mesh.mesh_dim_names).index("pp")
    mesh = device_mesh.mesh
    rows = mesh.movedim(pp_dim, -1).reshape(-1, mesh.shape[pp_dim]).tolist()
    group, _ = dist.new_subgroups_by_enumeration(rows, backend="gloo")
    return group


def split_generator_for_pipeline(models: dict[str, Any], ctx: Any) -> PipelinedGenerator:
    """Replace ``models["generator"]`` by this rank's :class:`PipelinedGenerator`."""
    size = pipeline_size(ctx)
    refused = _refusals(ctx)
    if refused:
        raise RuntimeError(
            f"parallel.mesh.pipeline={size} is verified with one optimizer step per batch, "
            f"without EMA, compile or CPU offload; refusing {', '.join(refused)}."
        )
    others = sorted(k for k, v in models.items() if v is not None and k != "generator")
    if others:
        raise RuntimeError(f"pipeline parallelism splits the generator alone; refusing {others}.")
    generator = models["generator"]
    if not has_pipeline_hook(generator):
        raise RuntimeError(
            f"{type(generator).__name__} has no pipeline_stages(n) hook, so it cannot be split "
            f"into {size} stages. Models with the hook: {_models_with_the_hook()}."
        )
    microbatches = int(ctx.parallel.mesh.pipeline_microbatches)
    if microbatches > 1 and any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in generator.modules()
    ):
        raise RuntimeError(
            f"{type(generator).__name__} has BatchNorm, whose statistics over a microbatch "
            "differ from the batch's; use parallel.mesh.pipeline_microbatches: 1."
        )
    sub = ctx.mesh.device_mesh["pp"]
    index = sub.get_local_rank()
    stages = generator.pipeline_stages(size)
    pipelined = PipelinedGenerator(
        stages[index],
        group=sub.get_group(),
        meta_group=_meta_group(ctx.mesh.device_mesh),
        ranks=sub.mesh.tolist(),
        index=index,
        microbatches=microbatches,
        signature=inspect.signature(generator.forward),
    )
    pipelined.record = {
        "size": size,
        "microbatches": microbatches,
        # Forward streams microbatches stage to stage; the last stage back-propagates the
        # full-batch loss once, upstream stages microbatch by microbatch. Not measured.
        "schedule": "gpipe_fill_drain",
        "last_stage_backward": "full_batch",
        "short_batch": "gcd(batch, microbatches) microbatches",
        "cascades_per_stage": [len(stage.cascades) for stage in stages],
        "grad_sync": "fsdp2" if ctx.parallel.strategy == "fsdp" else "allreduce",
    }
    if ctx.parallel.strategy != "fsdp" and dp_world_size() > 1:
        # No DDP wrapper (its reducer cannot span several microbatch backwards): start the
        # replicas equal, as DDP's constructor does, and let the step policy average.
        pipelined.replica_group, pipelined.replica_src = dp_group(), dp_src_rank()
        for tensor in [*pipelined.stage.parameters(), *pipelined.stage.buffers()]:
            dist.broadcast(tensor.data, src=pipelined.replica_src, group=pipelined.replica_group)
    models["generator"] = pipelined
    logger.info("[Parallel] pipeline stage %d/%d: %s", index, size, pipelined.record)
    return pipelined


def pipeline_provenance(models: dict[str, Any]) -> dict[str, Any]:
    from spectramr.infrastructure.distributed.pipeline_parallel.module import pipelined_of

    generator = models.get("generator")
    pipelined = pipelined_of(generator) if generator is not None else None
    return {"pipeline_parallel": dict(pipelined.record)} if pipelined is not None else {}
