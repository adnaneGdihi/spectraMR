"""The step policy of a pipelined generator: the schedule owns the backward.

The strategy's closure runs unchanged on every stage and yields the same loss on
each (the output is broadcast). Only the last stage holds that loss's graph, so this
policy back-propagates it there, moves the gradients back through the stages, then
averages them over the data-parallel replicas when no FSDP2 reduce-scatter did it,
clips by the norm over every stage (``clip_grad_norm_global_``) and steps.

Under fp16 the last stage back-propagates the scaled loss, so every stage receives
scaled gradients. Each stage unscales its own before the clip; the scaler's non-finite
flag is all-reduced over the world (``needs_cross_rank_inf_flag``), so an overflow on
any stage skips the step and halves the scale on every rank alike.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from spectramr.infrastructure.distributed.pipeline_parallel.module import (
    PipelinedGenerator,
    pipelined_of,
)
from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy

__all__ = ["PipelineStepPolicy", "average_replica_gradients"]


def average_replica_gradients(pipelined: PipelinedGenerator) -> None:
    """Mean of each gradient over the replicated data-parallel group, one flat reduce per dtype.

    A flat buffer has one dtype: flattening complex and real gradients together would
    promote the real ones to complex and drop the imaginary parts on the way back.
    """
    group = pipelined.replica_group
    if group is None:
        return
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for p in pipelined.stage.parameters():
        if p.grad is not None:
            by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    size = dist.get_world_size(group)
    for grads in by_dtype.values():
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, group=group)
        flat.div_(size)
        for grad, synced in zip(
            grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=True
        ):
            grad.copy_(synced)


class PipelineStepPolicy(AMPPolicy):
    """Backward through the pipeline, then the ordinary (unscale-)clip-and-step."""

    @staticmethod
    def _holds_the_loss_graph(pipelined: PipelinedGenerator) -> bool:
        # Elsewhere the loss was computed on a broadcast copy of the output.
        return pipelined.is_last

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        model_name: str,
        epoch: int,
        model_type: str,
        scaler: Any | None = None,
        perform_step: bool = True,
        gradient_clipping_fn: Callable[[torch.nn.Module], float] | None = None,
        skip_scaler_update: bool = False,
    ) -> None:
        pipelined = pipelined_of(model)
        if pipelined is None:
            raise RuntimeError(
                f"step config {model_name!r} trains {type(model).__name__}, not the pipelined "
                "generator; a pipeline run has one optimizer over its stage."
            )
        if optimizer is None or not perform_step:
            raise RuntimeError(
                "the pipeline step policy runs one optimizer, no accumulation; the plugin "
                "refuses the rest at build, so reaching here is a wiring defect."
            )
        if scaler is not None and not getattr(scaler, "sharded", False):
            raise RuntimeError(
                "a pipelined fp16 run needs the scaler's world-wide non-finite flag "
                "(ComplexGradScaler(sharded=True)); each stage holds different gradients."
            )
        # Every stage scales (GradScaler builds its scale lazily there); the loss graph
        # exists on the last stage only.
        scaled = scaler.scale(loss) if scaler is not None else loss
        if self._holds_the_loss_graph(pipelined):
            scaled.backward()
        pipelined.pipeline_backward()
        average_replica_gradients(pipelined)
        if scaler is not None:
            self._unscale_clip_and_step(
                optimizer, model, model_name, scaler, gradient_clipping_fn, skip_scaler_update
            )
            return
        self._ensure_gradients_arrived(optimizer, model_name)
        if gradient_clipping_fn:
            gradient_clipping_fn(model)
        else:
            self._apply_gradient_clipping(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
