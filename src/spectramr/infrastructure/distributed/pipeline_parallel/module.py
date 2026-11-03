"""This rank's pipeline stage, behind the generator's own forward signature.

Every rank of a pipeline group calls the generator with the same batch (they share a
data-parallel index, so the sampler hands them the same samples). The first stage
enters and splits the payload into microbatches; each stage runs its cascades on one
microbatch, posts the result to the next stage at once and moves on to the next
microbatch, so stage k+1 works on microbatch i while stage k computes i+1 (GPipe:
all forwards, then all backwards). The last stage concatenates, exits, and broadcasts
the output to the other stages, so every rank returns the full output: the strategy
computes the same loss and the same validation metrics everywhere, and the writer
(global rank 0, the first stage) logs, snapshots and checkpoints as it would without
a pipeline. The last stage back-propagates the full-batch loss once; upstream stages
then back-propagate microbatch by microbatch as each gradient arrives.

Only the last stage's output carries the autograd graph. Elsewhere the returned
tensor is a detached leaf; :meth:`PipelinedGenerator.pipeline_backward`, called by
the pipeline step policy, moves gradients back stage by stage.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from spectramr.infrastructure.distributed.pipeline_parallel import transport

__all__ = ["PP_GROUP_ATTR", "PipelinedGenerator", "pipelined_of"]

logger = logging.getLogger(__name__)

#: Attribute naming the pipeline process group on the root; read by ``grad_norm``.
PP_GROUP_ATTR = "_spectramr_pp_group"


@dataclass
class _Pending:
    """What one training forward left for its backward."""

    sent: list[list[torch.Tensor]] = field(default_factory=list)
    received: list[list[torch.Tensor]] = field(default_factory=list)
    roots: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


class PipelinedGenerator(nn.Module):
    """One stage of a split generator; ``forward`` keeps the unsplit model's signature."""

    def __init__(
        self,
        stage: nn.Module,
        *,
        group: Any,
        meta_group: Any,
        ranks: list[int],
        index: int,
        microbatches: int,
        signature: inspect.Signature,
    ):
        super().__init__()
        self.stage = stage
        self.pp_ranks = tuple(int(r) for r in ranks)
        self.pp_index = index
        self.microbatches = microbatches
        setattr(self, PP_GROUP_ATTR, group)
        #: Host-side (gloo) group for headers and host entries; reading them never syncs.
        self.meta_group = meta_group
        #: Set when gradients are averaged by hand over a replicated data-parallel group.
        self.replica_group: Any = None
        self.replica_src: int = 0
        self.record: dict[str, Any] = {}
        self._pending: _Pending | None = None
        self._peers_checked = False
        self._short_batches: set[int] = set()

        # The strategy picks forward kwargs by signature; it must see the model's.
        def forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            return self._run(*args, **kwargs)

        forward.__signature__ = signature  # type: ignore[attr-defined]
        self.forward = forward  # type: ignore[method-assign]

    @property
    def is_first(self) -> bool:
        return self.pp_index == 0

    @property
    def is_last(self) -> bool:
        return self.pp_index == len(self.pp_ranks) - 1

    @property
    def group(self) -> Any:
        return getattr(self, PP_GROUP_ATTR)

    def _device(self) -> torch.device:
        return next(self.stage.parameters()).device

    def _run(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        grad = torch.is_grad_enabled()
        if grad and self._pending is not None:
            raise RuntimeError(
                "a pipelined generator ran a second training forward before the first one's "
                "backward; only the pipeline step policy may back-propagate it."
            )
        if grad and self.training:
            self._check_peers_agree(args[0] if args else next(iter(kwargs.values())))
            self._sync_replica_buffers()
        pending = _Pending()
        if self.is_first:
            microbatches, batched, count = self._enter(args, kwargs, grad, pending)
        else:
            microbatches, batched, count = self._receive(grad, pending)
        outs = self._cascade_and_post(microbatches, batched, count)
        if self.is_last:
            result = self.stage("exit", *_concat(outs, batched))
        else:
            pending.sent = [[t for t in out if _carries_grad(t)] for out in outs]
            result = None
        if grad:
            self._pending = pending
        return self._broadcast_output(result, grad)

    def _enter(self, args, kwargs, grad, pending):
        payload = self.stage("enter", *args, **kwargs)
        batch = payload[0].shape[0]
        count = self._count(batch)
        batched = self.stage.batched_entries(payload, batch)
        if grad:  # cut the shared entry graph so each microbatch backward is independent
            leafed = []
            for entry in payload:
                if _carries_grad(entry):
                    leaf = entry.detach().requires_grad_(True)
                    pending.roots.append((entry, leaf))
                    entry = leaf
                leafed.append(entry)
            payload = tuple(leafed)
        return _split(payload, batched, count), batched, count

    def _receive(self, grad, pending):
        """The upstream microbatches, each received only when the loop asks for it."""
        src, device = self.pp_ranks[self.pp_index - 1], self._device()
        header = transport.new_header()
        dist.recv(header, src=src, group=self.meta_group)
        specs, count = transport.decode_header(header)

        def microbatches():
            for _ in range(count):
                mb, leaves = transport.recv_payload(
                    specs, src, self.group, self.meta_group, device, grad
                )
                pending.received.append(leaves)
                yield mb

        batched = tuple(spec is not None and spec.batched for spec in specs)
        return microbatches(), batched, count

    def _cascade_and_post(self, microbatches, batched, count) -> list[tuple]:
        """Run each microbatch's cascades and post the result downstream before the next."""
        outs: list[tuple] = []
        posted: list[transport.Posted] = []
        dst = None if self.is_last else self.pp_ranks[self.pp_index + 1]
        device = self._device()
        for mb in microbatches:
            out = self.stage("cascade", *mb)
            outs.append(out)
            if dst is None:
                continue
            if not posted:  # every microbatch shares the first one's header
                header = transport.encode_header(out, batched, count, device)
                posted.append(transport.Posted(header, dst, self.meta_group))
            posted += transport.send_payload(out, dst, self.group, self.meta_group, device)
        for work in posted:
            work.wait()
        return outs

    def _broadcast_output(self, result: torch.Tensor | None, grad: bool) -> torch.Tensor:
        src, device = self.pp_ranks[-1], self._device()
        if self.is_last:
            header = transport.encode_header((result,), (True,), 1, device)
        else:
            header = transport.new_header()
        dist.broadcast(header, src=src, group=self.meta_group)
        if self.is_last:
            dist.broadcast(transport.wire(result), src=src, group=self.group)
            return result
        spec = transport.decode_header(header)[0][0]
        buffer = transport.empty_wire(spec, device)
        dist.broadcast(buffer, src=src, group=self.group)
        # A leaf, so the strategy's loss passes the backward-ready guard on every stage.
        buffer.requires_grad_(grad and spec.requires_grad)
        return transport.from_wire(buffer, spec)

    def pipeline_backward(self) -> None:
        """Move gradients from the last stage to the first. The last stage's loss is
        already back-propagated (its leaves hold the gradients of the received payloads)."""
        pending, self._pending = self._pending, None
        if pending is None:
            raise RuntimeError("pipeline_backward without a pending training forward.")
        works: list[Any] = []
        prev = self.pp_ranks[self.pp_index - 1] if not self.is_first else None
        if self.is_last:
            for leaves in pending.received:
                works += transport.send_grads(leaves, prev, self.group)
        else:
            nxt = self.pp_ranks[self.pp_index + 1]
            for i, sent in enumerate(pending.sent):
                grads = transport.recv_grads(sent, nxt, self.group)
                # Sync on the last microbatch: the only backward reaching every cascade
                # unit. The root backward below syncs again for what the entry adds.
                self._backward(sent, grads, i == len(pending.sent) - 1)
                if prev is not None:
                    works += transport.send_grads(pending.received[i], prev, self.group)
        roots = [(t, leaf.grad) for t, leaf in pending.roots if leaf.grad is not None]
        if roots:
            self._backward([t for t, _ in roots], [g for _, g in roots], True)
        for work in works:
            work.wait()

    def _backward(self, tensors: list[torch.Tensor], grads: list[torch.Tensor], final: bool):
        sync = getattr(self.stage, "set_requires_gradient_sync", None)
        if callable(sync):  # FSDP2: reduce-scatter once, on this rank's last backward
            sync(final)
        if tensors:
            torch.autograd.backward(tensors, grads)

    def _count(self, batch: int) -> int:
        """Equal microbatches. A batch the count does not divide (a last short batch) runs
        as fewer: the loss is the full batch's either way, only the bubble grows."""
        count = math.gcd(batch, self.microbatches)
        if count != self.microbatches and batch not in self._short_batches:
            self._short_batches.add(batch)
            logger.warning(
                "[Pipeline] batch %d does not divide into %d microbatches; running %d.",
                batch,
                self.microbatches,
                count,
            )
        return count

    def _check_peers_agree(self, x: torch.Tensor) -> None:
        """Once: every stage must hold the same input, or stage 0's forward and the last
        stage's target belong to different samples."""
        if self._peers_checked:
            return
        flat = torch.view_as_real(x.detach()) if x.is_complex() else x.detach()
        digest = torch.stack([flat.double().sum(), flat.double().square().sum()])
        peers = [torch.empty_like(digest) for _ in self.pp_ranks]
        dist.all_gather(peers, digest, group=self.group)
        if any(not torch.equal(peers[0], p) for p in peers):
            raise RuntimeError(
                "pipeline stages received different batches; every rank of a pipeline group "
                "must draw the same samples (data-parallel rank, not global rank)."
            )
        self._peers_checked = True

    def _sync_replica_buffers(self) -> None:
        if self.replica_group is None:
            return
        for buf in self.stage.buffers():  # as DDP's broadcast_buffers does each forward
            dist.broadcast(buf, src=self.replica_src, group=self.replica_group)


def _carries_grad(entry: Any) -> bool:
    return isinstance(entry, torch.Tensor) and entry.requires_grad


def _split(payload: tuple[Any, ...], batched: tuple[bool, ...], count: int) -> list[tuple]:
    parts = [
        torch.tensor_split(entry, count) if flag else [entry] * count
        for entry, flag in zip(payload, batched, strict=True)
    ]
    return [tuple(part[i] for part in parts) for i in range(count)]


def _concat(outs: list[tuple], batched: tuple[bool, ...]) -> tuple[Any, ...]:
    return tuple(
        torch.cat([out[i] for out in outs]) if flag and outs[0][i] is not None else outs[0][i]
        for i, flag in enumerate(batched)
    )


def pipelined_of(model: Any) -> PipelinedGenerator | None:
    """The :class:`PipelinedGenerator` under ``model``'s wrappers, or ``None``."""
    from spectramr.core.module_utils import unwrap_model

    root = unwrap_model(model)
    return root if isinstance(root, PipelinedGenerator) else None
