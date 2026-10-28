"""One owner for the per-step non-finite loss check.

Reading a loss back to the host to test it drains the CUDA queue, so a check
on every step costs a sync on every step. The monitor keeps the check on the
device: ``record`` queues a flag, ``end_step`` reduces the step's flags across
ranks and copies them into pinned memory behind a CUDA event, and the verdict
is read one step later (lag 1). ``end_step`` for step k waits on step k-1's
event, so the host never drains the queue but runs at most one step ahead.
``resolve`` drains the newest step on demand; the training loop calls it before
anything persists or evaluates the weights, so a non-finite step is never saved.

``guard_nonfinite_loss`` is the pre-backward rule both ``IStepPolicy.guard_loss``
and ``StepExecutor``'s fallback apply: record into a monitor when one is
attached, otherwise the synchronous raise for callers outside the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

#: Host slots preallocated per ring entry; a step recording more grows its slot.
_SLOT_CAPACITY = 16


@dataclass(frozen=True)
class NonFiniteVerdict:
    """A step whose recorded losses were not all finite, on some rank."""

    iteration: int
    names: tuple[str, ...]  # which losses recorded on THIS rank were non-finite
    on_peer_rank: bool  # True when only a peer saw it, so ``names`` is empty


@dataclass(frozen=True)
class _Pending:
    iteration: int
    names: tuple[str, ...]
    host: torch.Tensor | None  # [per-name local flags..., agreed flag]; None = clean
    event: Any


def process_group_spans_ranks() -> bool:
    """Whether a process group with more than one rank is active."""
    dist = torch.distributed
    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)


class NonFiniteMonitor:
    """Per-step non-finite flags, agreed across ranks and read one step late.

    Under a process group, ``end_step`` all-reduces a ONE-element flag on every
    rank every step -- including a step that recorded nothing -- so the
    collective count never depends on how many losses a rank recorded, and
    every rank receives the same verdict at the same call. ``resolve`` enters no
    collective, but it must be called at rank-uniform points so that every rank
    breaks out of the loop together.
    """

    def __init__(self, device: torch.device | str, *, distributed: bool) -> None:
        self.device = torch.device(device)
        self.distributed = distributed
        self._cuda = self.device.type == "cuda"
        self._flags: list[torch.Tensor] = []
        self._names: list[str] = []
        self._pending: list[_Pending] = []
        self._turn = 0
        self._slots: list[torch.Tensor] = []
        self._events: list[Any] = []
        if self._cuda:
            # Allocated here, not per step: a pinned allocation is not free.
            self._slots = [
                torch.empty(_SLOT_CAPACITY, dtype=torch.uint8, pin_memory=True) for _ in range(2)
            ]
            self._events = [torch.cuda.Event(), torch.cuda.Event()]

    def record(self, loss: torch.Tensor, *, name: str) -> None:
        """Queue ``~isfinite(loss).all()`` for the current step. Never syncs."""
        flag = (~torch.isfinite(loss.detach())).any()
        self._flags.append(flag.to(self.device, non_blocking=True))
        self._names.append(name)

    def end_step(self, iteration: int) -> NonFiniteVerdict | None:
        """Close step ``iteration`` and return the PREVIOUS step's verdict (lag 1)."""
        entry = self._close(iteration)
        previous, self._pending = self._pending, [entry]
        return self._first_bad(previous)

    def resolve(self) -> NonFiniteVerdict | None:
        """Drain every closed step; call before saves, validation, yield and exit."""
        pending, self._pending = self._pending, []
        return self._first_bad(pending)

    def _close(self, iteration: int) -> _Pending:
        flags, names = self._flags, tuple(self._names)
        self._flags, self._names = [], []
        if not flags and not self.distributed:
            return _Pending(iteration, (), None, None)
        if flags:
            local = torch.stack(flags).to(torch.uint8)
            agreed = local.amax().reshape(1)
        else:
            local = torch.zeros(0, dtype=torch.uint8, device=self.device)
            agreed = torch.zeros(1, dtype=torch.uint8, device=self.device)
        if self.distributed:
            torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MAX)
        packed = torch.cat([local, agreed])
        if not self._cuda:
            return _Pending(iteration, names, packed, None)
        # The slot of the other parity holds the previous step, read below; this
        # one was consumed by the end_step or resolve that preceded it.
        slot = self._slots[self._turn]
        if slot.numel() < packed.numel():
            slot = torch.empty(packed.numel(), dtype=torch.uint8, pin_memory=True)
            self._slots[self._turn] = slot
        host = slot[: packed.numel()]
        host.copy_(packed, non_blocking=True)
        event = self._events[self._turn]
        event.record(torch.cuda.current_stream(self.device))
        self._turn ^= 1
        return _Pending(iteration, names, host, event)

    @staticmethod
    def _first_bad(entries: list[_Pending]) -> NonFiniteVerdict | None:
        for entry in entries:
            if entry.host is None:
                continue
            if entry.event is not None:
                entry.event.synchronize()
            *local, agreed = entry.host.tolist()
            if not agreed:
                continue
            names = tuple(n for n, bad in zip(entry.names, local, strict=True) if bad)
            return NonFiniteVerdict(entry.iteration, names, on_peer_rank=not names)
        return None


def guard_nonfinite_loss(
    loss: torch.Tensor,
    *,
    name: str,
    global_step: int,
    scaler: Any | None,
    monitor: NonFiniteMonitor | None,
) -> None:
    """The default pre-backward rule for a non-finite loss.

    A GradScaler already skips inf/nan steps, so it is deferred to. Otherwise a
    non-finite loss gives non-finite gradients that ``clip_grad_norm_`` does not
    clamp. With a monitor attached (the training loop) the flag is recorded and
    the loop stops on the verdict, never saving the step; without one the
    verdict is agreed across ranks and raised here, which costs one device sync.
    """
    if scaler is not None:
        return
    if monitor is not None:
        monitor.record(loss, name=name)
        return
    non_finite = (~torch.isfinite(loss)).any()
    # A rank that raises alone leaves its peers blocked in the next collective,
    # so under a process group the local and the agreed flag are read together.
    agreed_elsewhere = False
    if process_group_spans_ranks():
        local = non_finite.to(torch.float32).reshape(1)
        agreed = local.clone()
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MAX)
        local_bad, agreed_bad = torch.cat([local, agreed]).tolist()
        agreed_elsewhere = bool(agreed_bad) and not bool(local_bad)
        bad = bool(agreed_bad)
    else:
        bad = bool(non_finite)
    if bad:
        where = "on a peer rank" if agreed_elsewhere else "on this rank"
        raise RuntimeError(
            f"Non-finite loss for config {name!r} at step {global_step} "
            f"({where}) before backward; refusing to poison weights (check "
            "coil maps / fidelity terms)."
        )
