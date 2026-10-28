"""Allocator readings across one validation pass, for finding what keeps memory alive.

The allocator counters are host-side bookkeeping: reading them does not sync the
device, so they may be sampled per batch (non-negotiable 9). Peaks are the process
peak, not reset per pass, so the training loop's own peak statistics are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_MIB = 1024**2


class ValidationMemoryTrace:
    """Logs allocated and peak memory at the start, every ``every`` batches, and the end."""

    def __init__(
        self,
        device: torch.device | str | None,
        *,
        emit: Callable[[str], None],
        every: int = 25,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self.enabled = (
            self.device is not None and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self._emit = emit
        self._every = max(1, int(every))
        self.peak_mib = 0.0

    def _read(self) -> tuple[float, float]:
        return (
            torch.cuda.memory_allocated(self.device) / _MIB,
            torch.cuda.max_memory_allocated(self.device) / _MIB,
        )

    def start(self) -> None:
        if not self.enabled:
            return
        allocated, peak = self._read()
        self.peak_mib = peak
        self._emit(f"[VAL MEM] start: allocated={allocated:.0f} MiB peak={peak:.0f} MiB")

    def after_batch(self, batch_index: int) -> None:
        if not self.enabled:
            return
        allocated, peak = self._read()
        self.peak_mib = max(self.peak_mib, peak)
        if (batch_index + 1) % self._every == 0:
            self._emit(
                f"[VAL MEM] batch {batch_index + 1}: allocated={allocated:.0f} MiB "
                f"peak={peak:.0f} MiB"
            )

    def end(self) -> None:
        """Log the pass end and the largest peak over ranks.

        The all-reduce runs on every rank whenever ``torch.distributed`` is
        initialised, whatever this rank's ``enabled``, so no rank can skip it.
        """
        import torch.distributed as dist

        local_peak = self.peak_mib
        if self.enabled:
            allocated, peak = self._read()
            local_peak = max(self.peak_mib, peak)
        global_peak = local_peak
        if dist.is_available() and dist.is_initialized():
            reduced = torch.tensor([local_peak], dtype=torch.float64, device=self.device)
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
            global_peak = float(reduced.item())
        if self.enabled:
            self._emit(
                f"[VAL MEM] end: allocated={allocated:.0f} MiB peak={local_peak:.0f} MiB "
                f"max peak over ranks={global_peak:.0f} MiB"
            )


__all__ = ["ValidationMemoryTrace"]
