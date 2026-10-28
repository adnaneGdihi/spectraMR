"""``ValidationMemoryTrace``: one line per cadence point on a CUDA pass, and a rank-wide peak.

The CUDA readings are replaced with fixed numbers, so the cadence and the collective
are exercised on CPU. The collective is the contract that matters: every rank must
enter it, whether or not its own device reports memory.
"""

from __future__ import annotations

import torch.distributed as dist

from spectramr.infrastructure.training.validation_memory import ValidationMemoryTrace


def _trace(lines: list[str], *, enabled: bool) -> ValidationMemoryTrace:
    trace = ValidationMemoryTrace("cpu", emit=lines.append, every=2)
    trace.enabled = enabled
    return trace


def test_nothing_is_logged_or_read_when_the_device_is_not_cuda() -> None:
    lines: list[str] = []
    trace = _trace(lines, enabled=False)
    trace.start()
    trace.after_batch(0)
    trace.end()
    assert lines == []


def test_start_every_n_batches_and_end_are_each_logged_once(monkeypatch) -> None:
    lines: list[str] = []
    trace = _trace(lines, enabled=True)
    monkeypatch.setattr(trace, "_read", lambda: (100.0, 250.0))
    trace.start()
    for index in range(4):
        trace.after_batch(index)
    trace.end()
    assert [line.split(":")[0] for line in lines] == [
        "[VAL MEM] start",
        "[VAL MEM] batch 2",
        "[VAL MEM] batch 4",
        "[VAL MEM] end",
    ]
    assert "max peak over ranks=250 MiB" in lines[-1]


def test_end_enters_the_rank_reduction_even_when_this_rank_has_no_cuda(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op: calls.append(op))
    trace = _trace([], enabled=False)
    trace.end()
    assert calls == [dist.ReduceOp.MAX]


def test_the_reduced_peak_is_the_largest_rank_peak(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def _fake_all_reduce(tensor, op):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(max(float(tensor.item()), 900.0))

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)
    trace = _trace(lines, enabled=True)
    monkeypatch.setattr(trace, "_read", lambda: (10.0, 100.0))
    trace.end()
    assert "max peak over ranks=900 MiB" in lines[-1]
