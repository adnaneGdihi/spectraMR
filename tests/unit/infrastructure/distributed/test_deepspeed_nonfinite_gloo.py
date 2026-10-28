"""The fp32 DeepSpeed gradient check against a real two-rank gloo group.

Each rank holds its own ZeRO partition, so a NaN on one rank must make every rank
raise at the same optimizer step. A rank that raised alone would leave its peer
blocked in the next collective.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


class _PartitionEngine:
    """A ZeRO stage-1 engine whose only gradient is this rank's partition."""

    def __init__(self, partition: torch.Tensor) -> None:
        self.calls: list[str] = []
        self.global_steps = 3
        self.optimizer = SimpleNamespace(averaged_gradients={0: [partition]})
        self.module = SimpleNamespace(parameters=lambda: [])

    def backward(self, loss):
        self.calls.append("backward")

    def step(self):
        self.calls.append("step")

    def is_gradient_accumulation_boundary(self):
        return True

    def zero_optimization_stage(self):
        return 1

    def fp16_enabled(self):
        return False

    def bfloat16_enabled(self):
        return False


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
        DeepSpeedStepPolicy,
    )

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        bad = scenario == "one_rank_nan" and rank == 1
        partition = torch.tensor([0.5, float("nan") if bad else -0.5])
        engine = _PartitionEngine(partition)
        policy = DeepSpeedStepPolicy()
        optimizer = object()
        policy.register(optimizer, engine)
        if scenario == "all_finite":
            policy.backward_and_step(torch.tensor(1.0), optimizer)
            assert engine.calls == ["backward", "step"], (rank, engine.calls)
            return
        with pytest.raises(RuntimeError, match="Non-finite gradient at optimizer step 3"):
            policy.backward_and_step(torch.tensor(1.0), optimizer)
        assert engine.calls == ["backward"], (rank, engine.calls)
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_one_ranks_nan_partition_makes_every_rank_raise_before_step() -> None:
    _spawn("one_rank_nan")


def test_finite_partitions_step_on_every_rank() -> None:
    _spawn("all_finite")
