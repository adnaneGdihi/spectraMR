"""``guard_loss`` against a real two-rank gloo group.

A rank that stopped alone on a non-finite loss left its peer blocked in the next
gradient collective until the process-group timeout. CPU gloo is enough to show
that both ranks reach the same decision: the synchronous raise of a guard with no
monitor attached, and the same ``NonFiniteMonitor`` verdict at the same
``end_step`` when one is.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


def _monitored_steps(rank: int, policy, *, uneven: bool) -> None:
    """Three steps, NaN at step 2 on rank 1 only; every rank must agree at step 3."""
    from spectramr.infrastructure.training.nonfinite_monitor import (
        NonFiniteMonitor,
        NonFiniteVerdict,
    )

    monitor = NonFiniteMonitor(torch.device("cpu"), distributed=True)
    verdicts = []
    for step in (1, 2, 3):
        bad = rank == 1 and step == 2
        loss = torch.tensor(float("nan") if bad else 0.5)
        policy.guard_loss(loss, name="gen", global_step=step, scaler=None, monitor=monitor)
        # Ranks may record different numbers of losses (a data-dependent branch);
        # the collective must not depend on that count.
        if uneven and rank == 0:
            monitor.record(torch.tensor(0.25), name="extra")
        verdicts.append(monitor.end_step(step))
    assert verdicts[:2] == [None, None], verdicts
    expected = NonFiniteVerdict(
        iteration=2, names=("gen",) if rank == 1 else (), on_peer_rank=rank == 0
    )
    assert verdicts[2] == expected, (rank, verdicts[2])
    assert monitor.resolve() is None


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.training.optimizers import AMPPolicy

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        policy = AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
        if scenario in ("monitored", "monitored_uneven"):
            _monitored_steps(rank, policy, uneven=scenario == "monitored_uneven")
            return
        bad = scenario == "one_rank_nan" and rank == 1
        loss = torch.tensor(float("nan") if bad else 0.5)
        if scenario == "all_finite":
            policy.guard_loss(loss, name="gen", global_step=3, scaler=None)
            return
        try:
            policy.guard_loss(loss, name="gen", global_step=3, scaler=None)
        except RuntimeError as exc:
            expected = "on this rank" if bad else "on a peer rank"
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError(f"rank {rank} did not raise on a peer's non-finite loss")
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_one_rank_nan_makes_every_rank_raise() -> None:
    _spawn("one_rank_nan")


def test_finite_everywhere_passes_on_every_rank() -> None:
    _spawn("all_finite")


def test_a_monitored_nan_on_one_rank_gives_every_rank_the_same_verdict() -> None:
    _spawn("monitored")


def test_ranks_recording_different_counts_stay_in_lockstep() -> None:
    _spawn("monitored_uneven")
