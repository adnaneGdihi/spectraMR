"""Control flow that decides whether to enter a collective must be the same on every rank.

A divergence guard that looked only at its own rank's loss sent one rank out of the
loop while its peers waited in a collective it never joined. The loop's guard is now
``NonFiniteMonitor``, whose per-step all-reduce every rank enters the same number of
times. These run against a
real two-rank gloo group (CPU is enough), because a mocked reduction cannot show
that the ranks agree.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


def _run_steps(monitor, rank: int, bad_at: dict[int, set[int]], steps: int) -> list:
    """Drive ``steps`` monitored steps; ``bad_at[step]`` names the ranks whose loss is NaN."""
    verdicts = []
    for step in range(1, steps + 1):
        bad = rank in bad_at.get(step, set())
        monitor.record(torch.tensor(float("nan") if bad else 0.5), name="g_total_loss")
        verdict = monitor.end_step(step)
        if verdict is not None:
            verdicts.append((step, verdict.iteration))
    pending = monitor.resolve()
    if pending is not None:
        verdicts.append(("resolve", pending.iteration))
    return verdicts


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.training.nonfinite_monitor import NonFiniteMonitor

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        monitor = NonFiniteMonitor(torch.device("cpu"), distributed=True)
        if scenario == "one_rank_diverges":
            # Only rank 1 saw the NaN; both ranks learn of it at the same call.
            assert _run_steps(monitor, rank, {2: {1}}, 3) == [(3, 2)]
        elif scenario == "none_diverge":
            assert _run_steps(monitor, rank, {}, 3) == []
        elif scenario == "every_rank_diverges":
            assert _run_steps(monitor, rank, {2: {0, 1}}, 3) == [(3, 2)]
        elif scenario == "repeated":
            # A collective per step: the ranks must stay in step across many calls.
            assert _run_steps(monitor, rank, {7: {0}}, 20) == [(8, 7)]
        elif scenario == "last_step":
            # A NaN in the final step is only seen by the loop-exit resolve.
            assert _run_steps(monitor, rank, {5: {1}}, 5) == [("resolve", 5)]
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_a_divergence_on_one_rank_is_seen_by_every_rank() -> None:
    _spawn("one_rank_diverges")


def test_no_rank_diverging_is_agreed() -> None:
    _spawn("none_diverge")


def test_every_rank_diverging_is_agreed() -> None:
    _spawn("every_rank_diverges")


def test_the_flag_stays_in_step_across_many_iterations() -> None:
    _spawn("repeated")


def test_a_nan_in_the_last_step_is_seen_by_every_rank_at_exit() -> None:
    _spawn("last_step")


def test_single_process_needs_no_group() -> None:
    from spectramr.infrastructure.training.nonfinite_monitor import NonFiniteMonitor

    monitor = NonFiniteMonitor(torch.device("cpu"), distributed=False)
    assert _run_steps(monitor, 0, {2: {0}}, 3) == [(3, 2)]
