"""Inception Score must grade the whole validation set, not each rank's shard.

``InceptionScoreMetric`` keeps a plain list of class probabilities, so unlike
torchmetrics' FID and KID (which synchronise their own state inside ``compute``) it
reported the score of its own shard. Ranks then disagreed on the early-stopping and
best-checkpoint decision. The metric is built without its Inception weights (which
would need a download) because ``compute`` only reads the probability list.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


def _metric(probs: list[torch.Tensor]):
    from spectramr.core.metrics.evaluation_metrics import InceptionScoreMetric

    metric = InceptionScoreMetric.__new__(InceptionScoreMetric)
    torch.nn.Module.__init__(metric)
    metric.device = torch.device("cpu")
    metric.generated_features = probs
    return metric


def _probs(seed: int, n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, 10, generator=generator), dim=1)


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        shards = [_probs(1, 6), _probs(2, 4)]
        whole = _metric([torch.cat(shards)]).compute()
        if scenario == "matches_single_process":
            value = _metric([shards[rank]]).compute()
            assert value == pytest.approx(whole, rel=1e-5)
        elif scenario == "empty_rank_does_not_hang":
            own = [shards[0]] if rank == 0 else []
            value = _metric(own).compute()
            assert value == pytest.approx(_metric([shards[0]]).compute(), rel=1e-5)
        elif scenario == "all_empty":
            assert _metric([]).compute() == 0.0
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_each_rank_reports_the_score_of_the_whole_set() -> None:
    """Planted: before the gather each rank returned the IS of its own shard."""
    _spawn("matches_single_process")


def test_a_rank_with_no_samples_still_joins_the_gather() -> None:
    _spawn("empty_rank_does_not_hang")


def test_all_ranks_empty_is_zero_not_a_hang() -> None:
    _spawn("all_empty")


def test_single_process_is_unchanged() -> None:
    probs = _probs(3, 8)
    assert _metric([probs]).compute() == pytest.approx(_metric([probs[:4], probs[4:]]).compute())
    assert _metric([]).compute() == 0.0
