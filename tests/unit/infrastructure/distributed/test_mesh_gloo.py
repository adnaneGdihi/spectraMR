"""The run's one mesh on four CPU ranks (gloo): groups, batches, reduces and refusals.

Four ranks with ``tensor=2`` give two data-parallel replicas. Row-major order puts the
two tensor peers of a replica at ranks (0,1) and (2,3). The production helpers are
exercised directly: ``DDPStrategy.prepare_data_loaders``,
``pipelines.train._all_reduce_val_metrics`` and
``InceptionScoreMetric._gather_probs_across_ranks``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

WORLD = 4
N = 8  # divisible by the data-parallel size, so the sampler pads nothing


def _parallel(**axes):
    mesh = {"tensor": 1, "context": 1, "spatial": 1, "pipeline": 1, **axes}
    mesh["non_dp_size"] = mesh["tensor"] * mesh["context"] * mesh["spatial"] * mesh["pipeline"]
    return SimpleNamespace(strategy="ddp", mesh=SimpleNamespace(**mesh))


def _dataset():
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.arange(N, dtype=torch.float64))


def _production_indices(rank: int, mesh) -> list[int]:
    """Indices one rank draws through the DDP strategy's real sampler path."""
    from torch.utils.data import DataLoader

    from spectramr.infrastructure.distributed.strategies import DDPStrategy
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    loader = DataLoader(_dataset(), batch_size=2, shuffle=False)
    ctx = ParallelContext(config=None, device=torch.device("cpu"), parallel=None, mesh=mesh)
    wrapped = DDPStrategy().prepare_data_loaders({"val": loader}, ctx)["val"]
    return [int(v) for batch in wrapped for v in batch[0].tolist()]


def _gather(obj) -> list:
    import torch.distributed as dist

    out: list = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        globals()[f"_scenario_{scenario}"](rank)
    finally:
        dist.destroy_process_group()


def _scenario_tensor_peers(rank: int) -> None:
    from torch.utils.data import DistributedSampler

    from spectramr.core.parallel_groups import (
        dp_group,
        dp_rank,
        dp_src_rank,
        dp_world_size,
        is_dp_writer,
    )
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh

    pm = build_parallel_mesh(_parallel(tensor=2), device_type="cpu", world_size=WORLD)
    assert dp_world_size() == 2
    assert dp_rank() == rank // 2
    # The broadcast root of a replica is its tensor-coordinate-0 peer: rank 0 or rank 1.
    assert dp_src_rank() == rank % 2
    assert is_dp_writer() is (rank % 2 == 0)
    assert dp_group() is not None

    # 1. Tensor peers draw identical batches; the two replicas draw disjoint halves.
    seen = _gather(_production_indices(rank, pm))
    assert seen[0] == seen[1] and seen[2] == seen[3]
    assert seen[0] != seen[2]
    assert sorted(seen[0] + seen[2]) == list(range(N))

    # Planted wrong route: a sampler keyed on the GLOBAL rank splits tensor peers apart.
    wrong = DistributedSampler(_dataset(), num_replicas=WORLD, rank=rank, shuffle=False)
    wrong_seen = _gather([int(i) for i in wrong])
    assert wrong_seen[0] != wrong_seen[1], "control: global-rank sampling must split tp peers"

    # 2. The validation reduce counts each sample once, not once per tensor peer.
    from spectramr.pipelines.train import _all_reduce_val_metrics

    local = _production_indices(rank, pm)
    local_sum = float(sum(local))
    accum, _, _, weight = _all_reduce_val_metrics(
        {"mse": local_sum},
        {"mse": float(len(local))},
        {"mse": float(len(local))},
        len(local),
        torch.device("cpu"),
    )
    assert weight == N
    assert accum["mse"] == pytest.approx(float(sum(range(N))))

    # 3. Image-quality features are gathered across replicas, one piece per replica.
    from spectramr.core.metrics.evaluation_metrics import InceptionScoreMetric

    pieces = InceptionScoreMetric._gather_probs_across_ranks(torch.tensor([float(rank)]))
    assert len(pieces) == 2

    # 4. Non-finite agreement reads the global group, not the replica group.
    from spectramr.infrastructure.training.nonfinite_monitor import process_group_spans_ranks

    assert process_group_spans_ranks() is True

    # 5. An indivisible mesh raises on every rank, before any collective.
    with pytest.raises(ValueError, match="does not divide"):
        build_parallel_mesh(_parallel(tensor=3), device_type="cpu", world_size=WORLD)


def _scenario_trivial(rank: int) -> None:
    from torch.utils.data import DistributedSampler

    from spectramr.core.parallel_groups import (
        dp_group,
        dp_rank,
        dp_src_rank,
        dp_world_size,
        is_dp_writer,
    )
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh

    pm = build_parallel_mesh(_parallel(), device_type="cpu", world_size=WORLD)
    # A trivial mesh reproduces the global behaviour exactly.
    assert dp_world_size() == WORLD
    assert dp_rank() == rank
    assert dp_group() is None
    assert dp_src_rank() == 0
    assert is_dp_writer() is True

    assert _production_indices(rank, pm) == [
        int(i) for i in DistributedSampler(_dataset(), num_replicas=WORLD, rank=rank, shuffle=False)
    ]

    from spectramr.pipelines.train import _all_reduce_val_metrics

    local = _production_indices(rank, pm)
    _, _, _, weight = _all_reduce_val_metrics(
        {"mse": float(sum(local))},
        {"mse": float(len(local))},
        {"mse": float(len(local))},
        len(local),
        torch.device("cpu"),
    )
    assert weight == N


@pytest.mark.parametrize("scenario", ["tensor_peers", "trivial"])
def test_mesh_on_four_gloo_ranks(scenario: str, tmp_path) -> None:
    import torch.multiprocessing as mp

    store = str(tmp_path / f"store_{scenario}")
    mp.spawn(_worker, args=(store, scenario), nprocs=WORLD, join=True)
