"""A strategy-owned head under DDP stays identical across two ranks (#2629).

Each rank starts its head from a different seed and sees different data. Without the
wrap the head's gradients are never reduced, so the ranks drift apart; with the wrap
they must match after one step.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


def _step_once(rank: int, store: str, wrapped: bool, out_dir: str) -> None:
    import torch.distributed as dist
    import torch.nn as nn

    from spectramr.infrastructure.distributed.owned_modules import wrap_owned_module

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.manual_seed(100 + rank)
        head = nn.Linear(4, 1)
        parallel = SimpleNamespace(
            strategy="ddp",
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            static_graph=False,
        )
        forward = head
        if wrapped:
            forward = wrap_owned_module(
                head, owner="Probe", name="_head", parallel=parallel, device=torch.device("cpu")
            )
        opt = torch.optim.SGD(head.parameters(), lr=0.1)
        torch.manual_seed(7 + rank)
        x = torch.randn(8, 4) + rank
        loss = forward(x).pow(2).mean()
        loss.backward()
        opt.step()
        flat = torch.cat([p.detach().reshape(-1) for p in head.parameters()])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save([t.clone() for t in peers], os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(wrapped: bool) -> list[torch.Tensor]:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _step_once,
            args=(os.path.join(tmp, "store"), wrapped, tmp),
            nprocs=WORLD,
            join=True,
        )
        return torch.load(os.path.join(tmp, "r0.pt"))


def test_owned_head_is_identical_across_ranks_after_a_step() -> None:
    peers = _spawn(wrapped=True)
    assert torch.allclose(peers[0], peers[1]), (peers[0] - peers[1]).abs().max()


def test_control_without_the_wrap_diverges() -> None:
    """Shows the assertion above is not vacuous: unwrapped heads drift apart."""
    peers = _spawn(wrapped=False)
    assert not torch.allclose(peers[0], peers[1])
