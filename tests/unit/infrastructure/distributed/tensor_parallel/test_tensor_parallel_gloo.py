"""Tensor-parallel blocks against unsplit ones, on real gloo ranks (CPU).

DiTBlock zero-initialises its adaLN modulation, which makes a fresh block the identity
and every attention/MLP gradient zero; the tiny net randomises it so parity means
something. The end-to-end case builds the production mesh (dp=2 x tp=2 over 4 ranks)
and goes through DDPStrategy, so it covers the wiring, not just the layers.
"""

from __future__ import annotations

import copy
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = [pytest.mark.unit]
WIDTH, HEADS = 16, 4


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from spectramr.models.generators.dit_backbone import DiTBlock
        from spectramr.models.generators.uvit_backbone import UViTBlock

        self.blocks = nn.ModuleList(DiTBlock(WIDTH, HEADS, 2.0) for _ in range(2))
        self.tail = UViTBlock(WIDTH, HEADS, 2.0, takes_skip=False)
        for block in self.blocks:
            nn.init.normal_(block.modulation.weight, std=0.2)

    def forward(self, x):
        cond = x.mean(dim=1)
        for block in self.blocks:
            x = block(x, cond)
        return self.tail(x)


def _sgd(model, steps: int, rank_batch: int | None = None):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(steps):
        gen = torch.Generator().manual_seed(100 + step)
        x = torch.randn(4, 6, WIDTH, generator=gen)
        if rank_batch is not None:
            x = x[rank_batch * 2 : (rank_batch + 1) * 2]
        model(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


def _scenario_blocks_and_checkpoint(rank: int) -> None:
    from spectramr.infrastructure.distributed.tensor_parallel import parallelize_tensor

    group = torch.distributed.group.WORLD
    torch.manual_seed(0)
    model = _TinyDiT()
    reference = copy.deepcopy(model)
    keys = {k: v.shape for k, v in model.state_dict().items()}
    names = parallelize_tensor(model, group)
    assert len(names) == 6  # 3 attentions + 3 MLPs
    assert {k: v.shape for k, v in model.state_dict().items()} == keys, "checkpoint shape changed"
    x = torch.randn(2, 6, WIDTH)
    torch.testing.assert_close(model(x), reference(x))
    other = copy.deepcopy(reference)
    for p in other.parameters():
        p.data.add_(0.5)
    model.load_state_dict(other.state_dict())  # an unsplit checkpoint loads into the split model
    torch.testing.assert_close(model(x), other(x))


def _scenario_trains_like_one_process(rank: int) -> None:
    from spectramr.infrastructure.distributed.tensor_parallel import parallelize_tensor

    torch.manual_seed(0)
    model = _TinyDiT()
    reference = copy.deepcopy(model)
    parallelize_tensor(model, torch.distributed.group.WORLD)
    _sgd(model, 3)
    _sgd(reference, 3)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=1e-5, rtol=1e-4)


def _scenario_mesh_ddp_end_to_end(rank: int) -> None:
    """dp=2 x tp=2 through the production mesh and DDPStrategy, against one process."""
    from types import SimpleNamespace

    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.core.parallel_groups import dp_rank
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategies import DDPStrategy
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    parallel = ParallelismConfigSchema(strategy="ddp", mesh={"tensor": 2})
    mesh = build_parallel_mesh(parallel, device_type="cpu")
    ctx = ParallelContext(
        config=SimpleNamespace(), device=torch.device("cpu"), parallel=parallel, mesh=mesh
    )
    torch.manual_seed(0)
    model = _TinyDiT()
    reference = copy.deepcopy(model)
    plugin = DDPStrategy()
    models = plugin.prepare_models({"generator": model}, ctx)
    wrapped = plugin.adopt(models, {}, {}, ctx).models["generator"]
    _sgd(wrapped, 2, rank_batch=dp_rank())
    _sgd(reference, 2)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=1e-5, rtol=1e-4)


def _scenario_refusals(rank: int) -> None:
    from spectramr.infrastructure.distributed.tensor_parallel import (
        TensorParallelError,
        parallelize_tensor,
    )

    group = torch.distributed.group.WORLD
    with pytest.raises(TensorParallelError, match="no tensor-parallel plan"):
        parallelize_tensor(nn.Sequential(nn.Linear(4, 4)), group)
    torch.manual_seed(rank)  # planted: rank-dependent init
    with pytest.raises(TensorParallelError, match="weights differ"):
        parallelize_tensor(_TinyDiT(), group)
    from spectramr.models.generators.diffusion_transformer_stem import TokenAttention

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="heads do not split"):
        parallelize_tensor(nn.Sequential(TokenAttention(9, 3)), group)


def _worker(rank: int, world: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"](rank)
    finally:
        dist.destroy_process_group()


def _spawn(world: int, scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker, args=(world, os.path.join(tmp, "store"), scenario), nprocs=world, join=True
        )


@pytest.mark.parametrize(
    "scenario", ["blocks_and_checkpoint", "trains_like_one_process", "refusals"]
)
def test_tensor_parallel_two_ranks(scenario: str) -> None:
    _spawn(2, scenario)


def test_mesh_ddp_end_to_end_four_ranks() -> None:
    _spawn(4, "mesh_ddp_end_to_end")
