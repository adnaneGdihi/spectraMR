"""fp16 under FSDP2: the sharded grad scaler across two real ranks (gloo, CPU).

The non-finite flag is all-reduced, so an inf gradient on one rank must make every rank
skip the same step and back off the scale identically. Models are sharded through
``maybe_wrap_with_fsdp``, the production path.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

WORLD = 2


def _run_mesh(fsdp_cfg):
    """The run's one mesh for this worker, built the production way (after init)."""
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh

    parallel = SimpleNamespace(
        strategy="fsdp",
        fsdp=fsdp_cfg,
        mesh=SimpleNamespace(non_dp_size=1, tensor=1, context=1, spatial=1, pipeline=1),
    )
    return build_parallel_mesh(
        parallel, device_type="cpu", world_size=WORLD, local_world_size=WORLD
    ).device_mesh


def _shard(net, **fsdp):
    from spectramr.infrastructure.distributed.fsdp_wrap import maybe_wrap_with_fsdp

    base = {
        "enabled": True,
        "sharding_strategy": "full_shard",
        "auto_wrap_policy": "none",
        "transformer_layer_cls": [],
        "min_num_params": 1,
        "cpu_offload": False,
        "mixed_precision": "fp16",
        "reduce_dtype": "float32",
    }
    base.update(fsdp)
    cfg = SimpleNamespace(fsdp=SimpleNamespace(**base))
    return maybe_wrap_with_fsdp(
        net, cfg, None, torch.device("cpu"), mesh=_run_mesh(cfg.fsdp)
    )


def _net():
    from torch import nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def _full(net):
    return [p.full_tensor().detach().clone() for p in net.parameters()]


def _all_scales(scaler):
    import torch.distributed as dist

    gathered = [None] * WORLD
    dist.all_gather_object(gathered, scaler.get_scale())
    return gathered


def _one_step(net, opt, scaler, *, rank, poison_rank):
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
    loss = net(x).float().square().sum()
    scaler.scale(loss).backward()
    if rank == poison_rank:
        first = next(iter(net.parameters())).grad.to_local()
        first.view(-1)[0] = float("inf")
    scaler.step(opt)
    scaler.update()


def _scenario_inf_on_rank1_skips_everywhere(rank: int) -> None:
    from spectramr.infrastructure.training.mixed_precision import ComplexGradScaler

    net = _shard(_net())
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    scaler = ComplexGradScaler("cpu", enabled=True, init_scale=1024.0, sharded=True)
    before = _full(net)
    _one_step(net, opt, scaler, rank=rank, poison_rank=1)
    for old, new in zip(before, _full(net), strict=True):
        assert torch.equal(old, new), "a step ran although a rank saw inf"
    assert _all_scales(scaler) == [512.0, 512.0]


def _scenario_clean_step_runs_everywhere(rank: int) -> None:
    from spectramr.infrastructure.training.mixed_precision import ComplexGradScaler

    net = _shard(_net())
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    scaler = ComplexGradScaler("cpu", enabled=True, init_scale=1024.0, sharded=True)
    before = _full(net)
    _one_step(net, opt, scaler, rank=rank, poison_rank=-1)
    after = _full(net)
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert _all_scales(scaler) == [1024.0, 1024.0]


def _scenario_complex_parameters_refused(rank: int) -> None:
    from torch import nn

    from spectramr.infrastructure.training.mixed_precision import ComplexGradScaler

    class _Complex(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8, dtype=torch.complex64))

        def forward(self, x):
            return x @ self.w

    for half in ("fp16", "bf16"):  # neither half type has a complex counterpart
        with pytest.raises(ValueError, match=f"FSDP {half} cannot hold the complex parameters"):
            _shard(_Complex(), mixed_precision=half)
    # The scaler itself is the one owner for complex grads: it builds and steps on CPU.
    assert ComplexGradScaler("cpu", enabled=True, sharded=True).get_scale() > 0


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        globals()[f"_scenario_{scenario}"](rank)
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


@pytest.mark.parametrize(
    "scenario",
    [
        "inf_on_rank1_skips_everywhere",
        "clean_step_runs_everywhere",
        "complex_parameters_refused",
    ],
)
def test_fsdp2_fp16_two_ranks(scenario: str) -> None:
    _spawn(scenario)
