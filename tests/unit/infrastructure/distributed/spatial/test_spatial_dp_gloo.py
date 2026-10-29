"""Depth-split training on 4 CPU ranks (gloo): dp=2 x sp=2, one DDP step against one process.

Ranks are laid out row-major on the mesh, so the spatial peers of data-parallel replica
``d`` are ranks ``(2d, 2d+1)``. The unsplit reference runs on every rank with the full
model and both replicas' batches; the sharded run must reproduce its SGD update. Two
planted gradients must miss it: one without the ``x sp`` factor on a gathered (split)
output, one with it wrongly applied to a replicated output.
"""

from __future__ import annotations

import copy
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

pytestmark = [pytest.mark.unit]

WORLD, DP, SP = 4, 2, 2
VOLUME = (2, 2, 8, 6, 6)  # [B per replica, C, D, H, W]; D = 4 planes per rank
GATE = (1, 2, 8, 6, 6)
LR = 0.1
TOLERANCE = 1e-3


class _Net(nn.Module):
    """Convs with a stride, a norm and a transposed conv, plus a replicated pooled head."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(2, 4, 3, padding=1),
            nn.GroupNorm(2, 4),
            nn.PReLU(),
            nn.Conv3d(4, 4, 3, stride=2, padding=1),
            nn.ConvTranspose3d(4, 2, 2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        y = self.body(x)
        return y, self.pool(y).flatten(1)


def _loss(out, target):
    # The pooled term is weighted up so a replicated-path error cannot hide under the MSE.
    y, pooled = out
    return ((y - target) ** 2).mean() + 10 * (pooled**2).mean()


def _batches():
    gens = [torch.Generator().manual_seed(100 + d) for d in range(DP)]
    xs = [torch.randn(VOLUME, generator=g) for g in gens]
    ts = [torch.randn(VOLUME, generator=g) for g in gens]
    return xs, ts


def _flat(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _parallel():
    mesh = SimpleNamespace(tensor=1, context=1, spatial=SP, pipeline=1, non_dp_size=SP)
    return SimpleNamespace(strategy="ddp", mesh=mesh)


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"]()
    finally:
        dist.destroy_process_group()


def _one_step_relative_error(plant: str | None = None) -> float:
    """Sharded SGD step vs the unsplit reference step; returns |sharded - ref| / |update|."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from spectramr.core.module_utils import unwrap_model
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.spatial import halo
    from spectramr.infrastructure.distributed.spatial import wrapper as wrapper_mod

    if plant == "gather_without_sp":

        class _NoScale(torch.autograd.Function):
            @staticmethod
            def forward(ctx, slab, group):
                ctx.group = group
                parts = [torch.empty_like(slab) for _ in range(dist.get_world_size(group))]
                dist.all_gather(parts, slab.contiguous(), group=group)
                return torch.cat(parts, dim=2)

            @staticmethod
            def backward(ctx, grad):  # the x sp factor left out
                return halo.split_depth(grad.contiguous(), ctx.group), None

        wrapper_mod._GatherDepth = _NoScale
    elif plant == "replicated_with_sp":

        class _ScaleReplicated(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, group):
                ctx.world = dist.get_world_size(group)
                return x.clone()

            @staticmethod
            def backward(ctx, grad):  # the x sp factor wrongly applied to a replicated leaf
                return grad * ctx.world, None

        def _forward(self, x):
            slab = halo.split_depth(x, self._spatial_group)
            with halo.spatial_scope():
                out = self.module(slab)
            leaves, spec = wrapper_mod.tree_flatten(out)
            gathered = [
                wrapper_mod._GatherDepth.apply(leaf, self._spatial_group)
                if split
                else _ScaleReplicated.apply(leaf, self._spatial_group)
                for leaf, split in zip(leaves, self.split_mask, strict=True)
            ]
            return wrapper_mod.tree_unflatten(gathered, spec)

        wrapper_mod.SpatialModel.forward = _forward

    xs, ts = _batches()
    pm = build_parallel_mesh(
        _parallel(), device_type="cpu", world_size=WORLD, local_world_size=WORLD
    )
    replica = pm.layout.rank

    torch.manual_seed(0)
    model = _Net()
    start = _flat(model)

    # Unsplit reference: mean of the two replicas' gradients, one SGD step.
    reference = copy.deepcopy(model)
    grads = [torch.zeros_like(p) for p in reference.parameters()]
    for x, t in zip(xs, ts, strict=True):
        reference.zero_grad(set_to_none=True)
        _loss(reference(x), t).backward()
        for g, p in zip(grads, reference.parameters(), strict=True):
            g += p.grad / DP
    with torch.no_grad():
        for g, p in zip(grads, reference.parameters(), strict=True):
            p -= LR * g
    expected = _flat(reference)

    wrapped = wrapper_mod.wrap_spatial(model, pm.spatial_group, GATE)
    ddp = DistributedDataParallel(wrapped, process_group=pm.grad_group)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=LR)
    optimizer.zero_grad(set_to_none=True)
    _loss(ddp(xs[replica]), ts[replica]).backward()
    optimizer.step()
    got = _flat(model)

    # Checkpoints: the unwrapped model has the unsplit keys and the stepped values.
    saved = unwrap_model(ddp).state_dict()
    plain = _Net()
    assert list(saved) == list(plain.state_dict()), "spatial checkpoint keys differ"
    plain.load_state_dict(saved, strict=True)
    torch.testing.assert_close(_flat(plain), got)

    update = (expected - start).norm()
    return float((got - expected).norm() / update)


def _scenario_parity() -> None:
    error = _one_step_relative_error()
    assert error < TOLERANCE, f"sharded step misses the unsplit step by {error:.2e}"


def _scenario_planted_gather_without_sp() -> None:
    error = _one_step_relative_error("gather_without_sp")
    assert error > TOLERANCE, f"planted gradient without x sp still matches ({error:.2e})"


def _scenario_planted_replicated_with_sp() -> None:
    error = _one_step_relative_error("replicated_with_sp")
    assert error > TOLERANCE, f"planted replicated x sp still matches ({error:.2e})"


def _scenario_production_strategy() -> None:
    """The real wiring: DDPStrategy.prepare_models + adopt, then one forward against the unsplit."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategies import DDPStrategy
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    xs, _ = _batches()
    pm = build_parallel_mesh(
        _parallel(), device_type="cpu", world_size=WORLD, local_world_size=WORLD
    )
    parallel = ParallelismConfigSchema(
        strategy="ddp", mesh={"spatial": SP}, spatial_probe_shape=[8, 6, 6]
    )
    config = SimpleNamespace(
        optimization=SimpleNamespace(compile=SimpleNamespace(enabled=False)),
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=(8, 6, 6))),
        model=SimpleNamespace(in_channels=2),
    )
    ctx = ParallelContext(config=config, device=torch.device("cpu"), parallel=parallel, mesh=pm)
    torch.manual_seed(0)
    model = _Net()
    reference = copy.deepcopy(model).eval()
    strategy = DDPStrategy()
    models = strategy.prepare_models({"generator": model}, ctx)
    result = strategy.adopt(models, {}, {}, ctx)
    ddp = result.models["generator"]
    assert isinstance(ddp, DistributedDataParallel)
    assert ddp.process_group is pm.grad_group
    record = result.provenance["spatial"]
    assert record["mesh_spatial"] == SP and record["spatial_size"] == SP
    assert record["probe_shape"] == [1, 2, 8, 6, 6]
    assert record["worst_relative_error"] < TOLERANCE
    assert record["split_outputs"] == 1 and record["replicated_outputs"] == 1
    x = xs[pm.layout.rank]
    with torch.no_grad():
        got_y, got_p = ddp(x)
        want_y, want_p = reference(x)
    torch.testing.assert_close(got_y, want_y, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(got_p, want_p, atol=1e-5, rtol=1e-4)
    dist.barrier()


def _scenario_replicas_and_peers() -> None:
    """Spatial peers share a data-parallel index; the data rank follows it."""
    import torch.distributed as dist

    from spectramr.core.parallel_groups import dp_rank, dp_world_size
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.pipelines.parallel import resolve_data_rank

    pm = build_parallel_mesh(
        _parallel(), device_type="cpu", world_size=WORLD, local_world_size=WORLD
    )
    assert dp_world_size() == DP
    assert pm.spatial_group is not None and pm.grad_group is not None
    cfg = SimpleNamespace(parallel=SimpleNamespace(strategy="ddp"))
    seen: list = [None] * WORLD
    dist.all_gather_object(seen, (dp_rank(), resolve_data_rank(cfg)))
    assert [s[0] for s in seen] == [0, 0, 1, 1]
    assert [s[1] for s in seen] == [0, 0, 1, 1]


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker,
            args=(os.path.join(tmp, "store"), scenario),
            nprocs=WORLD,
            join=True,
        )


@pytest.mark.parametrize(
    "scenario",
    [
        "parity",
        "planted_gather_without_sp",
        "planted_replicated_with_sp",
        "production_strategy",
        "replicas_and_peers",
    ],
)
def test_spatial_data_parallel_step(scenario: str) -> None:
    _spawn(scenario)
