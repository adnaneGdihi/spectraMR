"""Depth-split training under FSDP2 on 4 CPU ranks (gloo): dp=2 x sp=2 against one process.

The production order of the training director: a bare EMA shadow, then
``FSDPStrategy.prepare_models`` (convert, gate, shard over dp x sp), ``prepare_ema``,
the optimizer, ``adopt``. Three SGD-with-momentum steps must reproduce the unsplit
reference, the clip norm must be the reference's global norm, and the gathered
checkpoint must carry the unsplit model's keys. Two planted defects must miss: a shard
mesh over dp alone (sp peers left out of the reduce) and a gather without the ``x sp``.
"""

from __future__ import annotations

import copy
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.unit.infrastructure.distributed.spatial.test_spatial_dp_gloo import (  # noqa: E402
    DP,
    TOLERANCE,
    VOLUME,
    WORLD,
    _batches,
    _loss,
    _Net,
)

pytestmark = [pytest.mark.unit]

SP = 2
LR, MOMENTUM, STEPS, DECAY = 0.05, 0.9, 3, 0.5


def _plant(name: str | None) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed import fsdp_wrap
    from spectramr.infrastructure.distributed.spatial import halo
    from spectramr.infrastructure.distributed.spatial import wrapper as wrapper_mod

    if name == "dp_only_shard_mesh":
        fsdp_wrap.fsdp_shard_mesh = lambda mesh, sharding: mesh["dp_shard"]
    elif name == "gather_without_sp":

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


def _context(sharding: str):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    parallel = ParallelismConfigSchema(
        strategy="fsdp",
        fsdp={"enabled": True, "sharding_strategy": sharding, "min_num_params": 1},
        mesh={"spatial": SP},
        spatial_probe_shape=list(VOLUME[2:]),
    )
    # hybrid_shard: two ranks per "node" -> dp_replicate=2, and the shard axis is sp alone.
    local = WORLD if sharding == "full_shard" else 2
    pm = build_parallel_mesh(parallel, device_type="cpu", world_size=WORLD, local_world_size=local)
    config = SimpleNamespace(
        optimization=SimpleNamespace(
            compile=SimpleNamespace(enabled=False),
            precision=SimpleNamespace(enabled=False, dtype=None),
        ),
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=VOLUME[2:])),
        model=SimpleNamespace(in_channels=VOLUME[1]),
    )
    return ParallelContext(config=config, device=torch.device("cpu"), parallel=parallel, mesh=pm)


def _full(model) -> torch.Tensor:
    return torch.cat([p.full_tensor().reshape(-1) for p in model.parameters()])


def _grad_norm(model) -> float:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    return float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)


def _reference(model, xs, ts):
    """The unsplit run: the replicas' mean gradient, SGD with momentum, an EMA after each step."""
    from spectramr.infrastructure.optimization.ema import ModelEma

    ema = ModelEma(model, decay=DECAY, warmup=False)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    norms = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for x, t in zip(xs, ts, strict=True):
            (_loss(model(x), t) / DP).backward()
        norms.append(_grad_norm(model))
        opt.step()
        ema.update(model)
    return norms, ema


def _sharded_run(sharding: str = "full_shard", plant: str | None = None):
    """Train the sharded model; return it with its context, optimizer, EMA and step errors."""
    from spectramr.core.module_utils import unwrap_model
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy
    from spectramr.infrastructure.optimization.ema import ModelEma

    _plant(plant)
    xs, ts = _batches()
    ctx = _context(sharding)
    torch.manual_seed(0)
    model = _Net()
    reference = copy.deepcopy(model)
    ref_norms, ref_ema = _reference(reference, xs, ts)

    ema = ModelEma(model, decay=DECAY, warmup=False)  # built before Stage A, as the director does
    strategy = FSDPStrategy()
    models = strategy.prepare_models({"generator": model}, ctx)
    strategy.prepare_ema(ema, models, ctx)
    gen = models["generator"]
    opt = torch.optim.SGD(gen.parameters(), lr=LR, momentum=MOMENTUM)
    result = strategy.adopt(models, {"opt_g": opt}, {}, ctx)
    replica = ctx.mesh.layout.rank
    norm_errors = []
    for ref_norm in ref_norms:
        opt.zero_grad(set_to_none=True)
        _loss(gen(xs[replica]), ts[replica]).backward()
        norm_errors.append(abs(_grad_norm(gen) - ref_norm) / ref_norm)
        opt.step()
        ema.update(unwrap_model(gen))
    torch.manual_seed(0)
    start = torch.cat([p.detach().reshape(-1) for p in _Net().parameters()])
    expected = torch.cat([p.detach().reshape(-1) for p in reference.parameters()])
    update_error = float((_full(gen) - expected).norm() / (expected - start).norm())
    run = SimpleNamespace(gen=gen, ctx=ctx, opt=opt, ema=ema, ref_ema=ref_ema, result=result)
    return run, update_error, max(norm_errors), reference


def _scenario_parity_full_shard() -> None:
    run, update_error, norm_error, _ = _sharded_run("full_shard")
    assert update_error < TOLERANCE, f"update misses the unsplit run by {update_error:.2e}"
    assert norm_error < TOLERANCE, f"grad norm misses the global norm by {norm_error:.2e}"
    record = run.result.provenance["spatial"]
    assert record["grad_reduce_group"] == "dp x sp (fsdp shard mesh)"
    assert record["mesh_spatial"] == SP and record["spatial_size"] == SP
    mesh = next(run.gen.parameters()).device_mesh
    assert mesh.size() == DP * SP, "FSDP must shard over dp x sp"


def _scenario_parity_hybrid_shard() -> None:
    run, update_error, norm_error, _ = _sharded_run("hybrid_shard")
    assert update_error < TOLERANCE, f"hybrid update misses by {update_error:.2e}"
    assert norm_error < TOLERANCE, f"hybrid grad norm misses by {norm_error:.2e}"
    assert tuple(next(run.gen.parameters()).device_mesh.shape) == (DP, SP)


def _scenario_planted_dp_only_shard_mesh() -> None:
    _, update_error, norm_error, _ = _sharded_run(plant="dp_only_shard_mesh")
    assert update_error > TOLERANCE, f"dp-only reduce still matches ({update_error:.2e})"
    assert norm_error > TOLERANCE, f"dp-only grad norm still matches ({norm_error:.2e})"


def _scenario_planted_gather_without_sp() -> None:
    _, update_error, norm_error, _ = _sharded_run(plant="gather_without_sp")
    assert update_error > TOLERANCE, f"gather without x sp still matches ({update_error:.2e})"
    assert norm_error > TOLERANCE, f"gather without x sp: norm still matches ({norm_error:.2e})"


def _scenario_checkpoint_and_ema() -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.state_io import FSDPStateIO

    run, _, _, reference = _sharded_run("full_shard")
    io, gen = FSDPStateIO(), run.gen
    keys = list(_Net().state_dict())
    payload = [io.model_state(gen), io.optimizer_state(gen, run.opt), run.ema.shadow_state_dict()]
    dist.broadcast_object_list(payload, src=0)  # every rank reads the file on resume
    saved, optim, shadow = payload
    assert list(saved) == keys, f"checkpoint keys {list(saved)[:2]} are not the unsplit keys"
    plain = _Net()
    plain.load_state_dict(saved, strict=True)
    torch.testing.assert_close(_full(gen), torch.cat([p.reshape(-1) for p in plain.parameters()]))
    torch.testing.assert_close(plain.state_dict(), reference.state_dict(), atol=1e-5, rtol=1e-4)
    assert sorted(optim["state"]) == sorted(n for n, _ in _Net().named_parameters())
    want = run.ref_ema.shadow_state_dict()
    assert list(shadow) == list(want)
    torch.testing.assert_close(shadow, want, atol=1e-5, rtol=1e-4)

    # Resume: perturb, then load the gathered model and optimizer back into the sharded run.
    with torch.no_grad():
        for p in gen.parameters():
            p.add_(1.0)
    io.load_model_state(gen, saved)
    io.load_optimizer_state(gen, run.opt, optim)
    torch.testing.assert_close(_full(gen), torch.cat([p.reshape(-1) for p in plain.parameters()]))

    # Validation under the EMA swap runs collectively and grades the shadow's weights.
    x = _batches()[0][run.ctx.mesh.layout.rank]
    with io.swap_in(gen, run.ema.module), torch.no_grad():
        got = gen(x)[0]
    with torch.no_grad():
        ref_shadow = run.ref_ema.module
        torch.testing.assert_close(got, ref_shadow(x)[0], atol=1e-5, rtol=1e-4)


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"]()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        "parity_full_shard",
        "parity_hybrid_shard",
        "planted_dp_only_shard_mesh",
        "planted_gather_without_sp",
        "checkpoint_and_ema",
    ],
)
def test_spatial_fsdp_step(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)
