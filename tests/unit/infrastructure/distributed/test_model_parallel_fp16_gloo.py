"""fp16 loss scaling under model parallelism: one rank's inf skips every rank (4 gloo ranks).

Each case builds dp=2 x {tp, cp, sp}=2 through ``DDPStrategy`` / ``FSDPStrategy``, takes
the scaler the training loop would (``needs_cross_rank_inf_flag``), plants an inf in one
rank's gradient at unscale time and steps through ``AMPPolicy``. Every rank must keep its
weights and halve its scale. The planted variant drops the cross-rank flag (the rule
before this change: FSDP only) and must leave the ranks disagreeing.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

WORLD = 4
_DIM = {"tensor": "tp", "context": "cp", "spatial": "sp"}


def _ctx(strategy: str, axis: str):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    extra = {}
    if strategy == "fsdp":
        extra["fsdp"] = {
            "enabled": True,
            "auto_wrap_policy": "transformer_block",
            "transformer_layer_cls": ["DiTBlock", "UViTBlock"],
        }
    if axis == "spatial":
        extra["spatial_probe_shape"] = [8, 6, 6]
    parallel = ParallelismConfigSchema(strategy=strategy, mesh={axis: 2}, **extra)
    config = SimpleNamespace(
        optimization=SimpleNamespace(
            precision=SimpleNamespace(enabled=False, dtype=None),
            compile=SimpleNamespace(enabled=False),
        ),
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=(8, 6, 6))),
        model=SimpleNamespace(in_channels=2),
    )
    mesh = build_parallel_mesh(parallel, device_type="cpu", local_world_size=WORLD)
    return ParallelContext(config=config, device=torch.device("cpu"), parallel=parallel, mesh=mesh)


def _model_and_loss(axis: str):
    """A tiny model for the axis and a loss closure over this data-parallel rank's batch."""
    from spectramr.core.parallel_groups import dp_rank

    gen = torch.Generator().manual_seed(10 + dp_rank())
    if axis == "tensor":
        from tests.unit.infrastructure.distributed.tensor_parallel.test_tensor_parallel_gloo import (
            WIDTH,
            _TinyDiT,
        )

        torch.manual_seed(0)
        x = torch.randn(2, 6, WIDTH, generator=gen)
        return _TinyDiT(), lambda m: m(x).square().mean()
    if axis == "context":
        from tests.unit.infrastructure.distributed.context_parallel.test_context_parallel_gloo import (
            seeded,
        )

        x, t = torch.randn(2, 2, 8, 8, generator=gen), torch.rand(2, generator=gen)
        return seeded(), lambda m: m(x, t).square().mean()
    from tests.unit.infrastructure.distributed.spatial.test_spatial_dp_gloo import _loss, _Net

    torch.manual_seed(0)
    x, target = torch.randn(2, 2, 8, 6, 6, generator=gen), torch.randn(2, 2, 8, 6, 6)
    return _Net(), lambda m: _loss(m(x), target)


def _poisoning_scaler(sharded: bool, poison: bool):
    from spectramr.infrastructure.training.mixed_precision import ComplexGradScaler

    class _Poison(ComplexGradScaler):
        def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
            if poison:  # this rank alone holds a non-finite gradient
                grad = optimizer.param_groups[0]["params"][0].grad
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                local.view(-1)[0] = float("inf")
            return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16)

    return _Poison("cpu", enabled=True, init_scale=1024.0, sharded=sharded)


def _local(model) -> list:
    return [
        (p.to_local() if hasattr(p, "to_local") else p).detach().clone() for p in model.parameters()
    ]


def _prepared(strategy: str, axis: str):
    """``(ctx, wrapped model, loss closure)`` through the production plugin."""
    from spectramr.infrastructure.distributed.strategy_registry import resolve_parallel_strategy

    ctx = _ctx(strategy, axis)
    model, loss_of = _model_and_loss(axis)
    plugin = resolve_parallel_strategy(strategy)
    models = plugin.prepare_models({"generator": model}, ctx)
    return ctx, plugin.adopt(models, {}, {}, ctx).models["generator"], loss_of


def _step(strategy: str, axis: str, poison_at: str | None, sharded: bool | None = None):
    """One scaled step; returns ``(weights moved here, every rank's scale, clip norm)``."""
    import torch.distributed as dist

    from spectramr.core.parallel_groups import dp_rank
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_
    from spectramr.infrastructure.training.mixed_precision import needs_cross_rank_inf_flag
    from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy

    ctx, wrapped, loss_of = _prepared(strategy, axis)
    peer = ctx.mesh.device_mesh[_DIM[axis]].get_local_rank()
    poison = {"peer": peer == 1 and dp_rank() == 0, "replica": peer == 0 and dp_rank() == 1}
    if sharded is None:
        sharded = needs_cross_rank_inf_flag(ctx.parallel)
    scaler = _poisoning_scaler(sharded, poison.get(poison_at, False))
    opt = torch.optim.SGD(wrapped.parameters(), lr=0.5)
    before, norms = _local(wrapped), []

    def clip(m):  # inf: measure the norm the clip sees without changing the step
        norms.append(clip_grad_norm_global_(m, float("inf")))

    AMPPolicy().backward_and_step(
        loss_of(wrapped), opt, wrapped, "g", 0, "test", scaler=scaler, gradient_clipping_fn=clip
    )
    moved = any(not torch.equal(a, b) for a, b in zip(before, _local(wrapped), strict=True))
    scales: list = [None] * WORLD
    dist.all_gather_object(scales, scaler.get_scale())
    return moved, scales, float(norms[0])


def _scenario_global_skip(rank: int, strategy: str, axis: str) -> None:
    import torch.distributed as dist

    for where in ("peer", "replica"):
        moved, scales, _ = _step(strategy, axis, where)
        everyone: list = [None] * WORLD
        dist.all_gather_object(everyone, moved)
        assert not any(everyone), f"inf on one {where} rank, yet ranks stepped: {everyone}"
        assert scales == [512.0] * WORLD, scales

    # A clean step moves every rank, keeps the scale, and clips the unscaled gradients:
    # the norm equals the float32 run's, not 1024 times it.
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_

    moved, scales, norm = _step(strategy, axis, None)
    assert moved and scales == [1024.0] * WORLD
    _, wrapped, loss_of = _prepared(strategy, axis)
    loss_of(wrapped).backward()
    reference = float(clip_grad_norm_global_(wrapped, float("inf")))
    assert norm == pytest.approx(reference, rel=1e-5), (norm, reference)


def _scenario_planted_without_flag(rank: int, strategy: str, axis: str) -> None:
    import torch.distributed as dist

    # A rank-local flag: what DDP x tp/cp/sp ran with before (the flag was FSDP-only).
    moved, _, _ = _step(strategy, axis, "peer", sharded=False)
    everyone: list = [None] * WORLD
    dist.all_gather_object(everyone, moved)
    assert any(everyone) and not all(everyone), f"no divergence without the flag: {everyone}"


def _worker(rank: int, store: str, scenario: str, args: tuple) -> None:
    import torch.distributed as dist

    os.environ.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"](rank, *args)
        # The planted scenario leaves ranks with different scaler and optimizer state;
        # a rank that tears the group down while a peer is still in a collective aborts.
        dist.barrier()
    finally:
        from spectramr.core.parallel_groups import reset_dp_layout

        reset_dp_layout()
        dist.destroy_process_group()


def _spawn(scenario: str, *args) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        store = os.path.join(tmp, "store")
        mp.spawn(_worker, args=(store, scenario, args), nprocs=WORLD, join=True)


CASES = [("ddp", "tensor"), ("fsdp", "tensor"), ("ddp", "context"), ("fsdp", "context")]
CASES += [("ddp", "spatial")]


@pytest.mark.parametrize(("strategy", "axis"), CASES)
def test_inf_on_one_rank_skips_every_rank(strategy: str, axis: str) -> None:
    _spawn("global_skip", strategy, axis)


@pytest.mark.parametrize(("strategy", "axis"), [("ddp", "tensor"), ("fsdp", "context")])
def test_planted_rank_local_flag_lets_ranks_diverge(strategy: str, axis: str) -> None:
    _spawn("planted_without_flag", strategy, axis)


@pytest.mark.parametrize(
    ("strategy", "mesh", "expected"),
    [
        ("ddp", {}, False),
        ("fsdp", {}, True),
        ("ddp", {"tensor": 2}, True),
        ("ddp", {"context": 2}, True),
        ("ddp", {"spatial": 2}, True),
        ("ddp", {"pipeline": 2}, True),
    ],
)
def test_cross_rank_flag_follows_the_layout(strategy, mesh, expected) -> None:
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.training.mixed_precision import needs_cross_rank_inf_flag

    extra = {"fsdp": {"enabled": True}} if strategy == "fsdp" else {}
    parallel = ParallelismConfigSchema(strategy=strategy, mesh=mesh, **extra)
    assert needs_cross_rank_inf_flag(parallel) is expected
