"""fp16 loss scaling through the pipeline step policy, dp=2 x pp=2 on four gloo ranks.

The last stage back-propagates the scaled loss, every stage unscales its own gradients,
and the non-finite flag spans the world. Built through ``pipeline_cases`` (the plugin's
``prepare_models``/``adopt`` and a real ``StepExecutor``); only the scaler is added.
A clean scaled run must match the float32 pipelined run; an inf on one stage of one
replica must stop every rank and halve every scale; a rank-local flag must not.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.unit.infrastructure.distributed.pipeline_parallel import (  # noqa: E402
    pipeline_cases as pc,
)

pytestmark = [pytest.mark.unit]
_HERE = __name__
WORLD = 4


def _scaler(ctx, poison: bool, local_flag: bool = False):
    from spectramr.infrastructure.training.mixed_precision import (
        ComplexGradScaler,
        needs_cross_rank_inf_flag,
    )

    class _Poison(ComplexGradScaler):
        def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
            if poison:
                grad = optimizer.param_groups[0]["params"][0].grad
                local = grad.to_local() if hasattr(grad, "to_local") else grad
                local.view(-1)[0] = float("inf")
            if not local_flag:
                return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16)
            self.sharded = False  # planted: the flag stays on this rank
            try:
                return super()._unscale_grads_(optimizer, inv_scale, found_inf, allow_fp16)
            finally:
                self.sharded = True

    sharded = needs_cross_rank_inf_flag(ctx.parallel)
    return _Poison("cpu", enabled=True, init_scale=1024.0, sharded=sharded)


def _with_scaler(executor, scaler) -> None:
    executor.amp_helper = SimpleNamespace(
        scaler=scaler, get_autocast_context=contextlib.nullcontext
    )


def _local(gen) -> list:
    return [
        (p.to_local() if hasattr(p, "to_local") else p).detach().clone() for p in gen.parameters()
    ]


def _gather(value) -> list:
    out: list = [None] * WORLD
    torch.distributed.all_gather_object(out, value)
    return out


def _one_step(ctx, poison: bool, local_flag: bool = False):
    from spectramr.core.parallel_groups import dp_rank, dp_world_size

    gen, _, executor, _, _ = pc.build(ctx, "tiny")
    scaler = _scaler(ctx, poison, local_flag)
    _with_scaler(executor, scaler)
    before = _local(gen)
    pc.train(gen, executor, "tiny", dp_rank(), dp_world_size(), steps=[0])
    moved = any(not torch.equal(a, b) for a, b in zip(before, _local(gen), strict=True))
    return _gather(moved), _gather(scaler.get_scale())


def _scenario_scaled(rank: int, strategy: str) -> None:
    from spectramr.core.parallel_groups import dp_rank, dp_world_size

    ctx = pc.context(strategy, WORLD)
    stage = ctx.mesh.device_mesh["pp"].get_local_rank()

    # Clean: the scaled run equals the float32 pipelined run (1024 is a power of two,
    # so scale and unscale are exact) with the clip binding on the unscaled norm.
    gen, _, executor, _, _ = pc.build(ctx, "tiny")
    scaler = _scaler(ctx, poison=False)
    _with_scaler(executor, scaler)
    pc.train(gen, executor, "tiny", dp_rank(), dp_world_size(), steps=range(2))
    plain, _, plain_executor, _, _ = pc.build(ctx, "tiny")
    pc.train(plain, plain_executor, "tiny", dp_rank(), dp_world_size(), steps=range(2))
    for a, b in zip(_local(gen), _local(plain), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert _gather(scaler.get_scale()) == [1024.0] * WORLD

    # An inf on the last stage of replica 0, then on the first stage of replica 1.
    for poisoned in ((1, 0), (0, 1)):
        moved, scales = _one_step(ctx, (stage, dp_rank()) == poisoned)
        assert not any(moved), f"inf on stage/replica {poisoned}, yet ranks stepped: {moved}"
        assert scales == [512.0] * WORLD, scales


def _scenario_planted(rank: int, strategy: str) -> None:
    from spectramr.core.parallel_groups import dp_rank

    ctx = pc.context(strategy, WORLD)
    stage = ctx.mesh.device_mesh["pp"].get_local_rank()
    moved, scales = _one_step(ctx, (stage, dp_rank()) == (1, 0), local_flag=True)
    assert any(moved) and not all(moved), f"a rank-local flag kept the ranks in step: {moved}"
    assert len(set(scales)) == 2, scales

    # The policy refuses a scaler without the world-wide flag outright.
    gen, _, executor, _, _ = pc.build(ctx, "tiny")
    scaler = _scaler(ctx, poison=False)
    scaler.sharded = False
    _with_scaler(executor, scaler)
    with pytest.raises(RuntimeError, match="world-wide non-finite flag"):
        pc.train(gen, executor, "tiny", dp_rank(), 2, steps=[0])


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_pipeline_fp16_scaler_skips_on_every_stage(strategy: str) -> None:
    pc.spawn(WORLD, f"{_HERE}:_scenario_scaled", strategy)


def test_planted_rank_local_flag_lets_stages_diverge() -> None:
    pc.spawn(WORLD, f"{_HERE}:_scenario_planted", "ddp")
