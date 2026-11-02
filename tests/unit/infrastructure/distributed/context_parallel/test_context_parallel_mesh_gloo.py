"""dp=2 x cp=2 on four gloo ranks, through the production path, against one process.

``build_parallel_mesh`` -> ``DDPStrategy`` / ``FSDPStrategy.prepare_models`` -> ``adopt``,
then a few clipped SGD steps with the training loop's clip helper. The clip threshold
sits below the reference norm, so a norm that double-counts the context peers fails.
Each data-parallel rank takes half the batch; its two context peers take the same half.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.module_utils import unwrap_model  # noqa: E402
from tests.unit.infrastructure.distributed.context_parallel.test_context_parallel_gloo import (  # noqa: E402
    seeded,
    tiny_dit,
    tiny_uvit,
)

pytestmark = [pytest.mark.unit]
MAX_NORM = 0.05


def _config(precision: str | None = None, compile_: bool = False):
    return SimpleNamespace(
        optimization=SimpleNamespace(
            precision=SimpleNamespace(enabled=precision is not None, dtype=precision),
            compile=SimpleNamespace(enabled=compile_),
        )
    )


def _context(strategy: str, sharding: str = "full_shard", local_world: int = 4):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    fsdp = {}
    if strategy == "fsdp":
        fsdp = {
            "fsdp": {
                "enabled": True,
                "sharding_strategy": sharding,
                "auto_wrap_policy": "transformer_block",
                "transformer_layer_cls": ["DiTBlock", "UViTBlock"],
            }
        }
    parallel = ParallelismConfigSchema(strategy=strategy, mesh={"context": 2}, **fsdp)
    mesh = build_parallel_mesh(parallel, device_type="cpu", local_world_size=local_world)
    return ParallelContext(
        config=_config(), device=torch.device("cpu"), parallel=parallel, mesh=mesh
    )


def _batch(step: int, side: int, shard: int | None):
    gen = torch.Generator().manual_seed(100 + step)
    x, t = torch.randn(4, 2, side, side, generator=gen), torch.rand(4, generator=gen)
    return (x, t) if shard is None else (x[shard * 2 : shard * 2 + 2], t[shard * 2 : shard * 2 + 2])


def _train(model, clip, side: int, shard: int | None, ema=None, steps: int = 3) -> list[float]:
    opt, norms = torch.optim.SGD(model.parameters(), lr=0.5), []
    for step in range(steps):
        model(*_batch(step, side, shard)).square().mean().backward()
        norms.append(float(clip(model, MAX_NORM)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(unwrap_model(model))
    return norms


def _reference_clip(model, max_norm):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _prepare(plugin, make, ctx):
    from spectramr.infrastructure.optimization.ema import ModelEma

    model = seeded(make)
    reference = copy.deepcopy(model)
    # The shadow is a copy of the bare model, taken before the split, as in the pipeline.
    ema = ModelEma(model, decay=0.5, warmup=False)
    ref_ema = ModelEma(reference, decay=0.5, warmup=False)
    models = plugin.prepare_models({"generator": model}, ctx)
    plugin.prepare_ema(ema, models, ctx)
    result = plugin.adopt(models, {}, {}, ctx)
    return result.models["generator"], reference, (ema, ref_ema), result.provenance


def _check_parity(rank, wrapped, reference, emas, make, side, io) -> None:
    from spectramr.core.parallel_groups import dp_rank
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_

    ema, ref_ema = emas
    norms = _train(wrapped, clip_grad_norm_global_, side, dp_rank(), ema)
    ref_norms = _train(reference, _reference_clip, side, None, ref_ema)
    assert min(ref_norms) > MAX_NORM, "clipping must bind or the norm is untested"
    torch.testing.assert_close(torch.tensor(norms), torch.tensor(ref_norms), atol=1e-5, rtol=1e-4)
    state, shadow = io.model_state(wrapped), ema.shadow_state_dict()
    if rank == 0:
        for key, value in ref_ema.shadow_state_dict().items():
            torch.testing.assert_close(shadow[key], value, atol=1e-5, rtol=1e-4)
        expected = reference.state_dict()
        assert {k: v.shape for k, v in state.items()} == {k: v.shape for k, v in expected.items()}
        for key, value in expected.items():
            torch.testing.assert_close(state[key], value, atol=1e-5, rtol=1e-4)
        unsplit = seeded(make)
        unsplit.load_state_dict(state, strict=True)  # a cp checkpoint loads unsplit
        x, t = _batch(9, side, None)
        torch.testing.assert_close(unsplit(x, t), reference(x, t), atol=1e-5, rtol=1e-4)
    x, t = _batch(9, side, None)
    with torch.no_grad():  # the validation forward, collective on every rank
        torch.testing.assert_close(wrapped(x, t), reference(x, t), atol=1e-5, rtol=1e-4)
        with io.swap_in(wrapped, ema.module):  # the EMA validation swap
            torch.testing.assert_close(wrapped(x, t), ref_ema.module(x, t), atol=1e-5, rtol=1e-4)


def _scenario_ddp(rank: int) -> None:
    from spectramr.infrastructure.distributed.state_io import PassthroughStateIO
    from spectramr.infrastructure.distributed.strategies import DDPStrategy

    ctx = _context("ddp")
    for make, side, attentions in ((tiny_dit, 8, 2), (tiny_uvit, 6, 3)):
        wrapped, reference, emas, record = _prepare(DDPStrategy(), make, ctx)
        assert wrapped.process_group.size() == 4, "DDP must average over dp x cp"
        assert record["context_parallel"]["size"] == 2
        assert record["context_parallel"]["attention_layers"] == attentions
        _check_parity(rank, wrapped, reference, emas, make, side, PassthroughStateIO())


def _scenario_ddp_planted(rank: int) -> None:
    """Averaging over dp only leaves the trunk gradients x cp too large: the run diverges."""
    from spectramr.core.parallel_groups import dp_group, dp_rank
    from spectramr.infrastructure.distributed import strategies

    ctx = _context("ddp")
    original = strategies._grad_process_group
    strategies._grad_process_group = lambda _ctx: dp_group()
    try:
        wrapped, reference, _, _ = _prepare(strategies.DDPStrategy(), tiny_dit, ctx)
    finally:
        strategies._grad_process_group = original
    _train(wrapped, _reference_clip, 8, dp_rank())
    _train(reference, _reference_clip, 8, None)
    state = wrapped.module.state_dict()
    assert not all(
        torch.allclose(state[k], v, atol=1e-5, rtol=1e-4) for k, v in reference.state_dict().items()
    )


def _scenario_fsdp(rank: int, sharding: str, local_world: int) -> None:
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy

    ctx = _context("fsdp", sharding, local_world)
    wrapped, reference, emas, record = _prepare(FSDPStrategy(), tiny_dit, ctx)
    assert record["context_parallel"]["grad_reduce_group"] == "dp x cp"
    mesh = next(wrapped.parameters()).device_mesh
    assert mesh.size() == 4, "FSDP must shard and reduce over dp x cp"
    _check_parity(rank, wrapped, reference, emas, tiny_dit, 8, FSDPStateIO())


def _scenario_refusals(rank: int) -> None:
    from spectramr.infrastructure.distributed.strategies import DDPStrategy, FSDPStrategy

    with pytest.raises(ValueError, match="no_shard"):
        FSDPStrategy().prepare_models({"generator": seeded()}, _context("fsdp", "no_shard"))
    ctx = _context("fsdp")
    # Mixed precision is verified (test_fsdp_model_parallel_fp16_gloo.py); compile is not.
    with pytest.raises(RuntimeError, match=r"optimization\.compile"):
        FSDPStrategy().prepare_models(
            {"generator": seeded()}, dataclasses.replace(ctx, config=_config(compile_=True))
        )
    with pytest.raises(RuntimeError, match=r"torch\.compile"):
        DDPStrategy().prepare_models(
            {"generator": seeded()},
            dataclasses.replace(_context("ddp"), config=_config(compile_=True)),
        )


def _worker(rank: int, world: int, store: str, scenario: str, args: tuple) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"](rank, *args)
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str, *args) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(4, os.path.join(tmp, "store"), scenario, args), nprocs=4, join=True)


@pytest.mark.parametrize("scenario", ["ddp", "ddp_planted", "refusals"])
def test_context_parallel_ddp_four_ranks(scenario: str) -> None:
    _spawn(scenario)


@pytest.mark.parametrize(("sharding", "local_world"), [("full_shard", 4), ("hybrid_shard", 2)])
def test_context_parallel_fsdp_matches_one_process(sharding: str, local_world: int) -> None:
    _spawn("fsdp", sharding, local_world)
