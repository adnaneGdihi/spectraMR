"""FSDP2 over tensor-parallel blocks (dp=2 x tp=2 on four gloo ranks) against one process.

Every case goes through the production path: ``build_parallel_mesh`` ->
``FSDPStrategy.prepare_models`` / ``prepare_ema`` / ``adopt``, the clip helper the
training loop calls, and ``FSDPStateIO`` for checkpoints. The clip threshold is set
below the reference norm so clipping binds, which is what lets the planted
"no tensor-group reduction" variant fail.
"""

from __future__ import annotations

import contextlib
import copy
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.unit.infrastructure.distributed.tensor_parallel.test_tensor_parallel_gloo import (  # noqa: E402
    WIDTH,
    _TinyDiT,
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


def _context(sharding: str, local_world: int, **fsdp):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    block = {
        "enabled": True,
        "sharding_strategy": sharding,
        "auto_wrap_policy": "transformer_block",
        "transformer_layer_cls": ["DiTBlock", "UViTBlock"],
        **fsdp,
    }
    parallel = ParallelismConfigSchema(strategy="fsdp", fsdp=block, mesh={"tensor": 2})
    mesh = build_parallel_mesh(parallel, device_type="cpu", local_world_size=local_world)
    return ParallelContext(
        config=_config(), device=torch.device("cpu"), parallel=parallel, mesh=mesh
    )


def _batch(step: int, shard: int | None):
    x = torch.randn(4, 6, WIDTH, generator=torch.Generator().manual_seed(100 + step))
    return x if shard is None else x[shard * 2 : (shard + 1) * 2]


def _step(model, opt, clip, step: int, shard: int | None) -> float:
    model(_batch(step, shard)).square().mean().backward()
    norm = float(clip(model, MAX_NORM))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return norm


def _adam(model):
    return torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-3)


def _train(model, ema, clip, shard: int | None):
    """Two SGD then two Adam steps, clipped by ``clip``, with an EMA update after each.

    The k part of ``qkv.bias`` has an exactly zero gradient (softmax ignores it), so its
    float noise differs between layouts; Adam's default eps would normalise that noise
    into lr-sized steps, hence the larger eps. Returns the norms and the Adam optimizer.
    """
    norms, adam = [], _adam(model)
    for step, opt in enumerate([torch.optim.SGD(model.parameters(), lr=0.5)] * 2 + [adam] * 2):
        norms.append(_step(model, opt, clip, step, shard))
        ema.update(model)
    return norms, adam


def _reference_clip(model, max_norm):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _naive_clip(model, max_norm):  # planted: each tensor rank clips by its own partial norm
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return norm.full_tensor() if hasattr(norm, "full_tensor") else norm


def _build(ctx):
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy
    from spectramr.infrastructure.optimization.ema import ModelEma

    torch.manual_seed(0)
    model = _TinyDiT()
    reference = copy.deepcopy(model)
    ema = ModelEma(model, decay=0.5, warmup=False)
    ref_ema = ModelEma(reference, decay=0.5, warmup=False)
    plugin = FSDPStrategy()
    models = plugin.prepare_models({"generator": model}, ctx)
    plugin.prepare_ema(ema, models, ctx)
    record = plugin.adopt(models, {}, {}, ctx).provenance
    return model, reference, ema, ref_ema, record


def _assert_state_matches(state: dict, expected: dict, *, equal: bool = True) -> None:
    assert {k: tuple(v.shape) for k, v in state.items()} == {
        k: tuple(v.shape) for k, v in expected.items()
    }, "checkpoint keys/shapes differ from the unsplit model"
    close = all(torch.allclose(state[k], v, atol=1e-5, rtol=1e-4) for k, v in expected.items())
    assert close is equal, "weights match the single process" if not equal else "weights differ"


def _scenario_parity(rank: int, sharding: str, local_world: int) -> None:
    from spectramr.core.parallel_groups import dp_rank
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO

    ctx = _context(sharding, local_world)
    model, reference, ema, ref_ema, record = _build(ctx)
    # 3 attentions (qkv.weight, qkv.bias, proj.weight) + 3 MLPs (0.weight, 0.bias, 2.weight).
    assert record["tensor_parallel"] == {
        "size": 2,
        "blocks": 6,
        "split_parameters": 18,
        "order": "tp_then_fsdp",
    }, record
    norms, adam = _train(model, ema, clip_grad_norm_global_, dp_rank())
    ref_norms, ref_adam = _train(reference, ref_ema, _reference_clip, None)
    assert min(ref_norms) > MAX_NORM, "clipping must bind or the norm is untested"
    torch.testing.assert_close(torch.tensor(norms), torch.tensor(ref_norms), atol=1e-5, rtol=1e-4)

    io = FSDPStateIO()
    state, shadow = io.model_state(model), ema.shadow_state_dict()
    if rank == 0:  # cpu_offload gathers to rank 0 only
        _assert_state_matches(state, reference.state_dict())
        shadow_ref = ref_ema.shadow_state_dict()
        assert set(shadow) == set(shadow_ref)
        _assert_state_matches(
            {k: v for k, v in shadow.items() if k != "_ema_num_updates"},
            {k: v for k, v in shadow_ref.items() if k != "_ema_num_updates"},
        )
        unsplit = _TinyDiT()
        unsplit.load_state_dict(state, strict=True)  # an FSDP x tp checkpoint loads unsplit
        x = _batch(9, None)
        torch.testing.assert_close(unsplit(x), reference(x), atol=1e-5, rtol=1e-4)

    x = _batch(9, None)
    with io.swap_in(model, ema.module):  # the validation swap, collective on every rank
        torch.testing.assert_close(model(x), ref_ema.module(x), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(model(x), reference(x), atol=1e-5, rtol=1e-4)

    # Resume into a fresh FSDP x tp run: weights, Adam moments and the EMA shadow, read
    # by every rank as from the file; then one more step must still match one process.
    files = [state, io.optimizer_state(model, adam), shadow]
    torch.distributed.broadcast_object_list(files, src=0)
    fresh, _, fresh_ema, _, _ = _build(ctx)
    fresh_adam = _adam(fresh)
    io.load_model_state(fresh, files[0])
    io.load_optimizer_state(fresh, fresh_adam, files[1])
    fresh_ema.load_shadow_state_dict(files[2])
    _step(fresh, fresh_adam, clip_grad_norm_global_, 4, dp_rank())
    _step(reference, ref_adam, _reference_clip, 4, None)
    torch.testing.assert_close(fresh(x), reference(x), atol=1e-5, rtol=1e-4)
    with io.swap_in(fresh, fresh_ema.module):
        torch.testing.assert_close(fresh(x), ref_ema.module(x), atol=1e-5, rtol=1e-4)

    other = copy.deepcopy(reference)  # an unsplit checkpoint loads into FSDP x tp
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.25)
    io.load_model_state(model, other.state_dict())
    torch.testing.assert_close(model(x), other(x), atol=1e-5, rtol=1e-4)
    reloaded = io.model_state(model)
    if rank == 0:
        _assert_state_matches(reloaded, other.state_dict())


def _scenario_planted(rank: int) -> None:
    from spectramr.core.parallel_groups import dp_rank
    from spectramr.infrastructure.distributed import fsdp_wrap
    from spectramr.infrastructure.distributed import tensor_parallel as tp_pkg
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy

    ctx = _context("full_shard", 4)
    # 1. Skipping the tensor-group norm reduction makes the run diverge from one process.
    model, reference, ema, ref_ema, _ = _build(ctx)
    _train(model, ema, _naive_clip, dp_rank())
    _train(reference, ref_ema, _reference_clip, None)
    state = FSDPStateIO().model_state(model)
    if rank == 0:
        _assert_state_matches(state, reference.state_dict(), equal=False)

    # 2. Without the local-state view the EMA blend writes into gathered copies: frozen.
    model, _, ema, _, _ = _build(ctx)
    before = {k: v.clone() for k, v in ema.shadow_state_dict().items()}
    model(_batch(0, dp_rank())).square().mean().backward()
    torch.optim.SGD(model.parameters(), lr=0.5).step()
    original = tp_pkg.tensor_parallel_local_state
    tp_pkg.tensor_parallel_local_state = contextlib.nullcontext
    try:
        ema.update(model)
    finally:
        tp_pkg.tensor_parallel_local_state = original
    after = ema.shadow_state_dict()
    split = getattr(model, tp_pkg.TP_PLAN_ATTR).sharded
    if rank == 0:
        assert all(torch.equal(after[k], before[k]) for k in split), "split weights moved"
        assert not all(torch.equal(after[k], v) for k, v in before.items())

    # 3. A shard mesh that spans the tensor dim is refused before any shard.
    real = fsdp_wrap.fsdp_shard_mesh
    fsdp_wrap.fsdp_shard_mesh = lambda mesh, _s: mesh[("dp_shard", "tp")]
    try:
        with pytest.raises(RuntimeError, match="must shard over"):
            FSDPStrategy().prepare_models({"generator": _seeded()}, ctx)
    finally:
        fsdp_wrap.fsdp_shard_mesh = real

    # 4. A unit inside a split block (size-based picks the Linear layers) is refused.
    small = _context("full_shard", 4, auto_wrap_policy="size_based", min_num_params=1)
    with pytest.raises(ValueError, match="inside tensor-parallel blocks"):
        FSDPStrategy().prepare_models({"generator": _seeded()}, small)

    # 5. Unverified combinations raise (mixed precision is verified:
    # test_fsdp_model_parallel_fp16_gloo.py).
    with pytest.raises(RuntimeError, match=r"optimization\.compile"):
        FSDPStrategy().prepare_models(
            {"generator": _seeded()}, _replace_config(ctx, _config(compile_=True))
        )
    with pytest.raises(RuntimeError, match=r"fsdp\.cpu_offload"):
        FSDPStrategy().prepare_models(
            {"generator": _seeded()}, _context("full_shard", 4, cpu_offload=True)
        )


def _seeded():
    torch.manual_seed(0)
    return _TinyDiT()


def _replace_config(ctx, config):
    import dataclasses

    return dataclasses.replace(ctx, config=config)


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


@pytest.mark.parametrize(("sharding", "local_world"), [("full_shard", 4), ("hybrid_shard", 2)])
def test_fsdp_tensor_parallel_matches_one_process(sharding: str, local_world: int) -> None:
    _spawn("parity", sharding, local_world)


def test_fsdp_tensor_parallel_planted_violations_fail() -> None:
    _spawn("planted")
