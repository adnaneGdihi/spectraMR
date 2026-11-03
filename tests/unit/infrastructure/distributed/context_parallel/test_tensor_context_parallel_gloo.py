"""tp=2 x cp=2 on four gloo ranks, through the production path, against one process.

``build_parallel_mesh`` -> ``DDPStrategy`` / ``FSDPStrategy.prepare_models`` -> ``adopt``:
the tensor split and its gate run first, then the context split gates the tensor-split
backbone and adopts its attentions as ``TensorContextParallelAttention``. A few clipped
SGD steps must match one process; the checkpoint and optimizer state must be unsplit.
Each planted violation (keys and values gathered over the tensor group, gradients
averaged over tp, a clip that double-counts cp or skips tp, a shard mesh across tp)
must turn the run red.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from tests.unit.infrastructure.distributed.context_parallel.test_context_parallel_gloo import (  # noqa: E402
    seeded,
    tiny_dit,
    tiny_uvit,
)
from tests.unit.infrastructure.distributed.context_parallel.test_context_parallel_mesh_gloo import (  # noqa: E402
    MAX_NORM,
    _batch,
    _config,
    _prepare,
    _reference_clip,
    _train,
)

pytestmark = [pytest.mark.unit]


def _context(strategy: str):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    fsdp = {}
    if strategy == "fsdp":
        fsdp = {
            "fsdp": {
                "enabled": True,
                "sharding_strategy": "full_shard",
                "auto_wrap_policy": "transformer_block",
                "transformer_layer_cls": ["DiTBlock", "UViTBlock"],
            }
        }
    parallel = ParallelismConfigSchema(strategy=strategy, mesh={"tensor": 2, "context": 2}, **fsdp)
    mesh = build_parallel_mesh(parallel, device_type="cpu", local_world_size=4)
    return ParallelContext(
        config=_config(), device=torch.device("cpu"), parallel=parallel, mesh=mesh
    )


def _plain(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _check_optimizer_state(wrapped, reference, io, side: int) -> None:
    """Momentum buffers save in the unsplit shapes and load back into each rank's slice.

    The passthrough file keys state by index, FSDP's by parameter name; rank 0 writes it.
    """
    opts = [torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9) for m in (wrapped, reference)]
    for module, opt in zip((wrapped, reference), opts, strict=True):
        module(*_batch(7, side, None)).square().mean().backward()
        opt.step()
    files = [io.optimizer_state(wrapped, opts[0])]
    torch.distributed.broadcast_object_list(files, src=0)
    expected = {n: opts[1].state[p]["momentum_buffer"] for n, p in reference.named_parameters()}
    names = list(expected)
    assert len(files[0]["state"]) == len(expected)
    for key, entry in files[0]["state"].items():
        name = key if isinstance(key, str) else names[key]
        torch.testing.assert_close(entry["momentum_buffer"], expected[name])
    live = [_plain(opts[0].state[p]["momentum_buffer"]).clone() for p in wrapped.parameters()]
    io.load_optimizer_state(wrapped, opts[0], files[0])
    for param, buffer in zip(wrapped.parameters(), live, strict=True):
        torch.testing.assert_close(_plain(opts[0].state[param]["momentum_buffer"]), buffer)


def _check_parity(rank, wrapped, reference, emas, make, side, io) -> list[float]:
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_

    ema, ref_ema = emas
    # dp=1: every rank takes the whole batch; tp and cp peers share it.
    norms = _train(wrapped, clip_grad_norm_global_, side, None, ema)
    ref_norms = _train(reference, _reference_clip, side, None, ref_ema)
    assert min(ref_norms) > MAX_NORM, "clipping must bind or the norm is untested"
    torch.testing.assert_close(torch.tensor(norms), torch.tensor(ref_norms), atol=1e-5, rtol=1e-4)
    state = io.model_state(wrapped)
    if rank == 0:
        expected = reference.state_dict()
        assert {k: v.shape for k, v in state.items()} == {k: v.shape for k, v in expected.items()}
        for key, value in expected.items():
            torch.testing.assert_close(state[key], value, atol=1e-5, rtol=1e-4)
        unsplit = seeded(make)
        unsplit.load_state_dict(state, strict=True)  # a tp x cp checkpoint loads unsplit
        x, t = _batch(9, side, None)
        torch.testing.assert_close(unsplit(x, t), reference(x, t), atol=1e-5, rtol=1e-4)
    x, t = _batch(9, side, None)
    with torch.no_grad():  # the validation forward and the EMA swap, on every rank
        torch.testing.assert_close(wrapped(x, t), reference(x, t), atol=1e-5, rtol=1e-4)
        with io.swap_in(wrapped, ema.module):
            torch.testing.assert_close(wrapped(x, t), ref_ema.module(x, t), atol=1e-5, rtol=1e-4)
    _check_optimizer_state(wrapped, reference, io, side)
    return norms


def _check_record(record, strategy: str, attentions: int) -> None:
    assert record["tensor_parallel"]["order"] == f"tp_then_cp_then_{strategy}"
    cp = record["context_parallel"]
    assert cp["attention_layers"] == cp["tensor_split_attention_layers"] == attentions


def _scenario_ddp(rank: int) -> None:
    from spectramr.infrastructure.distributed.state_io import PassthroughStateIO
    from spectramr.infrastructure.distributed.strategies import DDPStrategy
    from spectramr.infrastructure.training.strategies.rng_streams import stream_rank

    ctx = _context("ddp")
    assert stream_rank() == 0, "tp and cp peers of the one replica share its random stream"
    for make, side, attentions in ((tiny_dit, 8, 2), (tiny_uvit, 6, 3)):
        wrapped, reference, emas, record = _prepare(DDPStrategy(), make, ctx)
        assert wrapped.process_group.size() == 2, "DDP must average over dp x cp, never tp"
        _check_record(record, "ddp", attentions)
        _check_parity(rank, wrapped, reference, emas, make, side, PassthroughStateIO())


def _scenario_fsdp(rank: int) -> None:
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy

    ctx = _context("fsdp")
    wrapped, reference, emas, record = _prepare(FSDPStrategy(), tiny_dit, ctx)
    _check_record(record, "fsdp", 2)
    mesh = next(wrapped.parameters()).device_mesh
    assert mesh.size() == 2 and "tp" not in mesh.mesh_dim_names, "FSDP shards over dp x cp"
    _check_parity(rank, wrapped, reference, emas, tiny_dit, 8, FSDPStateIO())


def _diverges(norms: list[float], ref_norms: list[float]) -> bool:
    return not torch.allclose(torch.tensor(norms), torch.tensor(ref_norms), atol=1e-5, rtol=1e-4)


def _scenario_planted(rank: int) -> None:
    from spectramr.core.module_utils import unwrap_model
    from spectramr.infrastructure.distributed import fsdp_wrap, strategies
    from spectramr.infrastructure.distributed.context_parallel import ContextParallelError
    from spectramr.infrastructure.distributed.context_parallel.attention import (
        CP_ADOPTERS,
        TensorContextParallelAttention,
    )
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_
    from spectramr.infrastructure.distributed.tensor_parallel import TP_PLAN_ATTR
    from spectramr.infrastructure.distributed.tensor_parallel.layers import (
        TensorParallelAttention,
    )

    ctx = _context("ddp")
    # 1. Keys and values gathered over the tensor group: the context gate must raise.
    real = CP_ADOPTERS[TensorParallelAttention]
    CP_ADOPTERS[TensorParallelAttention] = lambda m, _g: TensorContextParallelAttention.adopt(
        m, m.tp_group
    )
    try:
        with pytest.raises(ContextParallelError, match="does not reproduce"):
            strategies.DDPStrategy().prepare_models({"generator": seeded()}, ctx)
    finally:
        CP_ADOPTERS[TensorParallelAttention] = real

    # 2. Gradients averaged over dp x cp x tp mix different slices: the run diverges.
    original = strategies._grad_process_group
    strategies._grad_process_group = lambda _ctx: torch.distributed.group.WORLD
    try:
        wrapped, reference, _, _ = _prepare(strategies.DDPStrategy(), tiny_dit, ctx)
    finally:
        strategies._grad_process_group = original
    _train(wrapped, _reference_clip, 8, None)
    _train(reference, _reference_clip, 8, None)
    state = wrapped.module.state_dict()
    assert not all(
        torch.allclose(state[k], v, atol=1e-5, rtol=1e-4) for k, v in reference.state_dict().items()
    )

    # 3. A clip that sums the split norms over tp x cp counts every cp peer twice;
    # 4. one that skips the tensor reduction clips each rank by its partial norm.
    ref_norms = _train(seeded(), _reference_clip, 8, None)
    wrapped = _prepare(strategies.DDPStrategy(), tiny_dit, ctx)[0]
    root = unwrap_model(wrapped)
    plan = getattr(root, TP_PLAN_ATTR)
    setattr(root, TP_PLAN_ATTR, dataclasses.replace(plan, group=torch.distributed.group.WORLD))
    assert _diverges(_train(wrapped, clip_grad_norm_global_, 8, None), ref_norms)
    wrapped = _prepare(strategies.DDPStrategy(), tiny_dit, ctx)[0]
    assert _diverges(_train(wrapped, _reference_clip, 8, None), ref_norms)

    # 5. A shard mesh that spans the tensor dim is refused before any shard.
    real_mesh = fsdp_wrap.fsdp_shard_mesh
    fsdp_wrap.fsdp_shard_mesh = lambda mesh, _s: mesh[("cp", "tp")]
    try:
        with pytest.raises(RuntimeError, match="must shard over"):
            strategies.FSDPStrategy().prepare_models({"generator": seeded()}, _context("fsdp"))
    finally:
        fsdp_wrap.fsdp_shard_mesh = real_mesh


def _worker(rank: int, world: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"](rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["ddp", "fsdp", "planted"])
def test_tensor_context_parallel_four_ranks(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(4, os.path.join(tmp, "store"), scenario), nprocs=4, join=True)
