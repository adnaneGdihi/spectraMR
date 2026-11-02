"""FSDP2 fp16 parameters over tensor- and context-parallel blocks, dp=2 x {tp, cp}=2 (gloo).

``MixedPrecisionPolicy(param_dtype=fp16, reduce_dtype=fp32)`` through the production path
(``FSDPStrategy.prepare_models`` / ``adopt``), stepped by ``AMPPolicy`` with the scaler the
training loop builds, under fp16 autocast. The split collectives carry fp16, the
equivalence gates ran in float32 before the shard, the checkpoint is the float32 master
weights with the unsplit keys, and two clipped steps stay within fp16 error of float32.
"""

from __future__ import annotations

import copy
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

WORLD, MAX_NORM = 4, 0.05


def _ctx(axis: str, precision: str = "fp16"):
    from spectramr.config.schemas.base import ParallelismConfigSchema
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh
    from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

    fsdp = {
        "enabled": True,
        "auto_wrap_policy": "transformer_block",
        "transformer_layer_cls": ["DiTBlock", "UViTBlock"],
        "mixed_precision": precision,
    }
    parallel = ParallelismConfigSchema(strategy="fsdp", fsdp=fsdp, mesh={axis: 2})
    config = SimpleNamespace(
        optimization=SimpleNamespace(
            precision=SimpleNamespace(enabled=False, dtype=None),
            compile=SimpleNamespace(enabled=False),
        )
    )
    mesh = build_parallel_mesh(parallel, device_type="cpu", local_world_size=WORLD)
    return ParallelContext(config=config, device=torch.device("cpu"), parallel=parallel, mesh=mesh)


def _model_and_batch(axis: str, step: int, shard: int | None):
    gen = torch.Generator().manual_seed(100 + step)
    if axis == "tensor":
        from tests.unit.infrastructure.distributed.tensor_parallel.test_tensor_parallel_gloo import (
            WIDTH,
            _TinyDiT,
        )

        torch.manual_seed(0)
        x = (torch.randn(4, 6, WIDTH, generator=gen),)
        make = _TinyDiT
    else:
        from tests.unit.infrastructure.distributed.context_parallel.test_context_parallel_gloo import (
            seeded,
        )

        x = (torch.randn(4, 2, 8, 8, generator=gen), torch.rand(4, generator=gen))
        make = seeded
    if shard is not None:
        x = tuple(t[shard * 2 : shard * 2 + 2] for t in x)
    return make, x


def _scenario_fp16(rank: int, axis: str) -> None:
    import torch.distributed as dist

    from spectramr.core.parallel_groups import dp_rank
    from spectramr.infrastructure.distributed.context_parallel import attention
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy
    from spectramr.infrastructure.distributed.tensor_parallel import layers
    from spectramr.infrastructure.training.mixed_precision import (
        ComplexGradScaler,
        needs_cross_rank_inf_flag,
    )
    from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy

    ctx = _ctx(axis)
    make, _ = _model_and_batch(axis, 0, None)
    model = make()
    reference = copy.deepcopy(model)
    models = FSDPStrategy().prepare_models({"generator": model}, ctx)
    wrapped = FSDPStrategy().adopt(models, {}, {}, ctx).models["generator"]

    # The dtype the split collective carries: the K/V all-gather, the tp output reduce.
    owner, name = (
        (attention, "gather_keys_values")
        if axis == "context"
        else (layers, "reduce_from_tensor_parallel")
    )
    gathered: list[torch.dtype] = []
    real = getattr(owner, name)

    def recording(x, group):
        gathered.append(x.dtype)
        return real(x, group)

    setattr(owner, name, recording)
    scaler = ComplexGradScaler("cpu", enabled=True, init_scale=1024.0, sharded=True)
    assert needs_cross_rank_inf_flag(ctx.parallel)
    policy = AMPPolicy(max_grad_norm=MAX_NORM)
    sgd = torch.optim.SGD(wrapped.parameters(), lr=0.5)
    ref_sgd = torch.optim.SGD(reference.parameters(), lr=0.5)
    try:
        for step in range(2):
            # Autocast as the training loop runs it: base.py turns AMP on for FSDP fp16.
            with torch.amp.autocast("cpu", dtype=torch.float16):
                out = wrapped(*_model_and_batch(axis, step, dp_rank())[1])
            loss = out.float().square().mean()
            policy.backward_and_step(
                loss, sgd, wrapped, "g", 0, "test", scaler=scaler,
                gradient_clipping_fn=lambda m: clip_grad_norm_global_(m, MAX_NORM),
            )  # fmt: skip
            reference(*_model_and_batch(axis, step, None)[1]).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_NORM)
            ref_sgd.step()
            ref_sgd.zero_grad(set_to_none=True)
    finally:
        setattr(owner, name, real)
    assert gathered and set(gathered) == {torch.float16}, gathered
    assert scaler.get_scale() == 1024.0, "a clean fp16 run skipped a step"

    state = FSDPStateIO().model_state(wrapped)
    if rank == 0:
        expected = reference.state_dict()
        assert sorted(state) == sorted(expected), "checkpoint keys differ from the unsplit model"
        floats = {k: v.dtype for k, v in state.items() if v.is_floating_point()}
        assert set(floats.values()) == {torch.float32}, "the checkpoint is not the fp32 master"
        moved = max(float((state[k] - v).abs().max()) for k, v in expected.items())
        assert moved < 2e-3, f"fp16 steps drift {moved:.1e} from float32"
        unsplit = make()
        unsplit.load_state_dict(state, strict=True)
    dist.barrier()


def _worker(rank: int, store: str, scenario: str, args: tuple) -> None:
    import torch.distributed as dist

    os.environ.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.set_num_threads(1)
        globals()[f"_scenario_{scenario}"](rank, *args)
    finally:
        from spectramr.core.parallel_groups import reset_dp_layout

        reset_dp_layout()
        dist.destroy_process_group()


@pytest.mark.parametrize("axis", ["tensor", "context"])
def test_fsdp_fp16_params_over_model_parallel_blocks(axis: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        store = os.path.join(tmp, "store")
        mp.spawn(_worker, args=(store, "fp16", (axis,)), nprocs=WORLD, join=True)
