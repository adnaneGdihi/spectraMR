"""FSDP2 sharding, state IO, clipping and EMA against a real two-rank group.

FSDP2 runs on CPU over gloo, so this runs in the CPU lane (#2695: the FSDP1
version needed an accelerator and never ran there). Models are sharded through
``maybe_wrap_with_fsdp``, the production path, and every assertion compares the
sharded run against the same computation on one process.
"""

from __future__ import annotations

import copy
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

WORLD = 2


def _cfg(**fsdp):
    base = {
        "enabled": True,
        "sharding_strategy": "full_shard",
        "auto_wrap_policy": "size_based",
        "transformer_layer_cls": [],
        "min_num_params": 1,
        "cpu_offload": False,
        "mixed_precision": None,
        "reduce_dtype": "float32",
    }
    base.update(fsdp)
    return SimpleNamespace(fsdp=SimpleNamespace(**base))


def _net(seed: int):
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def _shard(net, **fsdp):
    from spectramr.infrastructure.distributed.fsdp_wrap import maybe_wrap_with_fsdp

    return maybe_wrap_with_fsdp(net, _cfg(**fsdp), None, torch.device("cpu"))


def _batch():
    torch.manual_seed(1)
    return torch.randn(4, 8)


def _step(model, opt) -> None:
    model(_batch()).square().sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        globals()[f"_scenario_{scenario}"](rank)
    finally:
        dist.destroy_process_group()


def _scenario_save_shapes(rank: int) -> None:
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO

    io = FSDPStateIO()
    model = _shard(_net(0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    _step(model, opt)
    state, opt_state = io.model_state(model), io.optimizer_state(model, opt)
    raw = model.state_dict()["0.weight"]
    # The reason every save goes through state_io: the raw dict is this rank's shard.
    assert tuple(raw.to_local().shape) == (4, 8)
    if rank == 0:
        assert tuple(state["0.weight"].shape) == (8, 8)
        assert tuple(state["2.weight"].shape) == (4, 8)
        full = sum(v["exp_avg"].numel() for v in opt_state["state"].values())
        assert full == 8 * 8 + 8 + 4 * 8 + 4
    else:
        assert state == {}


def _scenario_round_trip(rank: int) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.state_io import FSDPStateIO

    io = FSDPStateIO()
    model = _shard(_net(0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    _step(model, opt)
    box = [(io.model_state(model), io.optimizer_state(model, opt))]
    dist.broadcast_object_list(box, src=0)  # stands in for every rank reading the file
    state, opt_state = box[0]

    fresh = _shard(_net(123))
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-2)
    _step(fresh, fresh_opt)
    io.load_model_state(fresh, state)
    io.load_optimizer_state(fresh, fresh_opt, opt_state)
    again, again_opt = io.model_state(fresh), io.optimizer_state(fresh, fresh_opt)
    if rank == 0:
        for key, value in state.items():
            torch.testing.assert_close(again[key], value)
        for key, entry in opt_state["state"].items():
            torch.testing.assert_close(again_opt["state"][key]["exp_avg"], entry["exp_avg"])


def _scenario_matches_one_process(rank: int) -> None:
    """Clip norm and post-step weights equal the unsharded single-process run."""
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    ref = _net(0)
    ref(_batch()).square().sum().backward()
    ref_norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.5)
    torch.optim.SGD(ref.parameters(), lr=0.1).step()

    model = _shard(_net(0))
    model(_batch()).square().sum().backward()
    norm = BaseTrainingStrategy._compute_gradient_norm(None, model, True, 0.5)
    assert type(norm) is torch.Tensor, "callers must get a plain tensor, not a DTensor"
    torch.testing.assert_close(norm, ref_norm)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    state = FSDPStateIO().model_state(model)
    if rank == 0:
        for key, value in ref.state_dict().items():
            torch.testing.assert_close(state[key], value)


def _scenario_ema(rank: int) -> None:
    """The production order: shadow deep-copied BEFORE the shard, then sharded alike."""
    import torch.distributed as dist

    from spectramr.core.module_utils import unwrap_model
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO
    from spectramr.infrastructure.distributed.strategies import FSDPStrategy
    from spectramr.infrastructure.optimization.ema import ModelEma

    io = FSDPStateIO()
    bare = _net(0)
    ema = ModelEma(bare, decay=0.5, device="cpu")
    model = _shard(bare)
    FSDPStrategy().prepare_ema(ema, {"generator": model}, None)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    _step(model, opt)

    ema.update(unwrap_model(model))
    live_before = io.model_state(model)
    shadow = ema.shadow_state_dict()
    with io.swap_in(model, ema.module):
        swapped = io.model_state(model)
    live_after = io.model_state(model)

    if rank == 0:
        expected = copy.deepcopy(_net(0)).state_dict()  # the shadow's starting weights
        for key, value in expected.items():
            blended = torch.lerp(value, live_before[key], 1.0 - ema._current_decay())
            torch.testing.assert_close(shadow[key], blended)
            torch.testing.assert_close(swapped[key], shadow[key])
            torch.testing.assert_close(live_after[key], live_before[key])
        assert any(not torch.equal(swapped[k], live_before[k]) for k in expected)

    # The checkpoint form loads back into a sharded shadow.
    box = [shadow]
    dist.broadcast_object_list(box, src=0)
    other = ModelEma(_net(7), decay=0.5, device="cpu")
    FSDPStrategy().prepare_ema(other, {"generator": model}, None)
    other.load_shadow_state_dict(box[0])
    reloaded = other.shadow_state_dict()
    if rank == 0:
        for key, value in shadow.items():
            torch.testing.assert_close(reloaded[key], value)


def _scenario_unsharded_shadow_refused(rank: int) -> None:
    from spectramr.infrastructure.distributed.state_io import FSDPStateIO

    model = _shard(_net(0))
    with (
        pytest.raises(RuntimeError, match="sharded like the live model"),
        FSDPStateIO().swap_in(model, _net(0)),
    ):
        pass


def _scenario_complex_parameters(rank: int) -> None:
    from torch import nn

    class _Complex(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.w = nn.Parameter(torch.randn(8, 8, dtype=torch.complex64) * 0.1)

        def forward(self, x):
            return x @ self.w

    model = _shard(_Complex(), auto_wrap_policy="none")
    model(torch.randn(4, 8, dtype=torch.complex64)).abs().sum().backward()
    assert model.w.grad.dtype == torch.complex64
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm.full_tensor())


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


@pytest.mark.parametrize(
    "scenario",
    [
        "save_shapes",
        "round_trip",
        "matches_one_process",
        "ema",
        "unsharded_shadow_refused",
        "complex_parameters",
    ],
)
def test_fsdp2_two_ranks(scenario: str) -> None:
    _spawn(scenario)
