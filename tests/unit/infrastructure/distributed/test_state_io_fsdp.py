"""``FSDPStateIO`` against a real two-rank FSDP group.

FSDP1 needs an accelerator, so this runs only where CUDA is visible; both ranks
share the one device and talk over gloo, which is enough to shard. The planted
shapes are the ones the review found by reading and a probe then confirmed:
the unwrapped module's ``state_dict()`` is flat shard views, and a plain
``optimizer.state_dict()`` holds one rank's slice of the moments.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,  # selected by scripts/ci/run_gpu_tests.sbatch, excluded from the CPU lane
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP1 needs an accelerator"),
]

WORLD = 2


def _build(seed: int):
    import torch.distributed.fsdp as fsdp
    from torch import nn

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4)).cuda()
    model = fsdp.FullyShardedDataParallel(
        net, use_orig_params=True, device_id=torch.device("cuda", 0)
    )
    return model, torch.optim.Adam(model.parameters(), lr=1e-2)


def _train_one_step(model, opt) -> None:
    torch.manual_seed(1)
    model(torch.randn(2, 8).cuda()).sum().backward()
    opt.step()
    opt.zero_grad()


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.state_io import FSDPStateIO

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        io = FSDPStateIO()
        model, opt = _build(seed=0)
        _train_one_step(model, opt)
        # Both reads are collective: every rank enters, rank 0 receives.
        state = io.model_state(model)
        opt_state = io.optimizer_state(model, opt)
        if scenario == "save_shapes":
            _assert_save_shapes(rank, model, opt, state, opt_state)
        elif scenario == "round_trip":
            _assert_round_trip(io, state, opt_state)
        elif scenario == "ema":
            _assert_ema(io)
    finally:
        dist.destroy_process_group()


def _assert_save_shapes(rank, model, opt, state, opt_state) -> None:
    inner = model.module.state_dict()
    if rank == 0:
        assert tuple(state["0.weight"].shape) == (8, 8)
        assert tuple(state["1.weight"].shape) == (4, 8)
        # The reason the director must not hand the unwrapped module to the reader.
        assert tuple(inner["0.weight"].shape) != (8, 8)
        # Full Adam moments, not this rank's half of them.
        full = sum(v["exp_avg"].numel() for v in opt_state["state"].values())
        assert full == 8 * 8 + 8 + 4 * 8 + 4
        local = sum(v["exp_avg"].numel() for v in opt.state_dict()["state"].values())
        assert local < full
    else:
        assert state == {}


def _assert_round_trip(io, state, opt_state) -> None:
    import torch.distributed as dist

    # The real flow has every rank read the file; broadcasting stands in for it.
    box = [(state, opt_state)]
    dist.broadcast_object_list(box, src=0)
    state, opt_state = box[0]

    fresh, fresh_opt = _build(seed=123)  # different weights on purpose
    _train_one_step(fresh, fresh_opt)  # materialise optimizer state to load into
    io.load_model_state(fresh, state)
    io.load_optimizer_state(fresh, fresh_opt, opt_state)

    again = io.model_state(fresh)
    again_opt = io.optimizer_state(fresh, fresh_opt)
    if dist.get_rank() == 0:
        for key, value in state.items():
            assert torch.allclose(again[key], value), key
        for pid, entry in opt_state["state"].items():
            assert torch.allclose(again_opt["state"][pid]["exp_avg"], entry["exp_avg"]), pid


def _assert_ema(io) -> None:
    """The shadow is a bare deepcopy taken BEFORE the wrap, as in the real build."""
    import copy

    import torch.distributed as dist
    import torch.distributed.fsdp as fsdp
    from torch import nn

    from spectramr.core.module_utils import unwrap_model
    from spectramr.infrastructure.optimization.ema import ModelEma

    torch.manual_seed(0)
    bare = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4)).cuda()
    ema = ModelEma(copy.deepcopy(bare), decay=0.5)
    model = fsdp.FullyShardedDataParallel(
        bare, use_orig_params=True, device_id=torch.device("cuda", 0)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    _train_one_step(model, opt)

    # Without the unsharded params the blend sees flat shard views and either
    # raises on shape or blends nothing.
    with io.full_params(model):
        ema.update(unwrap_model(model))
    live_before = io.model_state(model)

    with io.swap_in(model, ema.module):
        swapped = io.model_state(model)
    live_after = io.model_state(model)

    if dist.get_rank() == 0:
        shadow = ema.module.state_dict()
        for key in shadow:
            assert torch.allclose(swapped[key].cpu(), shadow[key].cpu()), key
            assert torch.allclose(live_after[key].cpu(), live_before[key].cpu()), key
        # The swap changed something, or this proved nothing.
        assert any(not torch.allclose(swapped[k].cpu(), live_before[k].cpu()) for k in shadow)


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_root_gather_is_full_and_the_unwrapped_module_is_not() -> None:
    _spawn("save_shapes")


def test_model_and_optimizer_state_round_trip_across_ranks() -> None:
    _spawn("round_trip")


def test_ema_updates_and_swaps_against_sharded_weights() -> None:
    _spawn("ema")
