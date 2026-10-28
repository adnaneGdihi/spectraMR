"""The fp32 DeepSpeed gradient check against the real engine, ZeRO stages 1-3.

Needs CUDA and the [deepspeed] extra; skipped otherwise. The planted violation is
a parameter whose loss term is ``sqrt(z) * 0`` with ``z = 0``: the loss stays
finite (0) while its gradient is ``0 * inf = NaN``, so only the gradient check can
catch it. Run on the GPU lane:

    pytest tests/unit/infrastructure/distributed/test_deepspeed_nonfinite_gpu.py -m gpu -v
"""

from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
deepspeed = pytest.importorskip("deepspeed")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


class _Trap(torch.nn.Module):
    def __init__(self, *, plant_nan: bool) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        self.zero = torch.nn.Parameter(torch.zeros(()))
        self.plant_nan = plant_nan

    def forward(self, x):
        loss = self.body(x).sum()
        if self.plant_nan:
            loss = loss + torch.sqrt(self.zero) * 0.0
        return loss


_ENV = {"MASTER_ADDR": "127.0.0.1", "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def single_rank_process_group():
    # Module scope: DeepSpeed caches its group handles, so re-initialising per test
    # fails with "Group ... is not registered" on every test after the first.
    saved = {k: os.environ.get(k) for k in _ENV}
    os.environ.update({**_ENV, "MASTER_PORT": str(_free_port())})
    try:
        deepspeed.init_distributed(dist_backend="nccl")
        yield
        torch.distributed.destroy_process_group()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _engine(stage: int, *, plant_nan: bool):
    from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
        initialize_deepspeed_engine,
    )

    model = _Trap(plant_nan=plant_nan).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": stage},
        "zero_allow_untested_optimizer": True,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
    }
    engine, ds_optimizer, _ = initialize_deepspeed_engine(
        model=model, optimizer=optimizer, lr_scheduler=None, config=config
    )
    return engine, ds_optimizer, model


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_planted_nan_gradient_raises_before_the_weights_move(stage, single_rank_process_group):
    from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
        DeepSpeedStepPolicy,
    )

    engine, ds_optimizer, model = _engine(stage, plant_nan=True)
    policy = DeepSpeedStepPolicy()
    policy.register(ds_optimizer, engine)
    x = torch.ones(2, 4, device="cuda")
    before = torch.cat([p.detach().flatten().float() for p in model.body.parameters()]).clone()
    # Micro-batch 1 is not a boundary (gas=2) and is not checked.
    policy.backward_and_step(engine.module(x), ds_optimizer)
    with pytest.raises(RuntimeError, match="fp32 DeepSpeed, ZeRO stage"):
        policy.backward_and_step(engine.module(x), ds_optimizer)
    after = torch.cat([p.detach().flatten().float() for p in model.body.parameters()])
    assert torch.equal(before, after), "weights moved on a NaN gradient"


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_finite_gradients_step_at_the_boundary(stage, single_rank_process_group):
    from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
        DeepSpeedStepPolicy,
    )

    engine, ds_optimizer, model = _engine(stage, plant_nan=False)
    policy = DeepSpeedStepPolicy()
    policy.register(ds_optimizer, engine)
    x = torch.ones(2, 4, device="cuda")
    before = torch.cat([p.detach().flatten().float() for p in model.body.parameters()]).clone()
    policy.backward_and_step(engine.module(x), ds_optimizer)
    policy.backward_and_step(engine.module(x), ds_optimizer)
    after = torch.cat([p.detach().flatten().float() for p in model.body.parameters()])
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after), "a finite boundary must apply the update"
