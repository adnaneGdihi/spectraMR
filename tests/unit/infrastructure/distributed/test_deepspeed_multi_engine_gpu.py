"""Two DeepSpeed engines (generator + discriminator) reproduce a single-process GAN (#2631).

Each cycle runs the production order -- the discriminator step, then the
generator step with the critic frozen. After two cycles the discriminator's
weights on each rank must match a single-process run that averages the same
per-rank losses. A generator backward that leaked into the critic's reduce
buckets would show up as a drift at the second discriminator step.

GPU lane only (two CUDA devices and the ``[deepspeed]`` extra). From dev::

    pytest tests/unit/infrastructure/distributed/test_deepspeed_multi_engine_gpu.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
    deepspeed_available,
)

WORLD = 2
CYCLES = 2
LR = 1e-2

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
        reason="needs two CUDA devices",
    ),
    pytest.mark.skipif(not deepspeed_available(), reason="needs the [deepspeed] extra"),
]


def _nets() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(0)
    gen = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 4))
    torch.manual_seed(1)
    disc = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    return gen, disc


def _batch(cycle: int, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1000 * cycle + rank)
    x = torch.randn(1, 4, generator=g)
    y = torch.randn(1, 4, generator=g)
    return x.to(device), y.to(device)


def _worker(rank: int, store: str, stage: int, out_dir: str) -> None:
    import deepspeed

    from spectramr.infrastructure.training.strategies.mixins.adversarial import (
        set_critic_trainable,
    )

    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{store}", rank=rank, world_size=WORLD
    )
    try:
        gen, disc = _nets()
        gen, disc = gen.to(device), disc.to(device)
        cfg = {
            "train_batch_size": WORLD,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": stage},
            "steps_per_print": 1000,
        }
        g_eng, _, _, _ = deepspeed.initialize(
            model=gen, optimizer=torch.optim.Adam(gen.parameters(), lr=LR), config=cfg
        )
        d_eng, _, _, _ = deepspeed.initialize(
            model=disc, optimizer=torch.optim.Adam(disc.parameters(), lr=LR), config=cfg
        )
        for cycle in range(CYCLES):
            x, y = _batch(cycle, rank, device)
            # D step: critic trainable, fakes from the current generator.
            set_critic_trainable(d_eng, True)
            with torch.no_grad():
                fake = g_eng(x)
            d_loss = d_eng(y).mean() + d_eng(fake).mean()
            d_eng.backward(d_loss)
            d_eng.step()
            # G step: critic frozen, so the generator's objective must not reach D.grad.
            set_critic_trainable(d_eng, False)
            g_loss = d_eng(g_eng(x)).mean()
            g_eng.backward(g_loss)
            g_eng.step()
        state = {k: v.detach().cpu() for k, v in d_eng.module.state_dict().items()}
        torch.save(state, Path(out_dir) / f"rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def _single_process_reference() -> dict[str, torch.Tensor]:
    device = torch.device("cuda", 0)
    gen, disc = _nets()
    gen, disc = gen.to(device), disc.to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR)
    for cycle in range(CYCLES):
        batches = [_batch(cycle, r, device) for r in range(WORLD)]
        # Mean over ranks of each rank's loss is what DeepSpeed's gradient averaging computes.
        disc.requires_grad_(True)
        with torch.no_grad():
            fakes = [gen(x) for x, _ in batches]
        d_loss = (
            sum(disc(y).mean() + disc(f).mean() for (_, y), f in zip(batches, fakes, strict=True))
            / WORLD
        )
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()
        disc.requires_grad_(False)
        g_loss = sum(disc(gen(x)).mean() for x, _ in batches) / WORLD
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
    return {k: v.detach().cpu() for k, v in disc.state_dict().items()}


@pytest.mark.parametrize("stage", [0, 1, 2], ids=["zero0", "zero1", "zero2"])
def test_two_engines_match_single_process_critic(stage: int) -> None:
    mp = torch.multiprocessing
    with tempfile.TemporaryDirectory() as tmp:
        store = os.path.join(tmp, "store")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        mp.spawn(_worker, args=(store, stage, out_dir), nprocs=WORLD, join=True)
        ranks = [torch.load(Path(out_dir) / f"rank{r}.pt") for r in range(WORLD)]
    reference = _single_process_reference()
    for r, state in enumerate(ranks):
        assert state.keys() == reference.keys()
        for key, ref in reference.items():
            torch.testing.assert_close(
                state[key], ref, atol=1e-5, rtol=1e-4, msg=f"rank {r} param {key} at zero{stage}"
            )
