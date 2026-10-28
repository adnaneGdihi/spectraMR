"""IB-VF's critics and navigator encoder stay identical on two DDP ranks (#2629).

The full VF step needs a config-built digital-twin simulator, so the strategy is a shell
carrying only what its owned path reads. ``_register_ib_modules`` registers the three
modules, and two steps then run on rank-dependent data through ``_ib_features`` and
``_ib_term``, with the twin replaced by a fixed corruption. Two steps, because DDP reports
an unfinished reduction only on the next forward. On the base branch the three modules are
bare groups on ``opt_g``, so the ranks diverge.

VF's own conditioner is refused under ddp instead (see the class comment): wrapped, its
``null_token`` gets no gradient, and the second step fails DDP's reduction check.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2
_PARALLEL = SimpleNamespace(
    strategy="ddp", find_unused_parameters=False, gradient_as_bucket_view=False, static_graph=False
)


def _ib_shell():
    from spectramr.infrastructure.training.strategies.ib_vf_strategy import IBVFTrainingStrategy
    from spectramr.models.encoders.nav_encoder_1d import NavEncoder1D
    from spectramr.models.losses.infonce_critic import InfoNCECritic

    s = object.__new__(IBVFTrainingStrategy)
    s._owned_modules, s._owned_wrappers = {}, {}
    s.config = SimpleNamespace(parallel=_PARALLEL)
    s.device = torch.device("cpu")
    s.env = SimpleNamespace(opt_g=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1))
    s.beta_plus, s.beta_minus = 1.0, 0.1
    bank = {"negative_pool": "momentum_bank", "bank_size": 8}
    s.critic_max = InfoNCECritic(critic_hidden_dim=16, feature_dim=8, **bank)
    s.critic_min = InfoNCECritic(critic_hidden_dim=16, feature_dim=8, **bank)
    s.nav_encoder = NavEncoder1D(in_channels=2, nav_latent_dim=8, depth=2)
    s.simulator = lambda x: (x * 0.9 + 0.05, x, x)  # (corrupted, marker prior, clean)
    return s


def _run(rank: int, store: str, out_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.manual_seed(100 + rank)  # production seeds seed + rank
        s = _ib_shell()
        s._register_ib_modules()
        for _ in range(2):
            target = torch.complex(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)) + rank
            loss = s._ib_term(*s._ib_features(target))
            s.env.opt_g.zero_grad(set_to_none=True)
            loss.backward()
            s.env.opt_g.step()
        owned = (s.critic_max, s.critic_min, s.nav_encoder)
        flat = torch.cat([p.detach().reshape(-1) for m in owned for p in m.parameters()])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save([t.clone() for t in peers], os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_ib_vf_owned_modules_are_identical_across_ranks() -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_run, args=(os.path.join(tmp, "store"), tmp), nprocs=WORLD, join=True)
        peers = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.allclose(peers[0], peers[1]), (peers[0] - peers[1]).abs().max()
