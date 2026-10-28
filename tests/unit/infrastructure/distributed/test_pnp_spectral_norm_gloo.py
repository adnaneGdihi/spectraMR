"""PnP's spectral-norm vectors are identical on two DDP ranks (#2629).

PnP reparameterises the env generator with ``spectral_norm`` after it is DDP-wrapped, so
the power-iteration vectors ``weight_u`` / ``weight_v`` are buffers DDP never saw and does
not broadcast. Production seeds each rank differently (``seed + rank``), so each rank
starts from its own vectors, estimates a different spectral norm, and runs a different
denoiser. Here the generator is wrapped first, the strategy is built through its real
``__init__`` under rank-dependent seeds, and two RED steps run on rank-dependent data. The
vectors and the generator weights must then match across ranks.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2
_PARALLEL = SimpleNamespace(
    strategy="ddp", find_unused_parameters=False, gradient_as_bucket_view=False, static_graph=False
)


def _config() -> MagicMock:
    config = MagicMock()
    config.device = "cpu"
    config.parallel = _PARALLEL
    config.physics = None
    config.optimization.optimizer.learning_rate = 1e-2
    config.optimization.gradient.clip.enabled = False
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    config.optimization.precision.dtype = "float32"
    config.losses = SimpleNamespace(reconstruction=SimpleNamespace(lambda_red=0.5))
    config.training = SimpleNamespace(
        seed=0,
        model_fields_set=set(),
        model_extra={},
        pnp=SimpleNamespace(enabled=True, spectral_norm_iters=1, denoiser_lipschitz_constant=1.0),
    )
    return config


def _stub_recon_loss() -> None:
    """Recon loss = L1(generator(input), target); the RED term on top is PnP's own."""
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    def _loss(self, input_batch=None, target_batch=None, epoch=0, **_kw):
        pred = self.env.generator(input_batch)
        return {"g_total_loss": (pred - target_batch).abs().mean(), "prediction": pred}

    ReconstructionTrainingStrategy._compute_losses_impl = _loss


def _bootstrap_container(log_dir: str) -> None:
    from spectramr.domain.interfaces.service_interfaces import ILoggingService
    from spectramr.infrastructure.di.di_container import init_container
    from tests.utils.logging_service import make_logging_service

    container = init_container()
    if not container.has_service(ILoggingService):
        container.register(ILoggingService, make_logging_service(log_dir))


def _run(rank: int, store: str, out_dir: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.strategies import wrap_with_ddp
    from spectramr.infrastructure.training.strategies.pnp_strategy import PnPStrategy
    from tests.utils.mock_environment import create_mock_training_env

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        _bootstrap_container(out_dir)
        _stub_recon_loss()
        torch.manual_seed(0)
        inner = torch.nn.Sequential(
            torch.nn.Conv2d(2, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 2, kernel_size=3, padding=1),
        )
        gen = wrap_with_ddp(inner, _PARALLEL, torch.device("cpu"))  # as production: wrap first
        config = _config()
        opt = torch.optim.SGD(gen.parameters(), lr=1e-2)
        env = create_mock_training_env(config=config, device="cpu", generator=gen, opt_g=opt)
        torch.manual_seed(100 + rank)  # production seeds seed + rank
        strategy = PnPStrategy(env)
        for _ in range(2):
            x = torch.randn(2, 2, 16, 16) + rank
            loss = strategy._compute_losses_impl(x, x.flip(-1), epoch=0)["g_total_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        state = list(inner.parameters()) + [
            b for n, b in inner.named_buffers() if n.endswith(("weight_u", "weight_v"))
        ]
        flat = torch.cat([t.detach().reshape(-1) for t in state])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save([t.clone() for t in peers], os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_spectral_norm_vectors_and_weights_match_across_ranks() -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_run, args=(os.path.join(tmp, "store"), tmp), nprocs=WORLD, join=True)
        peers = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.allclose(peers[0], peers[1]), (peers[0] - peers[1]).abs().max()
