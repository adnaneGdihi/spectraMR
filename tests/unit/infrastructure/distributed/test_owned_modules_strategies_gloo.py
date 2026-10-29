"""Owned heads of migrated strategies stay identical across two DDP ranks (#2629).

Each strategy is built through its real ``__init__`` on both ranks with the same seed and
then takes one step on rank-dependent data, through its own loss path. The owned module
must be DDP-wrapped and reached through ``call_owned``, so its gradients are averaged and
the two copies match. On the base branch the head is a bare parameter group, so the ranks
drift apart (and ``owned_modules()`` is empty, which fails earlier).
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
    strategy="ddp",
    find_unused_parameters=False,
    gradient_as_bucket_view=False,
    static_graph=False,
)


def _generator() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
    )


def _config() -> MagicMock:
    config = MagicMock()
    config.device = "cpu"
    config.parallel = _PARALLEL
    config.optimization.optimizer.learning_rate = 1e-2
    config.optimization.gradient.clip.enabled = False
    config.optimization.precision.dtype = "float32"
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    # A MagicMock attribute is not a declared mode; StepExecutor refuses it.
    config.optimization.gradient.activation_offload = "none"
    config.physics = None
    return config


def _build(which: str):
    """The strategy as its own ``__init__`` builds it, with a mock environment."""
    from tests.utils.mock_environment import create_mock_training_env

    config = _config()
    model = _generator()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    if which == "teichmuller":
        from spectramr.config.schemas.training.teichmuller import (
            TeichmullerTrainingConfigSchema,
        )
        from spectramr.infrastructure.training.strategies.teichmuller_cold_diffusion_strategy import (
            TeichmullerColdDiffusionStrategy as Cls,
        )

        config.training = MagicMock()
        config.training.teichmuller = TeichmullerTrainingConfigSchema(spatial_size=8, n_steps=6)
        env = create_mock_training_env(
            config=config, device="cpu", generator=model, opt_g=optimizer
        )
        return Cls(env), "schedule_head", {}
    if which == "cycle_bloch":
        from spectramr.infrastructure.training.strategies.cycle_bloch_strategy import (
            CycleBlochStrategy as Cls,
        )

        config.physics = MagicMock(field_strength=0.064)
        config.losses.gan = None
        config.training = SimpleNamespace(
            seed=0, cycle_bloch=None, model_fields_set=set(), model_extra={}
        )
        disc = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        env = create_mock_training_env(
            config=config,
            device="cpu",
            generator=model,
            discriminator=disc,
            opt_g=optimizer,
            opt_d=torch.optim.SGD(disc.parameters(), lr=1e-2),
            model_type="gan",
        )
        return Cls(env), "estimator", {}
    if which == "beltrami":
        from spectramr.infrastructure.training.strategies.beltrami_motion_cortical_strategies import (
            BeltramiMotionCorrectionStrategy as Cls,
        )

        config.training = SimpleNamespace(seed=0, beltrami_motion_correction=None)
        env = create_mock_training_env(
            config=config, device="cpu", generator=model, opt_g=optimizer
        )
        return Cls(env), "mu_head", {}
    from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
        AdaptiveSFCHSSCStrategy as Cls,
    )

    config.training = SimpleNamespace(seed=0, adaptive_sfc_hssc=None)
    env = create_mock_training_env(config=config, device="cpu", generator=model, opt_g=optimizer)
    return (
        Cls(env),
        "sfc",
        {
            "batch": {
                "mask": torch.ones(2, 1, 32, 32),
                "conformal_jacobian": torch.ones(2, 1, 32, 32),
            }
        },
    )


def _bootstrap_container(log_dir: str) -> None:
    """The DI container the strategy base reads its logger from; conftest does this per test."""
    from spectramr.domain.interfaces.service_interfaces import ILoggingService
    from spectramr.infrastructure.di.di_container import init_container
    from tests.utils.logging_service import make_logging_service

    container = init_container()
    if not container.has_service(ILoggingService):
        container.register(ILoggingService, make_logging_service(log_dir))


def _step_once(rank: int, store: str, which: str, out_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        _bootstrap_container(out_dir)
        torch.manual_seed(0)  # identical initialisation on both ranks
        strategy, name, extra = _build(which)
        torch.manual_seed(100 + rank)  # rank-dependent data
        x = torch.randn(2, 1, 32, 32) + rank
        if which == "teichmuller":
            loss = strategy._compute_losses_impl(input_batch=None, target_batch=x)["g_total_loss"]
        elif which == "cycle_bloch":  # the generator closure is the real training path
            x = x.abs()
            steps = strategy.train_step({"input": x}, epoch=0, input_batch=x, target_batch=x)
            loss = steps[0]["closure"]()
        else:
            loss = strategy._compute_losses_impl(x, x, epoch=0, **extra)["g_total_loss"]
        loss.backward()
        strategy.env.opt_g.step()
        owned = getattr(strategy, name)  # the attribute, so a base-branch run fails on divergence
        flat = torch.cat([p.detach().reshape(-1) for p in owned.parameters()])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save([t.clone() for t in peers], os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("which", ["teichmuller", "beltrami", "sfc", "cycle_bloch"])
def test_owned_head_is_identical_across_ranks_after_one_step(which: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _step_once,
            args=(os.path.join(tmp, "store"), which, tmp),
            nprocs=WORLD,
            join=True,
        )
        peers = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.allclose(peers[0], peers[1]), (which, (peers[0] - peers[1]).abs().max())
