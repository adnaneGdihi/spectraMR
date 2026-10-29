"""Bare parameters a strategy owns stay identical across two DDP ranks (#2629).

Each strategy is built through its real ``__init__`` on both ranks with the same seed and
takes one step on rank-dependent data through its own loss path. Its owned parameter is
held by an ``OwnedParameter`` registered with ``register_owned_module`` and read through
``call_owned``, so DDP averages its gradient. On the base branch the parameter is a bare
group on ``opt_g``, so the two ranks drift apart.

LOUPE and PILOT reach the generator through ``ReconstructionTrainingStrategy``, whose
loss is replaced here by an L1 of the generator output; the owned path (mask sample,
density penalty, NUFFT sampling, feasibility penalty) is the strategy's own.
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


class _DictGenerator(torch.nn.Module):
    """Two-contrast input to a ``{"t1", "t2"}`` map dict, as multi_param expects."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.conv(x)
        return {"t1": y[:, :1], "t2": y[:, 1:]}


def _config() -> MagicMock:
    config = MagicMock()
    config.device = "cpu"
    config.parallel = _PARALLEL
    config.physics = None
    config.optimization.optimizer.learning_rate = 1e-2
    config.optimization.gradient.clip.enabled = False
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    # A MagicMock attribute is not a declared mode; StepExecutor refuses it.
    config.optimization.gradient.activation_offload = "none"
    config.optimization.precision.dtype = "float32"
    config.training = SimpleNamespace(seed=0, model_fields_set=set(), model_extra={})
    config.model.model_kwargs = {"n_shots": 2, "samples_per_shot": 16}
    return config


def _stub_recon_loss() -> None:
    """Recon loss = L1(generator(input), target); the owned path stays the strategy's."""
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    def _loss(self, input_batch=None, target_batch=None, epoch=0, **_kw):
        pred = self.env.generator(input_batch)
        return {"g_total_loss": (pred - target_batch).abs().mean()}

    ReconstructionTrainingStrategy._compute_losses_impl = _loss


def _build(which: str, config: MagicMock):
    from spectramr.config.schemas.training.base import MultiParameterTrainingConfigSchema
    from tests.utils.mock_environment import create_mock_training_env

    if which in ("multi_param", "spin_sde"):
        config.training.multi_parameter = MultiParameterTrainingConfigSchema(
            parameters=["t1", "t2"], uncertainty_weighting=True, bloch_consistency_lambda=0.0
        )
        config.training.spin_sde = None
        model = _DictGenerator()
    else:
        model = torch.nn.Conv2d(2, 2, kernel_size=3, padding=1)
    if which == "loupe":
        config.undersampling.adaptive = SimpleNamespace(
            enabled=True,
            target_rate=0.25,
            density_penalty_lambda=1.0,
            budget="soft",
            mask_learning_rate=1e-2,
            annealing=SimpleNamespace(tau_init=1.0, tau_final=0.5, anneal_steps=10),
        )
    if which == "pilot":
        from spectramr.config.schemas.acquisition import CodesignConfig

        config.acquisition.codesign = CodesignConfig(enabled=True, field_of_view_m=0.2)
        config.undersampling.adaptive.mask_learning_rate = 1e-2
        config.data.sampling.patch_size = [8, 8]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    env = create_mock_training_env(config=config, device="cpu", generator=model, opt_g=optimizer)
    if which == "multi_param":
        from spectramr.infrastructure.training.strategies.multi_param_mapping_strategy import (
            OneShotMultiParameterStrategy as Cls,
        )
    elif which == "spin_sde":
        from spectramr.infrastructure.training.strategies.spin_sde_strategy import (
            SpinSDEStrategy as Cls,
        )
    elif which == "loupe":
        from spectramr.infrastructure.training.strategies.loupe_strategy import (
            LOUPEStrategy as Cls,
        )
    else:
        from spectramr.infrastructure.training.strategies.pilot_strategy import (
            PILOTStrategy as Cls,
        )
    return Cls(env)


_OWNED = {
    "multi_param": ("_log_sigmas",),
    "spin_sde": ("_log_sigmas", "diffusion"),
    "loupe": ("_mask_logits",),
    "pilot": ("_trajectory",),
}


def _losses(which: str, strategy, rank: int) -> torch.Tensor:
    x = torch.randn(2, 2, 8, 8) + rank
    if which in ("multi_param", "spin_sde"):
        batch = {
            "observed_contrasts": x,
            "t1": torch.rand(2, 1, 8, 8) + rank,
            "t2": torch.rand(2, 1, 8, 8),
            "T1_map": torch.rand(2, 1, 8, 8) + 0.1,
            "T2_map": torch.rand(2, 1, 8, 8) + 0.05,
            "M0_map": torch.rand(2, 1, 8, 8) + rank,
            "multi_contrast_signal": torch.rand(2, 2, 8, 8),
        }
        out = strategy._compute_losses_impl(x, x, 0, batch=batch)
        return out["loss_total"]
    out = strategy._compute_losses_impl(input_batch=x, target_batch=x * 0.5, epoch=0)
    return out["g_total_loss"]


def _bootstrap_container(log_dir: str) -> None:
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
        _stub_recon_loss()
        torch.manual_seed(0)  # identical initialisation on both ranks
        strategy = _build(which, _config())
        torch.manual_seed(100 + rank)  # rank-dependent data and noise
        _losses(which, strategy, rank).backward()
        strategy.env.opt_g.step()
        # Read the attributes, so a base-branch run fails on divergence, not on a lookup.
        flat = torch.cat([getattr(strategy, n).detach().reshape(-1) for n in _OWNED[which]])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save([t.clone() for t in peers], os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("which", sorted(_OWNED))
def test_owned_parameter_is_identical_across_ranks_after_one_step(which: str) -> None:
    import torch.multiprocessing as mp

    if which == "pilot":
        pytest.importorskip("torchkbnufft")
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_step_once, args=(os.path.join(tmp, "store"), which, tmp), nprocs=WORLD, join=True)
        peers = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.allclose(peers[0], peers[1]), (which, (peers[0] - peers[1]).abs().max())
