"""SSL / distillation owners stay identical across two DDP ranks (#2629).

As in production, the env generator is DDP-wrapped before the strategy is built through
its real ``__init__``. Each rank then runs two training steps through the strategy's own
loss path, on rank-dependent data and after rank-dependent seeding (production seeds
``seed + rank``). Every module the strategy trains or tracks must then match across
ranks, and a frozen copy of the generator must hold the inner module's checkpoint keys,
not the wrapper's ``module.``-prefixed ones.

On the base branch, JEPA's predictor and GeoMamba's uncertainty weights are bare groups
that diverge. The EMA copies are taken of the DDP wrapper, and a frozen teacher loaded
with ``policy="overlap"`` keeps each rank's own initialisation where the file is silent.
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
_PKG = "spectramr.infrastructure.training.strategies"
# (module, class, attributes compared across ranks, frozen copy whose keys are checked)
_CASES = {
    "jepa": ("jepa_strategy", "JEPAStrategy", ("_predictor", "_target_encoder"), "_target_encoder"),
    "multi_contrast": (
        "multi_contrast_contrastive_strategy",
        "MultiContrastContrastiveStrategy",
        ("_target_encoder",),
        "_target_encoder",
    ),
    "vf_consistency": (
        "vf_consistency_distillation_strategy",
        "VFConsistencyDistillationStrategy",
        ("_ema_teacher",),
        "_ema_teacher",
    ),
    "geomamba": ("geomamba_ulf_strategy", "GeoMambaULFStrategy", ("_uncertainty_loss",), None),
}


class _MaskedGenerator(torch.nn.Module):
    """A conv that takes GeoMamba's ``mask`` / ``contrast_id`` keywords."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x, mask=None, contrast_id=None):
        return self.conv(x)


def _config(which: str) -> MagicMock:
    config = MagicMock()
    config.device = "cpu"
    config.parallel = _PARALLEL
    config.physics = None
    config.optimization.optimizer.learning_rate = 1e-2
    config.optimization.gradient.clip.enabled = False
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    config.optimization.precision.dtype = "float32"
    training = SimpleNamespace(seed=0, model_fields_set=set(), model_extra={})
    if which == "multi_contrast":
        training.ssl = SimpleNamespace(
            multi_contrast_contrastive={"use_momentum": True, "momentum": 0.9, "warmup_steps": 1}
        )
    if which == "geomamba":
        training.geomamba_ulf = SimpleNamespace(
            film=SimpleNamespace(n_contrasts=2),
            uncertainty=SimpleNamespace(enabled=True, init_log_sigma=0.0, eps=1e-6),
            topology=None,
            metric_sfc=None,
        )
    config.training = training
    if which == "vf_consistency":  # the diffusion parent reads its own block
        config.training = MagicMock()
        config.training.vf_consistency_distillation = None
        config.training.diffusion = SimpleNamespace(
            num_timesteps=10, timesteps=10, noise_schedule="linear", beta_start=1e-4, beta_end=0.02
        )
        config.acceleration = None
        config.data.prior_loading.enabled = False
        config.model.model_type = "diffusion"
    return config


def _build(which: str):
    import importlib

    from spectramr.infrastructure.distributed.strategies import wrap_with_ddp
    from tests.utils.mock_environment import create_mock_training_env

    module_name, cls_name, _, _ = _CASES[which]
    mod = importlib.import_module(f"{_PKG}.{module_name}")
    if which == "geomamba":  # the l1 weight comes from losses.image_losses in production
        mod.declared_loss_weights = lambda _config: {"l1": 1.0}
        inner = _MaskedGenerator()
    elif which == "multi_contrast":
        inner = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 4))
    else:
        inner = torch.nn.Conv2d(1, 4 if which == "jepa" else 1, kernel_size=3, padding=1)
    gen = wrap_with_ddp(inner, _PARALLEL, torch.device("cpu"))
    opt = torch.optim.SGD(gen.parameters(), lr=1e-2)
    env = create_mock_training_env(config=_config(which), device="cpu", generator=gen, opt_g=opt)
    return getattr(mod, cls_name)(env)


def _loss(strategy, which: str, rank: int) -> torch.Tensor:
    x = torch.rand(2, 1, 8, 8) + rank
    if which == "jepa":
        return strategy._compute_losses_impl({"input": x})["loss_total"]
    if which == "multi_contrast":
        batch = {"view_a": x, "view_b": x.flip(-1)}
        return strategy._compute_losses_impl(x, None, 0, batch=batch)["loss_total"]
    if which == "vf_consistency":
        return strategy._compute_losses_impl(batch={"target": x})["loss_total"]
    batch = {"contrast_id": torch.tensor([0, 1])}
    return strategy._compute_losses_impl(x, x.flip(-1), 0, batch=batch)["g_total_loss"]


def _bootstrap_container(log_dir: str) -> None:
    from spectramr.domain.interfaces.service_interfaces import ILoggingService
    from spectramr.infrastructure.di.di_container import init_container
    from tests.utils.logging_service import make_logging_service

    container = init_container()
    if not container.has_service(ILoggingService):
        container.register(ILoggingService, make_logging_service(log_dir))


def _run(rank: int, store: str, which: str, out_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        _bootstrap_container(out_dir)
        torch.manual_seed(0)
        strategy = _build(which)
        torch.manual_seed(100 + rank)  # production seeds seed + rank
        for _ in range(2):
            loss = _loss(strategy, which, rank)
            strategy.env.opt_g.zero_grad(set_to_none=True)
            loss.backward()
            strategy.env.opt_g.step()
        _, _, attrs, frozen = _CASES[which]
        modules = [getattr(strategy, a) for a in attrs] + [strategy.env.generator]
        flat = torch.cat([p.detach().reshape(-1) for m in modules for p in m.parameters()])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        keys = sorted(getattr(strategy, frozen).state_dict()) if frozen else []
        torch.save(([t.clone() for t in peers], keys), os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("which", sorted(_CASES))
def test_owned_and_frozen_modules_are_identical_across_ranks(which: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_run, args=(os.path.join(tmp, "store"), which, tmp), nprocs=WORLD, join=True)
        peers, keys = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.allclose(peers[0], peers[1]), (which, (peers[0] - peers[1]).abs().max())
    assert not any(k.startswith("module.") for k in keys), (which, keys)


def _sync_teacher(rank: int, store: str, out_dir: str) -> None:
    import torch.distributed as dist

    from spectramr.infrastructure.distributed.owned_modules import sync_frozen_module

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        torch.manual_seed(100 + rank)  # the tensors a partial load leaves untouched
        teacher = torch.nn.BatchNorm2d(3)
        torch.nn.init.normal_(teacher.weight)
        teacher.running_mean.normal_()
        sync_frozen_module(teacher)
        flat = torch.cat([t.reshape(-1) for t in teacher.state_dict().values() if t.ndim])
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save(peers, os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_a_frozen_teacher_holds_rank_zeros_values_on_every_rank() -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_sync_teacher, args=(os.path.join(tmp, "store"), tmp), nprocs=WORLD, join=True)
        peers = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.equal(peers[0], peers[1])
