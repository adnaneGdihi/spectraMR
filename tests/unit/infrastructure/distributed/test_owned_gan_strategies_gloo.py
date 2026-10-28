"""GAN-family strategies train identical networks on two DDP ranks (#2629).

As in production, ``env.generator`` / ``env.discriminator`` are DDP-wrapped before the
strategy is built through its real ``__init__``. Each rank then runs two full iterations
of the strategy's own ``train_step`` closures (D then G) on rank-dependent data. Every
network the strategy trains, its own and the env's, must match across ranks. Two
iterations, because DDP reports an unfinished reduction only on the next forward.

On the base branch the strategy-built networks are bare (cyclegan, privileged_learning),
the wrapped env generator is mistaken for a foreign class and replaced by an unwrapped
copy (stargan_v2), or the encoder tap reads an attribute the wrapper hides (cut).
"""

from __future__ import annotations

import functools
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2
_PKG = "spectramr.infrastructure.training.strategies"
# Networks each strategy trains, read by attribute after the steps.
_TRAINED = {
    "cyclegan": ("gen_ab", "gen_ba", "disc_a", "disc_b"),
    "cut": ("generator", "discriminator", "nce_loss"),
    "stargan_v2": ("generator", "mapping", "style_encoder", "discriminator"),
    "privileged_learning": ("_domain_discriminator",),
}


def _settings(model_type: str, gan: dict | None):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 2},
        model={"model_type": model_type, "in_channels": 1, "out_channels": 1},
        optimization={"learning_rate": 1e-2},
        logging={},
        losses={"gan": gan} if gan is not None else {},
        parallel={"strategy": "ddp"},
    )


def _tiny_nets(which: str):
    """Small env networks, and registry patches so strategy-built nets stay small too."""
    from spectramr.models.discriminators.patchgan_discriminator import PatchGANDiscriminator
    from spectramr.models.discriminators.stargan_v2_discriminator import StarGANv2Discriminator
    from spectramr.models.generators.cycle_gan import ResNetGenerator
    from spectramr.models.generators.stargan_v2 import (
        MappingNetwork,
        StarGANv2Generator,
        StyleEncoder,
    )

    if which == "stargan_v2":
        import spectramr.infrastructure.training.strategies.stargan_v2_strategy as mod

        mod.MappingNetwork = functools.partial(MappingNetwork, hidden_dim=16)
        mod.StyleEncoder = functools.partial(StyleEncoder, base_channels=8, max_channels=16)
        gen = StarGANv2Generator(img_channels=1, style_dim=64, base_channels=8, max_channels=16)
        disc = StarGANv2Discriminator(img_channels=1, base_channels=8, max_channels=16)
        return gen, disc
    if which == "privileged_learning":
        return torch.nn.Conv2d(1, 1, kernel_size=3, padding=1), None
    import spectramr.infrastructure.training.strategies.cyclegan_strategy as mod

    small = {
        "cyclegan_generator": functools.partial(ResNetGenerator, ngf=8, n_blocks=1),
        "patch_gan": functools.partial(PatchGANDiscriminator, ndf=8, n_layers=2),
    }
    mod.get_model_class = small.__getitem__
    return small["cyclegan_generator"](1, 1), small["patch_gan"](in_channels=1)


def _build(which: str):
    """The strategy through its real ``__init__``, over DDP-wrapped env networks."""
    import importlib

    from spectramr.infrastructure.distributed.strategies import wrap_with_ddp
    from tests.utils.mock_environment import create_mock_training_env

    gan = {
        "cyclegan": {"lambda_cycle": 10.0, "lambda_identity": 0.5},
        "cut": {"lambda_nce": 1.0},
        "stargan_v2": {"lambda_style": 1.0, "lambda_diversity": 1.0, "lambda_cycle": 1.0},
        "privileged_learning": None,
    }[which]
    model_type = {"privileged_learning": "unet", "stargan_v2": "stargan_v2_generator"}.get(
        which, "cyclegan_generator"
    )
    config = _settings(model_type, gan)
    gen, disc = _tiny_nets(which)
    cpu = torch.device("cpu")
    gen = wrap_with_ddp(gen, config.parallel, cpu)
    disc = wrap_with_ddp(disc, config.parallel, cpu) if disc is not None else None
    env = create_mock_training_env(
        config=config,
        device="cpu",
        generator=gen,
        discriminator=disc,
        opt_g=torch.optim.SGD(gen.parameters(), lr=1e-2),
        opt_d=torch.optim.SGD(disc.parameters(), lr=1e-2) if disc is not None else None,
        model_type="gan" if disc is not None else "reconstruction",
    )
    cls_name = {
        "cyclegan": ("cyclegan_strategy", "CycleGANTrainingStrategy"),
        "cut": ("cut_strategy", "CUTTrainingStrategy"),
        "stargan_v2": ("stargan_v2_strategy", "StarGANv2TrainingStrategy"),
        "privileged_learning": ("privileged_learning_strategy", "PrivilegedLearningStrategy"),
    }[which]
    return getattr(importlib.import_module(f"{_PKG}.{cls_name[0]}"), cls_name[1])(env)


def _batch(rank: int) -> dict:
    torch.manual_seed(100 + rank)  # rank-dependent data
    return {
        "input": torch.rand(2, 1, 32, 32) + rank,
        "target": torch.rand(2, 1, 32, 32),
        "field_strength": torch.tensor([0.1, 1.5]),
        "field_strength_target": torch.tensor([3.0, 7.0]),
        "marker": torch.rand(2, 1, 32, 32),
        "domain_label": torch.tensor([0.0, 1.0]),
    }


def _iterate(strategy, which: str, batch: dict) -> None:
    """One trainer iteration: each step config's closure, backward and optimizer step."""
    steps = strategy.train_step(batch, 0, input_batch=batch["input"], target_batch=batch["target"])
    for step in steps:
        step["optimizer"].zero_grad(set_to_none=True)
        step["closure"]().backward()
        step["optimizer"].step()


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
        torch.manual_seed(0)  # identical initialisation on both ranks
        strategy = _build(which)
        for _ in range(2):
            _iterate(strategy, which, _batch(rank))
        flat = torch.cat(
            [
                p.detach().reshape(-1)
                for name in _TRAINED[which]
                for p in getattr(strategy, name).parameters()
            ]
        )
        peers = [torch.zeros_like(flat) for _ in range(WORLD)]
        dist.all_gather(peers, flat)
        torch.save([t.clone() for t in peers], os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("which", sorted(_TRAINED))
def test_gan_strategy_networks_are_identical_across_ranks(which: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_run, args=(os.path.join(tmp, "store"), which, tmp), nprocs=WORLD, join=True)
        peers = torch.load(os.path.join(tmp, "r0.pt"))
    assert torch.allclose(peers[0], peers[1]), (which, (peers[0] - peers[1]).abs().max())
