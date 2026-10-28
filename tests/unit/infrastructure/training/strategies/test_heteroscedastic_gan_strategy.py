"""HeteroscedasticGANStrategy: the critic and metrics see the mean; the objective adds the NLL."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.training.heteroscedastic_ulf import HeteroscedasticULFConfig
from spectramr.infrastructure.training.loop_state import LoopState
from spectramr.infrastructure.training.strategies.heteroscedastic_gan_strategy import (
    HeteroscedasticGANStrategy,
)
from spectramr.models.losses.heteroscedastic_losses import heteroscedastic_ulf_loss


class _Head(nn.Module):
    """``[mean, logvar]`` from a one-channel input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)

    def forward(self, x):
        return self.conv(x)


class _Computer:
    """Records what the loss computer is handed and returns a fixed adversarial term."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def _to_critic(self, x):
        self.seen.append(x.shape[1])
        return x

    def compute_discriminator_loss(self, real, fake, **_):
        self.seen.append(fake.shape[1])
        return types.SimpleNamespace(total=fake.mean(), components={})

    def compute_generator_loss(self, pred, target, **_):
        self.seen.append(pred.shape[1])
        return types.SimpleNamespace(total=pred.mean() * 0 + 1.0, components={})


def _strategy(generator: nn.Module) -> HeteroscedasticGANStrategy:
    s = object.__new__(HeteroscedasticGANStrategy)
    s.env = types.SimpleNamespace(
        generator=generator, discriminator=lambda x: x.mean(dim=(1, 2, 3))
    )
    s.loss_computer = _Computer()
    s.config = types.SimpleNamespace(
        training=types.SimpleNamespace(enforce_output_range=False),
        validation=types.SimpleNamespace(scoring=types.SimpleNamespace(enable_image_metrics=False)),
    )
    s._het_cfg = HeteroscedasticULFConfig(source_field=0.064, lambda_var_prior=0.1)
    s._het_sigma = torch.tensor([0.2])
    s._last_step_metrics = {}
    s.loop_state = LoopState(iteration=0)
    s._compute_training_metrics = lambda pred, **_: s.loss_computer.seen.append(pred.shape[1]) or {}
    s._log_validation_images_to_tensorboard = lambda **_: None
    s.__dict__["generator_model"] = generator
    return s


def test_the_critic_only_ever_sees_the_mean_channel() -> None:
    torch.manual_seed(0)
    s = _strategy(_Head())
    x, y = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    s._train_discriminator_step(x, y, s.env.discriminator, 0, 0)()
    s._train_generator_step(x, y, s.env.discriminator, 0, 0)()
    s.validation_step(x, y)
    assert s.loss_computer.seen and set(s.loss_computer.seen) == {1}


def test_the_generator_objective_adds_the_heteroscedastic_nll() -> None:
    torch.manual_seed(0)
    gen = _Head()
    s = _strategy(gen)
    x, y = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    g_total = s._train_generator_step(x, y, s.env.discriminator, 0, 0)()
    out = gen(x)
    expected = heteroscedastic_ulf_loss(
        out[:, :1], out[:, 1:], y, torch.tensor([0.2]), lambda_var_prior=0.1
    )["loss_total"]
    assert torch.allclose(g_total, 1.0 + expected)
    assert {"g_het_nll", "g_het_var_prior", "g_het_logvar_mean"} <= s._last_step_metrics.keys()
    g_total.backward()
    assert gen.conv.weight.grad[1].abs().sum() > 0, "the log-variance channel gets no gradient"


def test_the_sigma_anchor_is_moved_once_not_per_step() -> None:
    s = _strategy(_Head().double())
    x, y = torch.rand(2, 1, 8, 8, dtype=torch.float64), torch.rand(2, 1, 8, 8, dtype=torch.float64)
    s._train_generator_step(x, y, s.env.discriminator, 0, 0)()
    moved = s._het_sigma
    assert moved.dtype == torch.float64
    s._train_generator_step(x, y, s.env.discriminator, 0, 0)()
    assert s._het_sigma is moved


def test_a_one_channel_generator_is_refused() -> None:
    s = _strategy(nn.Conv2d(1, 1, 1))
    with pytest.raises(ValueError, match="2-channel"):
        s._train_generator_step(
            torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4), s.env.discriminator, 0, 0
        )()


@pytest.mark.parametrize(
    ("block", "match"),
    [(None, "needs training.heteroscedastic_ulf"), (HeteroscedasticULFConfig(), "source field")],
)
def test_setup_refuses_a_missing_block_or_source_field(monkeypatch, block, match) -> None:
    from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy

    monkeypatch.setattr(
        GANTrainingStrategy, "_setup_strategy_specific_components", lambda self: None
    )
    s = object.__new__(HeteroscedasticGANStrategy)
    s.config = types.SimpleNamespace(training=types.SimpleNamespace(heteroscedastic_ulf=block))
    with pytest.raises(Exception, match=match):
        s._setup_strategy_specific_components()
