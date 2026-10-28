"""``learnable_flip_angle`` must reach an optimizer.

The Bloch layer's flip angle becomes an ``nn.Parameter`` under ``learnable_flip_angle``,
but nothing handed it to an optimizer, so an arm that advertised a learned flip angle
trained with it frozen at the default.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)
from tests.unit.infrastructure.training.strategies.test_disentangled_strategy_optional_disc import (
    DummyEnvironment,
    patch_loss_builder,  # noqa: F401  (fixture)
    patch_loss_computer,  # noqa: F401  (fixture)
)


def _build(*, learnable: bool, prefill: bool = False):
    env = DummyEnvironment(with_disc=False, enable_adv=False)
    env.config.model.model_kwargs = {"learnable_flip_angle": learnable}
    env.opt_g = torch.optim.Adam(env.generator.parameters(), lr=1e-4)
    strategy = DisentangledTrainingStrategy(
        env=env, metrics_service=MagicMock(), checkpoint_service=MagicMock()
    )
    return strategy, env.opt_g


def _group(optimizer, name):
    return next((g for g in optimizer.param_groups if g.get("name") == name), None)


@pytest.mark.usefixtures("patch_loss_computer", "patch_loss_builder")
def test_a_learnable_flip_angle_is_given_to_the_generator_optimizer() -> None:
    strategy, opt_g = _build(learnable=True)
    group = _group(opt_g, "bloch_flip_angle")
    assert group is not None, "the flip angle is a parameter no optimizer owns"
    assert any(p is strategy._bloch_layer.flip_angle for p in group["params"])
    assert group["lr"] == 1e-3


@pytest.mark.usefixtures("patch_loss_computer", "patch_loss_builder")
def test_a_fixed_flip_angle_adds_no_group() -> None:
    """Without ``learnable_flip_angle`` the angle is a buffer: nothing to train."""
    strategy, opt_g = _build(learnable=False)
    assert _group(opt_g, "bloch_flip_angle") is None
    assert not any(p.requires_grad for p in strategy._bloch_layer.parameters())


@pytest.mark.usefixtures("patch_loss_computer", "patch_loss_builder")
def test_the_flip_angle_actually_moves_under_an_optimizer_step() -> None:
    """The group is wired to the live parameter, not a copy."""
    strategy, opt_g = _build(learnable=True)
    before = strategy._bloch_layer.flip_angle.detach().clone()
    (strategy._bloch_layer.flip_angle * 3.0).backward()
    opt_g.step()
    assert not torch.equal(strategy._bloch_layer.flip_angle.detach(), before)


@pytest.mark.usefixtures("patch_loss_computer", "patch_loss_builder")
def test_the_setup_runs_once_so_no_orphan_group_is_left() -> None:
    """Planted: the base ``__init__`` already runs the setup hook, and a second explicit
    call built a second loss computer and Bloch layer, leaving the first copies' groups
    on the optimizer with nothing ever flowing into them."""
    strategy, opt_g = _build(learnable=True)
    flip_groups = [g for g in opt_g.param_groups if g.get("name") == "bloch_flip_angle"]
    assert len(flip_groups) == 1
    assert flip_groups[0]["params"][0] is strategy._bloch_layer.flip_angle
