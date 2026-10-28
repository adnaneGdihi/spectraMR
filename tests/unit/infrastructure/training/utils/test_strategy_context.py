"""``StrategyContext`` carries no diffusion horizon of its own (#2443).

Its ``num_timesteps = 1000`` field was set by nothing and read by nothing, and the
``DiffusionStrategyContext`` that did read it had no caller. A horizon here could
only disagree with ``training.diffusion.timesteps``, which the strategy samples from.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.utils import strategy_context as module  # noqa: E402


def test_the_factory_builds_a_context_on_the_requested_device():
    context = module.create_strategy_context(device="cpu", config=object())
    assert context.device == torch.device("cpu")


def test_the_context_declares_no_diffusion_horizon():
    """PLANTED: restoring the field reintroduces an unread 1000-step default."""
    fields = {f.name for f in dataclasses.fields(module.StrategyContext)}
    assert {"num_timesteps", "beta_schedule"}.isdisjoint(fields)


def test_a_horizon_passed_to_the_factory_is_refused():
    with pytest.raises(TypeError, match="num_timesteps"):
        module.create_strategy_context(device="cpu", config=object(), num_timesteps=29)
