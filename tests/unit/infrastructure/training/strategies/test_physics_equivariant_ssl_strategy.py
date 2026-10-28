"""``PhysicsEquivariantSSLStrategy``'s operator-draw stream (#2786) and its
forward-sized projection across a resume (#2796).

The stream is seeded per rank, so data-parallel ranks draw different operators,
and it is carried through ``strategy_state_dict``, so a resumed run continues the
sequence instead of replaying it from step 0. The strategy is built through its
real ``__init__`` over a minimal config, with the base constructor stubbed so no
model is built.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies import rng_streams
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.physics_equivariant_ssl_strategy import (
    PhysicsEquivariantSSLStrategy,
)


def _on_rank(
    monkeypatch, rank: int | None, *, encoder: nn.Module | None = None
) -> PhysicsEquivariantSSLStrategy:
    config = SimpleNamespace(
        training=SimpleNamespace(physics_eq_ssl={"latent_dim": 4}),
        run=SimpleNamespace(seed=11),
    )
    monkeypatch.setattr(rng_streams, "distributed_rank", lambda: rank)
    opt_g = torch.optim.Adam(encoder.parameters(), lr=1e-2) if encoder is not None else None

    def bare_init(self, env=None, **kwargs):
        self.config = config
        self.env = SimpleNamespace(generator=encoder, discriminator=None, losses={}, opt_g=opt_g)
        self.device = torch.device("cpu")

    monkeypatch.setattr(BaseTrainingStrategy, "__init__", bare_init)
    return PhysicsEquivariantSSLStrategy(env=None)


def _draws(generator: torch.Generator, pool: int, n: int = 16) -> list[int]:
    return [int(torch.randint(0, pool, (1,), generator=generator)) for _ in range(n)]


def _operator_draws(strategy: PhysicsEquivariantSSLStrategy) -> list[int]:
    return _draws(strategy._rng, len(strategy.operator_pool))


def test_two_ranks_draw_different_operators(monkeypatch) -> None:
    rank0, rank1 = _on_rank(monkeypatch, 0), _on_rank(monkeypatch, 1)
    assert _operator_draws(rank0) != _operator_draws(rank1)


def test_a_single_process_run_keeps_the_run_seed(monkeypatch) -> None:
    strategy = _on_rank(monkeypatch, None)
    expected = _draws(torch.Generator().manual_seed(11), len(strategy.operator_pool))
    assert _operator_draws(strategy) == expected


def test_a_resume_continues_the_operator_stream(monkeypatch) -> None:
    saved = _on_rank(monkeypatch, None)
    _operator_draws(saved)  # advance past the seed, so a replay cannot pass
    state = saved.strategy_state_dict()
    expected = _operator_draws(saved)

    resumed = _on_rank(monkeypatch, None)
    resumed.load_strategy_state_dict(state)
    assert _operator_draws(resumed) == expected


def test_the_step_draws_its_operator_without_item() -> None:
    assert ".item()" not in inspect.getsource(PhysicsEquivariantSSLStrategy._compute_losses_impl)


# ── the forward-sized projection survives a resume (#2796) ──────────────────


def _wide_encoder() -> nn.Module:
    # Flattened output is 2 x 3 x 3 = 18 features, not latent_dim = 4.
    torch.manual_seed(0)
    return nn.Conv2d(1, 2, kernel_size=3, padding=1)


def _trained(monkeypatch) -> PhysicsEquivariantSSLStrategy:
    strategy = _on_rank(monkeypatch, None, encoder=_wide_encoder())
    losses = strategy._compute_losses_impl(torch.randn(4, 1, 3, 3), None, epoch=0)
    strategy.env.opt_g.zero_grad(set_to_none=True)
    losses["g_total_loss"].backward()
    strategy.env.opt_g.step()
    return strategy


def _resume(monkeypatch, saved: PhysicsEquivariantSSLStrategy) -> PhysicsEquivariantSSLStrategy:
    """The order ``checkpointing.resume`` uses: strategy state, then optimizer."""
    strategy_state = saved.strategy_state_dict()
    optimizer_state = saved.env.opt_g.state_dict()
    resumed = _on_rank(monkeypatch, None, encoder=_wide_encoder())
    resumed.load_strategy_state_dict(strategy_state)
    resumed.env.opt_g.load_state_dict(optimizer_state)
    return resumed


def test_a_resume_restores_the_saved_projection(monkeypatch) -> None:
    saved = _trained(monkeypatch)
    resumed = _resume(monkeypatch, saved)

    assert torch.equal(resumed._projection.weight, saved._projection.weight)
    assert torch.equal(resumed._projection.bias, saved._projection.bias)


def test_a_resume_binds_the_projection_s_optimizer_state(monkeypatch) -> None:
    saved = _trained(monkeypatch)
    resumed = _resume(monkeypatch, saved)

    groups = resumed.env.opt_g.param_groups
    assert len(groups) == len(saved.env.opt_g.param_groups)
    assert groups[-1]["params"] == list(resumed._projection.parameters())
    saved_moment = saved.env.opt_g.state[saved._projection.weight]["exp_avg"]
    resumed_moment = resumed.env.opt_g.state[resumed._projection.weight]["exp_avg"]
    assert torch.equal(resumed_moment, saved_moment)


def test_the_resumed_step_keeps_the_restored_projection(monkeypatch) -> None:
    resumed = _resume(monkeypatch, _trained(monkeypatch))
    projection = resumed._projection

    resumed._compute_losses_impl(torch.randn(4, 1, 3, 3), None, epoch=1)
    assert resumed._projection is projection


def test_a_projection_of_the_wrong_width_is_refused(monkeypatch) -> None:
    state = _trained(monkeypatch).strategy_state_dict()
    state["modules"]["_projection"] = nn.Linear(18, 5).state_dict()

    fresh = _on_rank(monkeypatch, None, encoder=_wide_encoder())
    with pytest.raises(ValueError, match="latent_dim"):
        fresh.load_strategy_state_dict(state)


def test_a_saved_entry_without_a_target_raises(monkeypatch) -> None:
    state = _on_rank(monkeypatch, None).strategy_state_dict()
    state["modules"]["_renamed_head"] = nn.Linear(2, 2).state_dict()

    fresh = _on_rank(monkeypatch, None)
    with pytest.raises(RuntimeError, match="_renamed_head"):
        fresh.load_strategy_state_dict(state)
