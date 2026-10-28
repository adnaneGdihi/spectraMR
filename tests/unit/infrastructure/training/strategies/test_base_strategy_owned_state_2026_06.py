"""Tests for the BaseTrainingStrategy "strategy-owned state" checkpoint seam
(2026-06, section R).

A strategy is not an ``nn.Module`` but may own learnable modules / parameters
registered on ``opt_g``. ``strategy_state_dict`` / ``load_strategy_state_dict``
round-trip exactly those — and NOT anything owned by the environment
(generator / discriminator / losses), which is persisted through its own path.

The methods touch no construction state, so they are exercised on a bare
instance built via ``object.__new__`` with a duck-typed ``env`` (the existing
strategy-unit-test pattern).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    """Concrete subclass so ``object.__new__`` is not blocked by any abstract
    method; the seam under test lives entirely on the base."""

    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


def _bare(env) -> _Strat:
    s = object.__new__(_Strat)
    s.env = env
    return s


def _env(generator: nn.Module | None = None, losses=None):
    return SimpleNamespace(
        generator=generator,
        discriminator=None,
        losses=losses if losses is not None else {},
    )


def test_fresh_owned_module_is_captured() -> None:
    gen = nn.Linear(4, 4)
    s = _bare(_env(generator=gen))
    s.sfc = nn.Linear(3, 3)  # strategy-owned: not reachable from env

    state = s.strategy_state_dict()
    assert "sfc" in state["modules"]
    # The captured state-dict is the module's own weights.
    assert set(state["modules"]["sfc"]) == set(s.sfc.state_dict())


def test_env_owned_modules_are_excluded() -> None:
    gen = nn.Linear(4, 4)
    disc = nn.Linear(4, 1)
    loss_mod = nn.Linear(2, 2)
    s = _bare(SimpleNamespace(generator=gen, discriminator=disc, losses={"l1": loss_mod}))
    # Aliases of env-owned modules must NOT be re-saved.
    s.aliased_generator = gen
    s.aliased_disc = disc
    s.aliased_loss = loss_mod

    state = s.strategy_state_dict()
    assert state["modules"] == {}
    assert state["params"] == {}


def test_bare_parameter_round_trips() -> None:
    gen = nn.Linear(4, 4)
    s = _bare(_env(generator=gen))
    s.diffusion = nn.Parameter(torch.tensor(0.05))

    state = s.strategy_state_dict()
    assert "diffusion" in state["params"]
    assert torch.allclose(state["params"]["diffusion"], torch.tensor(0.05))

    # Restore onto a fresh strategy whose parameter starts at a different value.
    s2 = _bare(_env(generator=nn.Linear(4, 4)))
    s2.diffusion = nn.Parameter(torch.tensor(0.99))
    s2.load_strategy_state_dict(state)
    assert torch.allclose(s2.diffusion.detach(), torch.tensor(0.05))
    # copy_ must keep the SAME Parameter object (so opt_g's reference stays valid).
    assert isinstance(s2.diffusion, nn.Parameter)


def test_module_weights_round_trip() -> None:
    gen = nn.Linear(4, 4)
    s = _bare(_env(generator=gen))
    torch.manual_seed(0)
    s.sfc = nn.Linear(3, 3)
    # Mutate the owned module so its weights are non-default.
    with torch.no_grad():
        s.sfc.weight.add_(1.234)
    saved = s.strategy_state_dict()

    # Fresh strategy with a freshly-initialised sfc (different weights).
    s2 = _bare(_env(generator=nn.Linear(4, 4)))
    torch.manual_seed(1)
    s2.sfc = nn.Linear(3, 3)
    assert not torch.allclose(s2.sfc.weight, s.sfc.weight)

    s2.load_strategy_state_dict(saved)
    assert torch.allclose(s2.sfc.weight, s.sfc.weight)
    assert torch.allclose(s2.sfc.bias, s.sfc.bias)


def test_empty_and_none_state_is_noop() -> None:
    s = _bare(_env(generator=nn.Linear(4, 4)))
    s.sfc = nn.Linear(3, 3)
    before = s.sfc.weight.clone()
    s.load_strategy_state_dict(None)
    s.load_strategy_state_dict({})
    assert torch.allclose(s.sfc.weight, before)


def test_mocked_env_with_no_owned_modules_is_empty() -> None:
    # A MagicMock env (common in unit tests) is not an nn.Module, so the
    # env-owned set is empty and discovery must not crash. With no owned
    # modules on the strategy, the result is empty.
    s = _bare(MagicMock())
    s.some_scalar = 0.5
    s.some_flag = True
    state = s.strategy_state_dict()
    assert state == {"modules": {}, "params": {}, "generators": {}, "counters": {}}


def test_owned_module_alongside_env_aliases() -> None:
    # Mixed case: a real owned module IS captured even when env aliases are
    # also present on the strategy.
    gen = nn.Linear(4, 4)
    s = _bare(_env(generator=gen))
    s.generator_alias = gen          # excluded
    s.critic = nn.Linear(5, 1)       # owned → captured
    state = s.strategy_state_dict()
    assert "critic" in state["modules"]
    assert "generator_alias" not in state["modules"]


# ---------------------------------------------------------------------------
# #2786: declared random streams round-trip through the base seam
# ---------------------------------------------------------------------------


class _Streamed(_Strat):
    checkpointed_streams = ("_rng",)


def _streamed() -> _Streamed:
    s = object.__new__(_Streamed)
    s.env = _env()
    s._rng = torch.Generator().manual_seed(4)
    return s


def test_a_declared_stream_resumes_where_it_was_saved() -> None:
    saved = _streamed()
    torch.rand(8, generator=saved._rng)  # advance past the seed
    state = saved.strategy_state_dict()
    expected = torch.rand(8, generator=saved._rng)

    resumed = _streamed()
    resumed.load_strategy_state_dict(state)
    assert torch.equal(torch.rand(8, generator=resumed._rng), expected)


def test_an_undeclared_generator_is_not_saved() -> None:
    # A stream reseeded on every call, or built lazily after the first
    # validation (se3 navigator's), would make a resume of a fresh strategy
    # refuse a stream it does not yet own.
    s = _bare(_env())
    s._val_motion_generator = torch.Generator()
    assert s.strategy_state_dict()["generators"] == {}


# ---------------------------------------------------------------------------
# #2820: a saved entry with no live target of its kind raises
# ---------------------------------------------------------------------------


def _saved_with_sfc_and_scale() -> dict:
    s = _bare(_env(generator=nn.Linear(4, 4)))
    s.sfc = nn.Linear(3, 3)
    s.scale = nn.Parameter(torch.tensor(2.0))
    return s.strategy_state_dict()


def test_a_saved_module_with_no_attribute_raises() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.scale = nn.Parameter(torch.tensor(0.0))
    with pytest.raises(RuntimeError, match=r"modules\.sfc"):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())


def test_a_saved_module_whose_lazy_slot_is_still_none_raises() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.sfc = None
    fresh.scale = nn.Parameter(torch.tensor(0.0))
    with pytest.raises(RuntimeError, match=r"modules\.sfc"):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())


def test_a_saved_parameter_with_no_attribute_raises() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.sfc = nn.Linear(3, 3)
    with pytest.raises(RuntimeError, match=r"params\.scale"):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())


def test_a_target_of_the_wrong_kind_raises() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.sfc = nn.Linear(3, 3)
    fresh.scale = torch.tensor(0.0)  # a plain tensor, not an nn.Parameter
    with pytest.raises(RuntimeError, match=r"params\.scale"):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())


def test_a_parameter_whose_lazy_slot_is_still_none_raises() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.sfc = nn.Linear(3, 3)
    fresh.scale = None
    with pytest.raises(RuntimeError, match=r"params\.scale"):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())


def test_a_module_target_of_the_wrong_kind_raises() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.sfc = torch.zeros(3, 3)  # a tensor where a module was saved
    fresh.scale = nn.Parameter(torch.tensor(0.0))
    with pytest.raises(RuntimeError, match=r"modules\.sfc"):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())


def test_a_refused_restore_leaves_every_target_untouched() -> None:
    fresh = _bare(_env(generator=nn.Linear(4, 4)))
    fresh.scale = nn.Parameter(torch.tensor(0.0))
    with pytest.raises(RuntimeError):
        fresh.load_strategy_state_dict(_saved_with_sfc_and_scale())
    assert float(fresh.scale.detach()) == 0.0


class _Lazy(_Strat):
    def _materialize_owned_state(self, state: dict) -> None:
        if "sfc" in state["modules"]:
            self.sfc = nn.Linear(3, 3)


def test_the_materialize_hook_rebuilds_a_lazy_module_before_the_check() -> None:
    saved = _bare(_env(generator=nn.Linear(4, 4)))
    saved.sfc = nn.Linear(3, 3)
    state = saved.strategy_state_dict()

    fresh = object.__new__(_Lazy)
    fresh.env = _env(generator=nn.Linear(4, 4))
    fresh.load_strategy_state_dict(state)
    assert torch.equal(fresh.sfc.weight, saved.sfc.weight)


class _Counted(_Strat):
    checkpointed_counters = ("_steps_seen",)


def test_declared_counters_round_trip_and_old_checkpoints_still_load() -> None:
    # #2892: a schedule counter must survive resume, and a checkpoint written
    # before counters existed has no "counters" key and must still load.
    s = object.__new__(_Counted)
    s.env = _env(generator=nn.Linear(2, 2))
    s._steps_seen = 37
    state = s.strategy_state_dict()
    assert state["counters"] == {"_steps_seen": 37}

    fresh = object.__new__(_Counted)
    fresh.env = _env(generator=nn.Linear(2, 2))
    fresh._steps_seen = 0
    fresh.load_strategy_state_dict(state)
    assert fresh._steps_seen == 37

    legacy = {k: v for k, v in state.items() if k != "counters"}
    fresh.load_strategy_state_dict(legacy)
    assert fresh._steps_seen == 37
