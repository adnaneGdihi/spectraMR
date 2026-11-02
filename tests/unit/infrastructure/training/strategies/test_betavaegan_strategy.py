"""Tests for ``BetaVAEGANStrategy`` — the β-TC term on the generator objective.

The GAN ``train_step`` runs ``_train_generator_step`` and never
``_compute_losses_impl``, so the term must arrive through
``_auxiliary_generator_loss``. The strategy is built with ``__new__`` and a fake
generator so the GAN machinery is not needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies import gan as gan_mod  # noqa: E402
from spectramr.infrastructure.training.strategies.betavaegan_strategy import (  # noqa: E402
    BetaVAEGANStrategy,
)


class _FakeGen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def beta_tc_term(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x)
        return torch.tensor(2.0, requires_grad=True)


class _PlainGen(torch.nn.Module):
    """No beta_tc_term."""


def _make_strategy(generator, weight: float = 1.0) -> BetaVAEGANStrategy:
    s = BetaVAEGANStrategy.__new__(BetaVAEGANStrategy)
    s.env = SimpleNamespace(generator=generator)
    s.config = SimpleNamespace(training=SimpleNamespace(gan=SimpleNamespace(beta_tc_weight=weight)))
    s._beta_tc_input = None
    return s


def test_generator_step_feeds_its_input_to_the_aux_term(monkeypatch) -> None:
    gen = _FakeGen()
    s = _make_strategy(gen, weight=0.5)
    monkeypatch.setattr(
        gan_mod.GANTrainingStrategy, "_train_generator_step", lambda self, i, t, *a, **k: "closure"
    )
    x = torch.randn(2, 1, 16, 16)
    assert s._train_generator_step(x, x, None, 0, 0) == "closure"
    extra, comps = s._auxiliary_generator_loss(x, None, x)
    assert gen.seen[0] is x
    assert torch.isclose(extra, torch.tensor(1.0))  # 0.5 * 2.0
    assert torch.isclose(comps["loss_beta_tc"], torch.tensor(2.0))


def test_zero_weight_logs_the_term_but_adds_nothing() -> None:
    s = _make_strategy(_FakeGen(), weight=0.0)
    s._beta_tc_input = torch.randn(2, 1, 8, 8)
    extra, comps = s._auxiliary_generator_loss(s._beta_tc_input, None, s._beta_tc_input)
    assert float(extra.detach()) == 0.0
    assert "loss_beta_tc" in comps


def test_aux_term_without_recorded_input_raises() -> None:
    s = _make_strategy(_FakeGen())
    with pytest.raises(RuntimeError, match="no generator input"):
        s._auxiliary_generator_loss(torch.zeros(1), None, torch.zeros(1))


def test_generator_without_beta_tc_term_raises(monkeypatch) -> None:
    monkeypatch.setattr(gan_mod.GANTrainingStrategy, "__init__", lambda self, *a, **k: None)
    s = BetaVAEGANStrategy.__new__(BetaVAEGANStrategy)
    s.env = SimpleNamespace(generator=_PlainGen())
    with pytest.raises(TypeError, match="beta_tc_term"):
        BetaVAEGANStrategy.__init__(s)
