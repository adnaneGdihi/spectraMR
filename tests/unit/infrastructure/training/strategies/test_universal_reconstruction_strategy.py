"""Unit tests for UniversalReconstructionStrategy (PR-7).

The universal / foundation paradigm: one model handling mixed
contrast / field-strength / acceleration, conditioned on a
PHYSICS-DERIVED prompt embedding when acquisition params are present.

We test the override ``_compute_losses_impl`` in isolation. The base
``ReconstructionTrainingStrategy.__init__`` is heavy (LossBuilder, DI,
PINN modules), so — like the override pattern itself — we construct a
bare instance via ``__new__`` and wire only the attributes the override
reads: ``env`` (for ``env.generator``), ``device`` and the inherited
``_resolve_legacy_batch`` staticmethod.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.universal_reconstruction_strategy import (
    UniversalReconstructionStrategy,
)


class _DummyEnv:
    def __init__(self, generator):
        self.generator = generator


def _make_strategy(generator, task_mixture=None):
    strat = UniversalReconstructionStrategy.__new__(UniversalReconstructionStrategy)
    strat.env = _DummyEnv(generator)
    strat.device = torch.device("cpu")
    if task_mixture is not None:
        strat.task_mixture = task_mixture
    return strat


class _PlainGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _PromptableGen(nn.Module):
    """Generator that accepts an optional ``prompt`` conditioning vector."""

    def __init__(self, embed_dim=UniversalReconstructionStrategy.PROMPT_EMBED_DIM):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.film = nn.Linear(embed_dim, 1)
        self.saw_prompt = False

    def forward(self, x, prompt=None):
        y = self.conv(x)
        if prompt is not None:
            self.saw_prompt = True
            scale = self.film(prompt).view(-1, 1, 1, 1)
            y = y * (1.0 + scale)
        return y


class TestLossDictContract:
    def test_returns_loss_total_scalar(self):
        strat = _make_strategy(_PlainGen())
        batch = {
            "input": torch.randn(2, 1, 16, 16),
            "target": torch.randn(2, 1, 16, 16),
        }
        out = strat._compute_losses_impl(input_batch=batch, epoch=0)
        assert "loss_total" in out
        loss = out["loss_total"]
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_no_generator_returns_zero(self):
        strat = _make_strategy(None)
        batch = {"input": torch.randn(1, 1, 8, 8), "target": torch.randn(1, 1, 8, 8)}
        out = strat._compute_losses_impl(input_batch=batch, epoch=0)
        assert out["loss_total"].item() == 0.0


class TestPhysicsPromptConditioning:
    def test_conditioning_runs_when_params_present(self):
        gen = _PromptableGen()
        strat = _make_strategy(gen)
        batch = {
            "input": torch.randn(2, 1, 16, 16),
            "target": torch.randn(2, 1, 16, 16),
            "acquisition_params": torch.tensor(
                [[2000.0, 80.0, 90.0, 3.0], [500.0, 10.0, 15.0, 0.3]]
            ),
        }
        out = strat._compute_losses_impl(input_batch=batch, epoch=0)
        assert gen.saw_prompt is True
        assert out["loss_total"].requires_grad

    def test_conditioning_skipped_gracefully_when_absent(self):
        gen = _PromptableGen()
        strat = _make_strategy(gen)
        batch = {
            "input": torch.randn(2, 1, 16, 16),
            "target": torch.randn(2, 1, 16, 16),
        }
        out = strat._compute_losses_impl(input_batch=batch, epoch=0)
        assert gen.saw_prompt is False
        assert "loss_total" in out

    def test_plain_generator_without_prompt_kwarg_still_works(self):
        # Generator forward does not accept ``prompt`` — must not crash.
        strat = _make_strategy(_PlainGen())
        batch = {
            "input": torch.randn(2, 1, 16, 16),
            "target": torch.randn(2, 1, 16, 16),
            "acquisition_params": torch.tensor(
                [[2000.0, 80.0, 90.0, 3.0], [500.0, 10.0, 15.0, 0.3]]
            ),
        }
        out = strat._compute_losses_impl(input_batch=batch, epoch=0)
        assert "loss_total" in out


class TestTaskMixture:
    def test_task_mixture_kwarg_is_stamped(self):
        strat = _make_strategy(_PlainGen())
        batch = {"input": torch.randn(1, 1, 8, 8), "target": torch.randn(1, 1, 8, 8)}
        mixture = {"fastmri": 0.5, "m4raw": 0.5}
        out = strat._compute_losses_impl(input_batch=batch, epoch=0, task_mixture=mixture)
        assert strat.task_mixture == mixture
        assert "loss_total" in out

    def test_task_mixture_attr_is_read(self):
        strat = _make_strategy(_PlainGen(), task_mixture={"a": 1.0})
        batch = {"input": torch.randn(1, 1, 8, 8), "target": torch.randn(1, 1, 8, 8)}
        out = strat._compute_losses_impl(input_batch=batch, epoch=0)
        assert strat.task_mixture == {"a": 1.0}
        assert "loss_total" in out


class _CountingExplodingGen(nn.Module):
    """No ``prompt`` parameter; counts entries, then fails inside the forward."""

    def __init__(self):
        super().__init__()
        self.entries = 0

    def forward(self, x):
        self.entries += 1
        raise TypeError("stride tuple malformed inside the backbone")


class _OpaqueForward:
    """A forward that accepts ``prompt`` but whose signature cannot be read."""

    def __init__(self):
        self.__dict__["saw_prompt"] = False

    @property
    def __signature__(self):
        raise ValueError("no signature available for this callable")

    def __call__(self, x, prompt=None):
        self.saw_prompt = prompt is not None
        return x


class _OpaqueGen:
    """Generator whose ``forward`` defeats ``inspect.signature``."""

    def __init__(self):
        self.forward = _OpaqueForward()

    def __call__(self, x, prompt=None):
        return self.forward(x, prompt=prompt)


class TestForwardGeneratorIsIntrospectionOnly:
    """SAQ-001 (#1189): the signature check IS the decision, with no retry tail.

    ``_forward_generator`` asked ``_callable_accepts_kwarg`` and then ignored
    the answer: a ``try: gen(source, prompt=prompt) except TypeError:
    gen(source)`` tail re-attempted the call introspection had just ruled out,
    and in the ``prompt is None`` case retried ``gen(source)`` with the
    identical ``gen(source)``. Either way the only durable effect was to
    swallow a ``TypeError`` raised inside the forward.
    """

    def test_a_failing_unprompted_forward_is_entered_once_not_twice(self):
        # `prompt is None`, so the retired code ran `gen(source)`, caught the
        # in-forward TypeError, and retried the *identical* `gen(source)`. The
        # error surfaced either way -- so the retry's only effect was to run a
        # partially-executing forward a second time.
        gen = _CountingExplodingGen()

        with pytest.raises(TypeError, match="stride tuple malformed"):
            UniversalReconstructionStrategy._forward_generator(gen, torch.randn(1, 1, 8, 8), None)

        assert gen.entries == 1

    def test_an_unintrospectable_generator_now_runs_unconditioned(self):
        # Pins the one deliberate behaviour change. `_callable_accepts_kwarg`
        # returns False when `inspect.signature` raises, and that answer is now
        # final; the retired tail probed with the kwarg anyway and so DID
        # condition such a generator. Resolving an unreadable signature as
        # "unsupported" is how every other conditioning gate in this package
        # behaves, so the divergence was the bug, not the fix.
        gen = _OpaqueGen()
        source = torch.randn(2, 1, 16, 16)
        prompt = torch.randn(2, UniversalReconstructionStrategy.PROMPT_EMBED_DIM)

        UniversalReconstructionStrategy._forward_generator(gen, source, prompt)

        assert gen.forward.saw_prompt is False

    def test_the_promptable_path_still_conditions(self):
        gen = _PromptableGen()
        source = torch.randn(2, 1, 16, 16)
        prompt = torch.randn(2, UniversalReconstructionStrategy.PROMPT_EMBED_DIM)

        UniversalReconstructionStrategy._forward_generator(gen, source, prompt)

        assert gen.saw_prompt is True


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: l1_loss(pred, target) is the whole objective and nothing else is folded.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert UniversalReconstructionStrategy.__dict__["folds_image_losses"] is False
    assert UniversalReconstructionStrategy.__dict__["inline_losses"] == frozenset({"l1"})
