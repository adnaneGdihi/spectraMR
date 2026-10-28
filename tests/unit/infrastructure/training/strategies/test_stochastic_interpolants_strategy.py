"""Regression test for the 2026-06 strategy-audit finding SAQ-001 on
``stochastic_interpolants_strategy``.

Finding SAQ-001 [error / silent_fallback]:
    ``_compute_losses_impl`` dispatched the time-conditioned forward via
    ``try: gen(I_t, t) except TypeError: gen(I_t)``. A genuine ``TypeError``
    raised *inside* ``gen.forward`` (a shape/dtype/kwarg bug) was silently
    swallowed and retried with the wrong (1-arg) signature, masking the real
    failure (pitfall #9). The fix dispatches via the parent's
    ``_generator_accepts_time`` signature introspection so a real error
    propagates.

The heavy base ``__init__`` is bypassed via ``__new__`` and only the attributes
the tested path reads (``self.env``, ``self.device``) are wired.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.stochastic_interpolants_strategy import (
    StochasticInterpolantsStrategy,
)


class _TimeGenRaisesReal:
    """Time-conditioned generator (2 positional args) with a real internal bug.

    ``forward`` declares ``(x, t)`` so ``_generator_accepts_time`` returns True
    and the strategy calls ``gen(I_t, t)``. The forward then raises a TypeError
    that is NOT a signature mismatch (it is a genuine internal failure), which
    must propagate rather than trigger a silent 1-arg retry.
    """

    def forward(self, x, t):
        raise TypeError("internal forward bug: bad dtype in time embedding")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _TimeGen:
    """Well-behaved time-conditioned generator (records the call arity)."""

    def __init__(self):
        self.last_nargs = None

    def forward(self, x, t):
        self.last_nargs = 2
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _NoTimeGen:
    """Generator that only accepts the state (1 positional arg)."""

    def __init__(self):
        self.last_nargs = None

    def forward(self, x):
        self.last_nargs = 1
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _make_strategy(gen):
    strat = StochasticInterpolantsStrategy.__new__(StochasticInterpolantsStrategy)
    strat.env = SimpleNamespace(generator=gen)
    strat.device = torch.device("cpu")
    return strat


def _batch():
    return {
        "target": torch.randn(2, 1, 8, 8),
        "source": torch.randn(2, 1, 8, 8),
    }


class TestTimeDispatchDoesNotSwallowForwardError:
    def test_real_forward_typeerror_propagates(self):
        # The old try/except would have retried gen(I_t) and hidden this.
        strat = _make_strategy(_TimeGenRaisesReal())
        with pytest.raises(TypeError, match="internal forward bug"):
            strat._compute_losses_impl(input_batch=_batch())

    def test_time_conditioned_gen_called_with_time(self):
        gen = _TimeGen()
        strat = _make_strategy(gen)
        out = strat._compute_losses_impl(input_batch=_batch())
        assert gen.last_nargs == 2
        assert "loss_total" in out

    def test_non_time_gen_called_without_time(self):
        gen = _NoTimeGen()
        strat = _make_strategy(gen)
        out = strat._compute_losses_impl(input_batch=_batch())
        assert gen.last_nargs == 1
        assert "loss_total" in out


class TestSourceUsesIntrospectionNotTryExcept:
    def test_source_dispatches_via_accepts_time(self):
        src = inspect.getsource(StochasticInterpolantsStrategy._compute_losses_impl)
        assert "self._generator_accepts_time(gen)" in src
        # The silent ``except TypeError:`` *fallback statement* is gone. Strip
        # comment lines first so the explanatory comment that legitimately
        # names the old pattern does not trip the assertion.
        code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "except TypeError" not in code_only


class TestModelInputContract:
    """Non-negotiable 14: the declared tag must name a tensor that exists.

    This strategy overrides ``_compute_losses_impl``, so it inherited
    ``DiffusionTrainingStrategy``'s ``snapshot_prepared_is_model_input = False``
    / ``snapshot_model_input_tag = "diffusion_step"`` without ever emitting that
    tag — the artifacts pointed a reader at a snapshot that did not exist.
    """

    @staticmethod
    def _bare(recorder: dict) -> StochasticInterpolantsStrategy:
        s = object.__new__(StochasticInterpolantsStrategy)
        s.device = torch.device("cpu")
        s._resolve_legacy_batch = lambda input_batch, kwargs: input_batch  # type: ignore[method-assign]
        s._declared_model_input = None

        def _gen(x: torch.Tensor) -> torch.Tensor:  # 1-arg -> gen(I_t) branch
            recorder["model_saw"] = x
            return torch.zeros_like(x)

        s.env = SimpleNamespace(generator=_gen)
        return s

    def test_declares_the_interpolant_it_feeds(self) -> None:
        recorder: dict = {}
        s = self._bare(recorder)
        x1 = torch.randn(2, 1, 8, 8)
        x0 = torch.randn(2, 1, 8, 8)

        s._compute_losses_impl(input_batch={"target": x1, "source": x0}, epoch=0)

        assert s._declared_model_input is not None, (
            "SI inherits the carve-out, so it owes the wrapper a declaration"
        )
        tensors, extra, in_kspace_keys = s._declared_model_input
        assert torch.equal(tensors["model_input"], recorder["model_saw"])
        assert extra["model_input_key"] == "model_input"
        assert in_kspace_keys == set(), "must be explicit, not None"

    def test_declared_input_carries_the_stochastic_term(self) -> None:
        """What separates SI from flow matching is `sigma(t) * z`.

        Declaring the *deterministic* interpolant would be a plausible-looking
        near-miss: same shape, same scale, wrong tensor. Pin that the declared
        input is not reproducible from x0/x1/t alone.
        """
        recorder: dict = {}
        s = self._bare(recorder)
        torch.manual_seed(0)
        x1 = torch.randn(2, 1, 8, 8)
        x0 = torch.randn(2, 1, 8, 8)

        s._compute_losses_impl(input_batch={"target": x1, "source": x0}, epoch=0)
        tensors, _extra, _keys = s._declared_model_input
        i_t = tensors["model_input"]

        # Recover t from the deterministic part is impossible without z, so
        # instead assert the weaker-but-decisive thing: no convex combination
        # of the endpoints equals I_t, because sigma(t) * z displaces it.
        for frac in torch.linspace(0.0, 1.0, 21):
            deterministic = (1 - frac) * x0 + frac * x1
            assert not torch.allclose(i_t, deterministic, atol=1e-6), (
                "declared input must be the STOCHASTIC interpolant, not the "
                "deterministic flow-matching path"
            )


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: mse_loss(pred, true_velocity) regresses the interpolant velocity.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert StochasticInterpolantsStrategy.__dict__["folds_image_losses"] is False
    assert StochasticInterpolantsStrategy.__dict__["inline_losses"] == frozenset()
