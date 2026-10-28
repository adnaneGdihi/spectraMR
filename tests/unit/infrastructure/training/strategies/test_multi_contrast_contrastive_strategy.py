"""Regression tests for ``MultiContrastContrastiveStrategy``.

Covers three confirmed-broken behaviors (audit 2026-05-31) that these tests
would FAIL on in the pre-fix code:

1. The ``symmetrize`` knob silently no-opped whenever a momentum target
   encoder was active (default ``use_momentum=True``). The fix makes the
   incompatible combination raise during knob resolution (pitfall #9), so the
   advertised knob can never be silently dropped.
2. A dict batch missing ``view_a`` / ``view_b`` passed ``None`` straight into
   the generator (``gen(None)``) producing an obscure crash. The fix raises a
   descriptive ``RuntimeError`` naming the missing keys (audit F1).
3. The hyperparameters were read via flat-config fallback +
   ``getattr(..., default)``, silently bypassing the nested YAML SSOT. The fix
   reads ``config.training.ssl.multi_contrast_contrastive`` directly and raises
   when the block is absent (pitfalls #9 / #15); the flat ``config.ssl``
   fallback is gone.

Construction note: ``BaseTrainingStrategy.__init__`` requires a fully-built
``TrainingEnvironment`` (optimization/model/physics sections, DI services,
``create_strategy_context``), which is out of scope for a focused unit test.
Following the established pattern in ``test_compute_losses_dispatch_regression``
we bypass ``__init__`` with ``object.__new__`` and exercise the exact
production code paths directly: ``_resolve_mc_knobs`` (the validation that
``__init__`` calls) and ``_compute_losses_impl`` (with attrs the real
``__init__`` would have set).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from spectramr.config.schemas.base import (
    MultiContrastContrastiveConfigSchema,
    SSLConfigSchema,
)
from spectramr.infrastructure.training.strategies.multi_contrast_contrastive_strategy import (
    MultiContrastContrastiveStrategy,
)


class _ProjEncoder(nn.Module):
    """Tiny encoder producing a projection embedding from a flattened view."""

    def __init__(self, in_dim: int = 4, out_dim: int = 3, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def _config_with_block(mc_block: Any | None) -> SimpleNamespace:
    """A config exposing the nested SSOT ``training.ssl.multi_contrast_contrastive``."""
    ssl = SimpleNamespace(multi_contrast_contrastive=mc_block)
    return SimpleNamespace(training=SimpleNamespace(ssl=ssl))


def _block(
    *,
    temperature: float = 0.07,
    symmetrize: bool = False,
    use_momentum: bool = True,
    momentum: float = 0.999,
    warmup_steps: int = 1000,
) -> MultiContrastContrastiveConfigSchema:
    return MultiContrastContrastiveConfigSchema(
        temperature=temperature,
        symmetrize=symmetrize,
        use_momentum=use_momentum,
        momentum=momentum,
        warmup_steps=warmup_steps,
    )


def _make_strategy(
    *,
    block: MultiContrastContrastiveConfigSchema | None = None,
    generator: nn.Module | None = None,
) -> MultiContrastContrastiveStrategy:
    """Bypass __init__ and apply the real knob-resolution path.

    Mirrors ``test_compute_losses_dispatch_regression._make``: build via
    ``object.__new__`` (no DI/env), then run ``_apply_mc_knobs`` (exactly what
    the production ``__init__`` calls) so the strategy carries the same resolved
    attrs. ``_resolve_mc_knobs`` raises for the invalid-config cases.
    """
    s = object.__new__(MultiContrastContrastiveStrategy)
    if block is None:
        block = _block()
    s._apply_mc_knobs(_config_with_block(block))
    s._target_encoder = None
    s._step = 0
    s.env = SimpleNamespace(generator=generator)
    return s


# ---------------------------------------------------------------------------
# Finding #1 — symmetrize must never silently no-op under a momentum encoder.
# ---------------------------------------------------------------------------


def test_symmetrize_with_momentum_raises_not_silent_noop() -> None:
    """symmetrize=True + use_momentum=True must RAISE (pre-fix: silent no-op).

    In the pre-fix code this combination resolved fine and the
    ``and self._target_encoder is None`` guard silently dropped the second
    InfoNCE term — the advertised symmetrize knob became a no-op with no
    warning, raise, or provenance. The fix rejects the combination outright.
    """
    config = _config_with_block(_block(symmetrize=True, use_momentum=True))
    with pytest.raises(ValueError, match="symmetrize=True is incompatible"):
        MultiContrastContrastiveStrategy._resolve_mc_knobs(config)


def test_symmetrize_without_momentum_actually_changes_loss() -> None:
    """When symmetrize is honored, the loss differs from the one-sided value.

    Positive half of the regression: with use_momentum=False the symmetric
    branch has a real gradient path and the loss MUST add the second InfoNCE
    direction (so it differs from the one-sided loss). Not a tautology — it
    compares symmetric vs one-sided, which differ only if the knob is honored.
    """
    gen = _ProjEncoder()

    sym = _make_strategy(block=_block(symmetrize=True, use_momentum=False), generator=gen)
    one = _make_strategy(
        block=_block(symmetrize=False, use_momentum=False), generator=gen
    )

    torch.manual_seed(7)
    view_a = torch.randn(3, 4)
    view_b = torch.randn(3, 4)
    batch = {"view_a": view_a, "view_b": view_b}

    loss_sym = sym._compute_losses_impl(
        input_batch=None, target_batch=None, epoch=0, batch=batch
    )["loss_total"]
    loss_one = one._compute_losses_impl(
        input_batch=None, target_batch=None, epoch=0, batch=batch
    )["loss_total"]

    assert not torch.allclose(loss_sym, loss_one), (
        "symmetrize=True produced the same loss as the one-sided value — the "
        "knob was silently dropped"
    )


# ---------------------------------------------------------------------------
# Finding #2 — missing batch views must raise, never reach gen(None).
# ---------------------------------------------------------------------------


def test_missing_view_a_raises_descriptive_error() -> None:
    """A dict batch lacking ``view_a`` must raise (pre-fix: ``gen(None)``)."""
    strat = _make_strategy(generator=_ProjEncoder())
    batch = {"view_b": torch.randn(2, 4)}  # view_a missing

    with pytest.raises(RuntimeError, match=r"view_a"):
        strat._compute_losses_impl(
            input_batch=None, target_batch=None, epoch=0, batch=batch
        )


def test_present_dict_missing_view_b_raises_naming_contract() -> None:
    """A non-empty dict batch missing ``view_b`` raises citing the contract.

    Uses a present-but-incomplete dict (only ``view_a``) so the resolved batch
    is a truthy dict that reaches the view-key check (an *empty* dict is falsy
    and routes through ``_resolve_legacy_batch`` to the no-usable-batch raise,
    covered by ``test_empty_batch_raises_no_gen_none`` below).
    """
    strat = _make_strategy(generator=_ProjEncoder())
    batch = {"view_a": torch.randn(2, 4)}  # view_b missing

    with pytest.raises(RuntimeError, match=r"co-registered contrast"):
        strat._compute_losses_impl(
            input_batch=None, target_batch=None, epoch=0, batch=batch
        )


def test_empty_batch_raises_no_gen_none() -> None:
    """An empty/absent batch must raise descriptively, never reach gen(None).

    Both the missing-views path and the no-usable-batch path are acceptable —
    the regression is that the strategy raises a descriptive ``RuntimeError``
    instead of passing ``None`` into the generator.
    """
    strat = _make_strategy(generator=_ProjEncoder())

    with pytest.raises(RuntimeError, match=r"multi_contrast_contrastive"):
        strat._compute_losses_impl(
            input_batch=None, target_batch=None, epoch=0, batch={}
        )


# ---------------------------------------------------------------------------
# Finding #3 — knobs read from the nested SSOT; missing block raises; no flat
# fallback.
# ---------------------------------------------------------------------------


def test_missing_nested_block_raises() -> None:
    """An absent ``multi_contrast_contrastive`` block must raise (no defaults)."""
    config = _config_with_block(mc_block=None)
    with pytest.raises(ValueError, match="requires the nested block"):
        MultiContrastContrastiveStrategy._resolve_mc_knobs(config)


def test_knobs_read_from_nested_home() -> None:
    """Knobs are taken verbatim from config.training.ssl.multi_contrast_contrastive."""
    strat = _make_strategy(
        block=_block(
            temperature=0.123,
            symmetrize=False,
            use_momentum=False,
            momentum=0.5,
            warmup_steps=42,
        )
    )
    assert strat._temperature == pytest.approx(0.123)
    assert strat._symmetrize is False
    assert strat._use_momentum is False
    assert strat._momentum == pytest.approx(0.5)
    assert strat._warmup_steps == 42


def test_flat_config_ssl_fallback_is_gone() -> None:
    """A flat ``config.ssl`` (no nested ``training.ssl``) must NOT be honored.

    Pre-fix, ``getattr(self.config, "ssl", None)`` reached outside the nested
    SSOT. The fix deleted that operand, so a flat ``config.ssl`` is ignored and
    the missing nested block raises.
    """
    # training.ssl absent, but a flat config.ssl present with a (would-be) block.
    config = SimpleNamespace(
        training=SimpleNamespace(ssl=None),
        ssl=SimpleNamespace(multi_contrast_contrastive=_block()),
    )
    with pytest.raises(ValueError, match="requires the nested block"):
        MultiContrastContrastiveStrategy._resolve_mc_knobs(config)


# ---------------------------------------------------------------------------
# Schema sanity for the nested sub-block (paired with config/schemas/base.py).
# ---------------------------------------------------------------------------


def test_ssl_config_mounts_block_and_rejects_unknown() -> None:
    ssl = SSLConfigSchema(
        multi_contrast_contrastive={
            "temperature": 0.07,
            "use_momentum": True,
            "symmetrize": False,
        }
    )
    assert ssl.multi_contrast_contrastive is not None
    assert ssl.multi_contrast_contrastive.temperature == pytest.approx(0.07)
    assert SSLConfigSchema().multi_contrast_contrastive is None
    with pytest.raises(ValidationError):
        MultiContrastContrastiveConfigSchema(unknown_field=1)


# ---------------------------------------------------------------------------
# #2820 — the momentum encoder's EMA weights survive a resume.
# ---------------------------------------------------------------------------


def test_a_resume_restores_the_momentum_encoder_not_a_copy_of_the_generator() -> None:
    saved = _make_strategy(generator=_ProjEncoder(seed=0))
    saved._ensure_momentum_encoder()
    with torch.no_grad():
        saved._target_encoder.fc.weight.add_(1.0)

    resumed = _make_strategy(generator=_ProjEncoder(seed=0))
    resumed.load_strategy_state_dict(saved.strategy_state_dict())

    assert torch.equal(resumed._target_encoder.fc.weight, saved._target_encoder.fc.weight)
    assert not torch.equal(resumed._target_encoder.fc.weight, resumed.env.generator.fc.weight)


def test_a_saved_momentum_encoder_is_refused_when_momentum_is_off() -> None:
    saved = _make_strategy(generator=_ProjEncoder(seed=0))
    saved._ensure_momentum_encoder()

    resumed = _make_strategy(block=_block(use_momentum=False), generator=_ProjEncoder(seed=0))
    with pytest.raises(RuntimeError, match="_target_encoder"):
        resumed.load_strategy_state_dict(saved.strategy_state_dict())
