"""Tests for the cross-field-strength foundation model strategy (XFieldFMStrategy).

Regression coverage for two confirmed bugs:

1. ``XFieldFMStrategy._prepare_generator_inputs`` must honor the parent's
   ``(lr_image, forward_kwargs)`` 2-tuple contract. The pre-fix override was
   annotated ``-> dict[str, Any]`` and str-indexed the tuple returned by
   ``super()._prepare_generator_inputs`` (``inputs["field_strength"] = field``),
   which raises ``TypeError: tuple indices must be integers``. Even absent the
   crash, returning a dict would break the sole call site's 2-tuple unpack.

2. ``field_strength`` must propagate from the raw batch / kwargs through
   :class:`ReconstructionMixin._prepare_batch_context_reconstruction` into the
   batch_context. Pre-fix neither the kwargs whitelist nor the raw-batch
   key_mapping carried ``field_strength``, so the override's guard
   (``"field_strength" not in batch``) was always true on real batches and the
   entire native / hypernet conditioning path was dead code.

These tests isolate the override (by stubbing the parent's 2-tuple return) and
the mixin (via a lightweight harness), so neither requires a full DI-built
TrainingEnvironment.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
    ReconstructionMixin,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.xfield_fm_strategy import (
    XFieldFMStrategy,
)


class _FakeEnv:
    """Minimal env stub exposing the attributes the override reads."""

    def __init__(self, generator: Any = None) -> None:
        self.generator = generator
        self.opt_g = None


def _make_strategy(generator: Any = None) -> XFieldFMStrategy:
    """Construct an XFieldFMStrategy bypassing the DI-driven __init__."""
    strat = object.__new__(XFieldFMStrategy)
    strat.device = torch.device("cpu")  # type: ignore[attr-defined]
    strat.env = _FakeEnv(generator)  # type: ignore[attr-defined]
    strat._hypernet = None  # type: ignore[attr-defined]
    # What BaseTrainingStrategy.__init__ sets up for owned modules (#2629).
    strat.config = SimpleNamespace(parallel=None)  # type: ignore[attr-defined]
    strat._owned_modules = {}  # type: ignore[attr-defined]
    strat._owned_wrappers = {}  # type: ignore[attr-defined]
    return strat


def _patch_parent_two_tuple(
    monkeypatch: Any, lr: torch.Tensor, fwd: dict[str, Any]
) -> None:
    """Make the parent return a fixed (lr_image, forward_kwargs) 2-tuple.

    This is the authentic parent contract; the override must thread it through.
    """

    def _fake_parent(self: Any, batch: Any, *args: Any, **kwargs: Any):
        return lr, dict(fwd)

    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_prepare_generator_inputs",
        _fake_parent,
        raising=True,
    )


def test_override_returns_unpackable_two_tuple(monkeypatch: Any) -> None:
    """The override must return a 2-tuple (pre-fix it raised TypeError)."""
    lr = torch.ones(1, 1, 4, 4)
    _patch_parent_two_tuple(monkeypatch, lr, {})
    strat = _make_strategy(generator=None)

    batch = {"field_strength": torch.tensor([3.0])}
    result = strat._prepare_generator_inputs(batch)

    assert isinstance(result, tuple)
    assert len(result) == 2
    out_lr, forward_kwargs = result  # would fail on a dict/3-key pre-fix return
    assert torch.is_tensor(out_lr)
    assert isinstance(forward_kwargs, dict)


def test_field_strength_routes_into_forward_kwargs_native(monkeypatch: Any) -> None:
    """Native path: a generator with a field_strength kwarg gets it in forward_kwargs."""
    lr = torch.ones(1, 1, 4, 4)
    _patch_parent_two_tuple(monkeypatch, lr, {})

    class _GenWithField(torch.nn.Module):
        def forward(self, x: torch.Tensor, field_strength: torch.Tensor | None = None):
            return x

    strat = _make_strategy(generator=_GenWithField())
    batch = {"field_strength": torch.tensor([3.0])}

    _out_lr, forward_kwargs = strat._prepare_generator_inputs(batch)
    assert "field_strength" in forward_kwargs
    assert torch.allclose(forward_kwargs["field_strength"], torch.tensor([3.0]))
    # Native path does NOT engage the hypernet.
    assert "hypernet_modulation" not in forward_kwargs


def test_hypernet_modulation_routes_into_forward_kwargs(monkeypatch: Any) -> None:
    """Hypernet path: generator lacking the kwarg triggers (γ,β) modulation."""
    lr = torch.ones(1, 1, 4, 4)
    _patch_parent_two_tuple(monkeypatch, lr, {})

    class _PlainGen(torch.nn.Module):
        out_channels = 4

        def forward(self, x: torch.Tensor):
            return x

    strat = _make_strategy(generator=_PlainGen())
    batch = {"field_strength": torch.tensor([3.0])}

    _out_lr, forward_kwargs = strat._prepare_generator_inputs(batch)
    assert "hypernet_context" in forward_kwargs
    assert "hypernet_modulation" in forward_kwargs
    mod = forward_kwargs["hypernet_modulation"]
    assert set(mod.keys()) == {"gamma", "beta"}
    assert mod["gamma"].shape[-1] == 4


def test_no_field_strength_is_transparent_passthrough(monkeypatch: Any) -> None:
    """Absent field_strength, the override returns the parent tuple unchanged."""
    lr = torch.ones(1, 1, 4, 4)
    _patch_parent_two_tuple(monkeypatch, lr, {"mask": torch.ones(1, 1, 4, 4)})
    strat = _make_strategy(generator=None)

    out_lr, forward_kwargs = strat._prepare_generator_inputs({})
    assert out_lr is lr
    assert "field_strength" not in forward_kwargs
    assert "hypernet_context" not in forward_kwargs
    assert "mask" in forward_kwargs  # parent kwargs preserved


# --------------------------------------------------------------------------- #
# Mixin propagation (shared file) — without it the override guard is dead code  #
# --------------------------------------------------------------------------- #


class _MixinHarness(ReconstructionMixin):
    """Minimal harness exercising _prepare_batch_context_reconstruction.

    Stubs the env/config attributes the mixin reads (input_type, dc_layer,
    multislice flags are read later; context build only needs env/config here).
    """

    class _Cfg:
        class _Model:
            input_type = "image"

        model = _Model()

    class _Env:
        config = None
        generator = None

    def __init__(self) -> None:
        self.env = self._Env()
        self.env.config = self._Cfg()
        self.state = self.env
        self.logging_service = None
        self.dc_layer = None


def test_mixin_propagates_field_strength_from_raw_batch() -> None:
    """field_strength from the raw dataloader batch reaches the context."""
    harness = _MixinHarness()
    lr = torch.ones(1, 1, 4, 4)
    hr = torch.ones(1, 1, 4, 4)
    ctx = harness._prepare_batch_context_reconstruction(
        lr, hr, batch={"field_strength": torch.tensor([3.0])}
    )
    assert "field_strength" in ctx
    assert torch.allclose(ctx["field_strength"], torch.tensor([3.0]))


def test_mixin_propagates_field_strength_from_kwargs() -> None:
    """field_strength passed as an explicit kwarg reaches the context (whitelist)."""
    harness = _MixinHarness()
    lr = torch.ones(1, 1, 4, 4)
    hr = torch.ones(1, 1, 4, 4)
    ctx = harness._prepare_batch_context_reconstruction(
        lr, hr, field_strength=torch.tensor([1.5])
    )
    assert "field_strength" in ctx
    assert torch.allclose(ctx["field_strength"], torch.tensor([1.5]))


def test_mixin_propagates_field_strength_b0_alias() -> None:
    """The 'b0' alias maps onto field_strength via the raw-batch key_mapping."""
    harness = _MixinHarness()
    lr = torch.ones(1, 1, 4, 4)
    hr = torch.ones(1, 1, 4, 4)
    ctx = harness._prepare_batch_context_reconstruction(
        lr, hr, batch={"b0": torch.tensor([7.0])}
    )
    assert "field_strength" in ctx
    assert torch.allclose(ctx["field_strength"], torch.tensor([7.0]))
