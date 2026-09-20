"""Tests for the validation EMA weight swap (#2172).

The swap mutates the live generator in place, so every failure here is a
training failure rather than a metrics one. Each test below plants one of the
shapes that shipped: the mutation escaping its own ``try``, a partial swap
reported as an EMA run, and a partial restore that silently keeps shadow
weights.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.optimization.ema import (
    EMAKeyMismatchError,
    EMAWeightRestoreError,
)
from spectramr.infrastructure.optimization.ema_swap import ema_weights_swapped_in


class _Rebuildable(nn.Module):
    """A model whose parameter shape can change mid-block.

    Stands in for the ``channel_adapter`` the production comments describe as
    being rebuilt during a validation forward -- the case that makes a saved
    tensor un-restorable.
    """

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def rebuild(self, width: int) -> None:
        self.lin = nn.Linear(width, width)


class _PrefixWrapper(nn.Module):
    """Registers the model under ``.module``, as DDP / DeepSpeedEngine do."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


def _constant_like(model: nn.Module, value: float) -> nn.Module:
    """A structural clone of *model* with every float tensor set to *value*."""
    shadow = type(model)()
    with torch.no_grad():
        for param in shadow.parameters():
            param.fill_(value)
    return shadow


# ---------------------------------------------------------------------------
# The happy path, and the property the whole thing exists for.
# ---------------------------------------------------------------------------


def test_shadow_weights_are_live_inside_and_gone_outside():
    target = _Rebuildable()
    shadow = _constant_like(target, 1.0)
    before = target.lin.weight.detach().clone()

    with ema_weights_swapped_in(target, shadow):
        assert torch.allclose(target.lin.weight, torch.ones_like(target.lin.weight))

    assert torch.allclose(target.lin.weight, before)


def test_the_capture_is_a_clone_not_a_reference():
    """``state_dict`` hands back references and ``load_state_dict`` copies in
    place, so a capture that skipped the clone would be destroyed by the swap
    it exists to undo."""
    target = _Rebuildable()
    shadow = _constant_like(target, 7.0)
    before = target.lin.weight.detach().clone()

    with ema_weights_swapped_in(target, shadow):
        pass

    assert torch.allclose(target.lin.weight, before)
    assert not torch.allclose(target.lin.weight, torch.full_like(before, 7.0))


# ---------------------------------------------------------------------------
# Risk 1: the mutation used to outrange its own try/finally.
# ---------------------------------------------------------------------------


def test_weights_are_restored_when_the_block_raises():
    """The fix for the 66-line unprotected window.

    The swap previously ran far above the ``with`` that restored it, with no
    enclosing ``try``, so anything raising in between left the generator on
    shadow weights and training resumed from them.
    """
    target = _Rebuildable()
    shadow = _constant_like(target, 1.0)
    before = target.lin.weight.detach().clone()

    with pytest.raises(ValueError, match="boom"), ema_weights_swapped_in(target, shadow):
        raise ValueError("boom")

    assert torch.allclose(target.lin.weight, before)


# ---------------------------------------------------------------------------
# Risk: a total key mismatch must raise BEFORE mutating anything.
# ---------------------------------------------------------------------------


def test_a_wrapped_target_raises_and_leaves_the_weights_untouched():
    inner = _Rebuildable()
    wrapped = _PrefixWrapper(inner)
    shadow = _constant_like(inner, 1.0)
    before = inner.lin.weight.detach().clone()

    with (
        pytest.raises(EMAKeyMismatchError, match="matched 0 of"),
        ema_weights_swapped_in(wrapped, shadow),
    ):
        pytest.fail("the block must not run")

    assert torch.allclose(inner.lin.weight, before)


def test_the_mismatch_message_points_at_unwrapping():
    inner = _Rebuildable()
    shadow = _constant_like(inner, 1.0)
    with (
        pytest.raises(EMAKeyMismatchError) as excinfo,
        ema_weights_swapped_in(_PrefixWrapper(inner), shadow),
    ):
        pass
    assert "unwrapped" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Risk 2: a partial swap grades a blend but was only ever warned per key.
# ---------------------------------------------------------------------------


class _TwoLayer(nn.Module):
    """Two independent layers, so one can mismatch while the other applies."""

    def __init__(self, width: int = 4, second: int = 4) -> None:
        super().__init__()
        self.kept = nn.Linear(width, width)
        self.mismatched = nn.Linear(second, second)


def test_a_partial_swap_reports_the_fraction(caplog):
    """Per-key warnings never add up to the one number that matters."""
    target = _TwoLayer(width=4, second=4)
    shadow = _TwoLayer(width=4, second=6)  # `mismatched` cannot be applied
    with torch.no_grad():
        for param in shadow.parameters():
            param.fill_(1.0)
    kept_before = target.kept.weight.detach().clone()

    with caplog.at_level(logging.WARNING), ema_weights_swapped_in(target, shadow):
        # the matching half really is swapped...
        assert torch.allclose(target.kept.weight, torch.ones_like(target.kept.weight))
        # ...and the mismatching half is still the live weight
        assert not torch.allclose(
            target.mismatched.weight, torch.ones_like(target.mismatched.weight)
        )

    # `getMessage()`, not `.message`: the latter is set by a handler when it
    # formats the record, so it is absent whenever another module in the run
    # has reconfigured logging -- this test passed alone and failed after
    # tests/unit/infrastructure/logging with AttributeError.
    assert any("PARTIAL" in r.getMessage() for r in caplog.records)
    assert torch.allclose(target.kept.weight, kept_before)


class _ThreeLayer(nn.Module):
    """Carries a layer the two-layer target does not have."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.kept = nn.Linear(width, width)
        self.mismatched = nn.Linear(width, width)
        self.only_in_shadow = nn.Linear(width, width)


def test_a_shadow_key_the_target_lacks_is_counted_as_unapplied(caplog):
    """Planted: the shape clash is the visible half of a partial swap.

    A key the target does not have is just as unapplied and leaves no trace --
    ``load_state_dict(..., strict=False)`` never remarks on what it was not
    handed. Counting only the clashes let a stale shadow swap a fraction of the
    model and report a clean EMA run, which is #2172 one size down from the
    total miss.
    """
    target = _TwoLayer(width=4, second=4)
    shadow = _ThreeLayer(width=4)
    with torch.no_grad():
        for param in shadow.parameters():
            param.fill_(1.0)

    with caplog.at_level(logging.WARNING), ema_weights_swapped_in(target, shadow):
        # every shape here agrees, so the clash count is zero and only the
        # absent-key count can raise the warning.
        assert torch.allclose(target.kept.weight, torch.ones_like(target.kept.weight))

    warnings = [r.getMessage() for r in caplog.records if "PARTIAL" in r.getMessage()]
    assert warnings, "a shadow key with no target counterpart must be reported"
    assert "2 absent" in warnings[0], warnings[0]
    assert "0 kept live" in warnings[0], warnings[0]


# ---------------------------------------------------------------------------
# Risk 3: a partial restore is permanent, and was silent.
# ---------------------------------------------------------------------------


def test_an_unrestorable_tensor_raises():
    target = _Rebuildable(width=4)
    shadow = _constant_like(target, 1.0)

    with (
        pytest.raises(EMAWeightRestoreError, match="could not be restored"),
        ema_weights_swapped_in(target, shadow),
    ):
        target.rebuild(6)  # the channel_adapter case


def test_an_unrestorable_tensor_does_not_mask_an_in_flight_failure():
    """A raise from ``finally`` would replace the real cause with this one."""
    target = _Rebuildable(width=4)
    shadow = _constant_like(target, 1.0)

    with pytest.raises(ValueError, match="the real cause"), ema_weights_swapped_in(target, shadow):
        target.rebuild(6)
        raise ValueError("the real cause")


def test_a_clean_restore_does_not_raise():
    """The guard must not fire on the ordinary path."""
    target = _Rebuildable()
    shadow = _constant_like(target, 1.0)

    with ema_weights_swapped_in(target, shadow):
        pass  # no shape change -- everything restorable
