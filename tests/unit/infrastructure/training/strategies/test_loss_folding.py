"""Tests for the shared builder-image-loss folding SSOT (loss_folding.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.loss_folding import (
    declared_loss_weights,
    fold_builder_image_losses,
)


def _cfg(entries):
    return SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": n, "weight": w} for n, w in entries],
            kspace_losses=[],
            complex_losses=[],
        )
    )


def test_declared_loss_weights_reads_entries() -> None:
    w = declared_loss_weights(_cfg([("l1", 1.0), ("hfen", 0.2), ("ms_ssim", 0.1)]))
    assert w == {"l1": 1.0, "hfen": 0.2, "ms_ssim": 0.1}


def test_declared_loss_weights_empty_when_no_losses() -> None:
    assert declared_loss_weights(SimpleNamespace()) == {}


def test_fold_returns_none_when_no_env_losses() -> None:
    comp: dict[str, torch.Tensor] = {}
    assert (
        fold_builder_image_losses(
            {}, {}, {}, torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4), comp
        )
        is None
    )


def test_fold_skips_inline_fidelity_terms() -> None:
    """l1/l2/mse are computed inline by the strategy; the fold must skip them so the
    ubiquitous ``[{l1, 1.0}]`` placeholder never double-counts."""
    pred, target = torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)
    env = {"l1": nn.L1Loss(), "hfen": nn.L1Loss()}
    comp: dict[str, torch.Tensor] = {}
    out = fold_builder_image_losses(
        env, {"l1": 1.0, "hfen": 0.5}, {}, pred, target, comp
    )
    assert out is not None
    assert "loss_hfen" in comp and "loss_l1" not in comp  # l1 skipped


def test_fold_scheduled_overrides_declared() -> None:
    """The curriculum override supersedes the static declared weight."""
    pred, target = torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)
    env = {"hfen": nn.L1Loss()}
    base_comp: dict[str, torch.Tensor] = {}
    base = fold_builder_image_losses(env, {"hfen": 0.5}, {}, pred, target, base_comp)
    hi_comp: dict[str, torch.Tensor] = {}
    hi = fold_builder_image_losses(
        env, {"hfen": 0.5}, {"hfen": 2.0}, pred, target, hi_comp
    )
    assert base is not None and hi is not None
    # same raw loss, 4x the weight (2.0 vs 0.5)
    assert float(hi) == float(base) * 4.0


def test_fold_zero_weight_is_skipped() -> None:
    pred, target = torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)
    env = {"hfen": nn.L1Loss()}
    comp: dict[str, torch.Tensor] = {}
    assert fold_builder_image_losses(env, {"hfen": 0.0}, {}, pred, target, comp) is None


def test_fold_zero_valued_dict_loss_not_replaced() -> None:
    """A dict loss whose 'loss' is exactly 0.0 must fold as 0, not the first diagnostic
    entry (the falsy-zero `or` bug)."""

    class _ZeroDictLoss(nn.Module):
        def forward(self, pred, target):  # noqa: ARG002
            return {"psnr": torch.tensor(40.0), "loss": torch.tensor(0.0)}

    env = {"hfen": _ZeroDictLoss()}
    comp: dict = {}
    out = fold_builder_image_losses(
        env, {"hfen": 1.0}, {}, torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4), comp
    )
    assert float(comp["loss_hfen"]) == 0.0
    assert out is not None and float(out) == 0.0


def test_scheduled_overrides_reads_loop_state() -> None:
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.loss_folding import (
        scheduled_overrides,
    )

    obj = SimpleNamespace(loop_state=SimpleNamespace(loss_weight_overrides={"a": 2.0}))
    assert scheduled_overrides(obj) == {"a": 2.0}
    assert scheduled_overrides(SimpleNamespace()) == {}


# ------------------------------------------------------- per-strategy inline_managed
# A strategy that computes a REGISTERED loss inline must be able to declare that loss
# on `losses.image_losses` — that is the only surface LossScheduleController can
# resolve a curriculum rule's base weight from — without the fold applying the module
# a second time. `field_cocycle_anyfield` crashed mid-run for want of that declaration.


def test_inline_managed_with_widens_the_default_skip_set() -> None:
    from spectramr.infrastructure.training.strategies.loss_folding import (
        inline_managed_with,
    )

    widened = inline_managed_with("cocycle_consistency")
    assert {"l1", "l2", "cocycle_consistency"} <= widened
    # The default is unchanged when nothing extra is declared.
    assert inline_managed_with() == inline_managed_with()
    assert "cocycle_consistency" not in inline_managed_with()


def test_fold_skips_a_strategy_declared_inline_term() -> None:
    from spectramr.infrastructure.training.strategies.loss_folding import (
        inline_managed_with,
    )

    class _Const(nn.Module):
        def forward(self, pred, target):  # noqa: D102
            return torch.tensor(1.0)

    pred, target = torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)
    env = {"cocycle_consistency": _Const(), "hfen": _Const()}
    declared = {"cocycle_consistency": 0.1, "hfen": 0.2}

    comp: dict[str, torch.Tensor] = {}
    both = fold_builder_image_losses(env, declared, {}, pred, target, comp)
    assert float(both) == pytest.approx(0.3)  # 0.1 + 0.2, the double-count

    comp = {}
    only_hfen = fold_builder_image_losses(
        env,
        declared,
        {},
        pred,
        target,
        comp,
        inline_managed=inline_managed_with("cocycle_consistency"),
    )
    assert float(only_hfen) == pytest.approx(0.2)
    assert "loss_cocycle_consistency" not in comp  # skipped, not folded


def test_declaring_an_inline_term_makes_its_weight_resolvable() -> None:
    """The point of the declaration: the schedule controller can now find a base."""
    from spectramr.models.losses.weights import build_loss_weight_table

    table = build_loss_weight_table(
        _cfg([("l1", 1.0), ("cocycle_consistency", 0.1)]).losses
    )
    assert table.weight("cocycle_consistency", iteration=1_000_000) == pytest.approx(0.1)


# --- declares_inline_objective: the pairing LossBuilder.validate() rests on -----------


@pytest.mark.parametrize(
    ("inline", "folds", "expected"),
    [
        (frozenset(), False, True),  # owns its objective, folds nothing
        (frozenset({"l1"}), False, True),  # a non-empty declaration is ownership too
        (frozenset(), True, False),  # folds the builder's modules -- it needs them
        (None, False, False),  # half-declared: never says what it computes
        (None, None, False),  # silent
        (frozenset(), None, False),  # half-declared the other way
    ],
    ids=["empty-nofold", "named-nofold", "empty-folds", "nofold-only", "silent", "inline-only"],
)
def test_declares_inline_objective_truth_table(inline, folds, expected) -> None:
    """Both declarations are tri-state and ``None`` is falsy, so the predicate must test
    ``is False`` rather than ``not folds`` -- the ``nofold-only`` and ``inline-only`` rows
    are what separate the two spellings."""
    from spectramr.infrastructure.training.strategies.loss_folding import (
        declares_inline_objective,
    )

    cls = type("_Probe", (), {"inline_losses": inline, "folds_image_losses": folds})
    assert declares_inline_objective(cls) is expected


def test_the_bases_own_none_is_not_a_declaration() -> None:
    """``BaseTrainingStrategy`` sets both ClassVars to ``None`` in its own ``__dict__``.
    A subclass that declares nothing must still read as undeclared, or every strategy in
    the tree would inherit the exemption."""
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
    from spectramr.infrastructure.training.strategies.loss_folding import (
        declares_inline_objective,
    )

    silent = type("_SilentStrategy", (BaseTrainingStrategy,), {})
    assert declares_inline_objective(silent) is False
