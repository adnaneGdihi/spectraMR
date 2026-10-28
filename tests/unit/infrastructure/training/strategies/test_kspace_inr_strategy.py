"""Loss-ownership pin for ``kspace_inr_strategy`` (issues #1918, #2801)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from spectramr.infrastructure.training.strategies.kspace_inr_strategy import KSpaceINRStrategy
from spectramr.infrastructure.training.strategies.loss_folding import declares_inline_objective
from spectramr.infrastructure.validation.witness.checks.loss_ownership_checks import (
    declared_image_loss_names,
    loss_ownership_defect,
)

_ARM = (
    Path(__file__).resolve().parents[5]
    / "experiments/inprogress/promoted/exp_promoted_kspace_inr.yaml"
)


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: l1_loss(pred_k, kspace) is a k-space term, not the canonical image-space l1.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for
    ITSELF and becomes a lie the moment a subclass replaces
    ``_compute_losses_impl``. An inherited True makes the audit's
    ``image_losses_reach_the_objective`` witness PASS every declared
    ``losses.image_losses`` entry on this strategy's arms while the training
    step discards them.
    """
    assert KSpaceINRStrategy.__dict__["folds_image_losses"] is False
    assert KSpaceINRStrategy.__dict__["inline_losses"] == frozenset()


def _declared_on_the_arm() -> list[str]:
    """The arm's enabled ``losses.image_losses`` names, read as the witness reads them."""
    raw = yaml.safe_load(_ARM.read_text(encoding="utf-8"))
    entries = (raw.get("losses") or {}).get("image_losses") or []
    return declared_image_loss_names(SimpleNamespace(losses=SimpleNamespace(image_losses=entries)))


def test_a_declared_image_l1_is_reported_as_dropped() -> None:
    """Planted shape of #2801: an image ``l1`` on this strategy reaches no route."""
    message, declared = loss_ownership_defect(["l1"], KSpaceINRStrategy)
    assert declared is True
    assert message is not None and "['l1']" in message


def test_the_arm_declares_no_image_loss_the_strategy_drops() -> None:
    """#2801: the arm's loss census states the objective the strategy actually computes."""
    names = _declared_on_the_arm()
    assert loss_ownership_defect(names, KSpaceINRStrategy) == (None, True), names


def test_an_empty_builder_stack_is_the_declared_design() -> None:
    """``LossBuilder.validate()`` accepts the arm's empty image-loss stack only through this."""
    assert declares_inline_objective(KSpaceINRStrategy) is True


class _OffsetGen(torch.nn.Module):
    """Returns ``kspace + delta``, so the residual the loss sees is exactly ``delta``."""

    def __init__(self, delta: torch.Tensor) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(delta)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.delta


def _loss_total(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s = object.__new__(KSpaceINRStrategy)
    s.env = SimpleNamespace(generator=_OffsetGen(delta))
    s.device = torch.device("cpu")
    batch = {"kspace": torch.randn(1, 2, 4, 4), "mask": mask}
    return s._compute_losses_impl(input_batch=batch, target_batch=None, epoch=0)["loss_total"]


def test_the_inline_objective_is_the_mask_weighted_kspace_l1() -> None:
    """The realised objective: residual off the mask costs nothing, on it costs sum/mask-sum."""
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :, :2] = 1.0
    off_mask = (1.0 - mask).expand(1, 2, 4, 4) * 3.0
    assert _loss_total(off_mask.clone(), mask).item() == pytest.approx(0.0)

    on_mask = mask.expand(1, 2, 4, 4) * 0.5
    expected = (on_mask.abs() * mask).sum() / mask.sum()
    loss = _loss_total(on_mask.clone(), mask)
    assert loss.item() == pytest.approx(expected.item())
    assert loss.requires_grad
