"""Tests for ``BaseLoss`` and ``IComputableLoss`` protocol.

Targets ``spectramr.models.losses.base_loss``. Common reduction +
input-validation logic shared by all loss subclasses.

Categories:

- Construction: invalid ``reduction`` raises
- ``_validate_inputs``: shape / device mismatch raises
- ``_apply_reduction``: mean / sum / none branches
- Base ``forward`` raises NotImplementedError
- ``IComputableLoss`` protocol satisfied by anything implementing
  ``forward(pred, target, **context) -> dict``
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.base_loss import BaseLoss, IComputableLoss

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_valid_reductions_accepted(reduction: str) -> None:
    """Documented reductions construct successfully."""
    loss = BaseLoss(reduction=reduction)
    assert loss.reduction == reduction


def test_invalid_reduction_raises() -> None:
    """Unknown reduction → ``ValueError``."""
    with pytest.raises(ValueError, match="Invalid reduction type"):
        BaseLoss(reduction="median")


def test_default_reduction_is_mean() -> None:
    """Default reduction is ``mean``."""
    loss = BaseLoss()
    assert loss.reduction == "mean"


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------


def test_validate_inputs_passes_for_compatible() -> None:
    """Compatible tensors don't raise."""
    loss = BaseLoss()
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    loss._validate_inputs(pred, target)  # no exception


def test_validate_inputs_shape_mismatch_raises() -> None:
    """Mismatched shape → ``ValueError``."""
    loss = BaseLoss()
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 8, 8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss._validate_inputs(pred, target)


# ---------------------------------------------------------------------------
# _apply_reduction
# ---------------------------------------------------------------------------


def test_apply_reduction_mean_returns_scalar() -> None:
    """``mean`` reduction → 0-dim tensor."""
    loss = BaseLoss(reduction="mean")
    out = loss._apply_reduction(torch.tensor([1.0, 2.0, 3.0]))
    assert out.dim() == 0
    assert out.item() == 2.0


def test_apply_reduction_sum_returns_scalar() -> None:
    """``sum`` reduction → 0-dim scalar."""
    loss = BaseLoss(reduction="sum")
    out = loss._apply_reduction(torch.tensor([1.0, 2.0, 3.0]))
    assert out.dim() == 0
    assert out.item() == 6.0


def test_apply_reduction_none_returns_unchanged() -> None:
    """``none`` reduction → tensor unchanged."""
    loss = BaseLoss(reduction="none")
    inp = torch.tensor([1.0, 2.0, 3.0])
    out = loss._apply_reduction(inp)
    assert torch.equal(out, inp)


# ---------------------------------------------------------------------------
# Base forward raises
# ---------------------------------------------------------------------------


def test_base_forward_raises_not_implemented() -> None:
    """Calling ``forward`` on the base class raises."""
    loss = BaseLoss()
    pred = torch.randn(1, 1, 4, 4)
    target = torch.randn(1, 1, 4, 4)
    with pytest.raises(NotImplementedError):
        loss(pred, target)


def test_subclass_can_override_forward() -> None:
    """A concrete subclass can override forward + use the helpers."""

    class MyMSE(BaseLoss):
        def forward(self, pred, target):
            self._validate_inputs(pred, target)
            return self._apply_reduction((pred - target) ** 2)

    loss = MyMSE(reduction="mean")
    pred = torch.full((1, 1, 4, 4), 2.0)
    target = torch.full((1, 1, 4, 4), 0.0)
    out = loss(pred, target)
    assert pytest.approx(out.item()) == 4.0  # mean of 4.0² = 4.0


# ---------------------------------------------------------------------------
# IComputableLoss protocol
# ---------------------------------------------------------------------------


def test_protocol_satisfied_by_dict_returning_loss() -> None:
    """An object with ``forward(pred, target, **ctx) -> dict`` satisfies the protocol."""

    class _MyLoss:
        def forward(self, pred, target, **ctx):
            return {"loss": (pred - target).abs().mean()}

    instance: IComputableLoss = _MyLoss()
    out = instance.forward(torch.zeros(2), torch.ones(2))
    assert "loss" in out


def test_protocol_runtime_checkable() -> None:
    """``IComputableLoss`` is ``@runtime_checkable``."""

    class _MyLoss:
        def forward(self, pred, target, **ctx):
            return {"loss": torch.tensor(0.0)}

    assert isinstance(_MyLoss(), IComputableLoss)


def test_protocol_rejects_objects_without_forward() -> None:
    """Objects without ``forward`` are not instances of the protocol."""

    class _NotALoss:
        pass

    assert not isinstance(_NotALoss(), IComputableLoss)


@pytest.mark.parametrize("cls", [BaseLoss, IComputableLoss])
def test_the_math_block_survives_as_latex(cls: type) -> None:
    """Non-raw docstrings turned ``\\text`` into a TAB plus ``ext``."""
    doc = cls.__doc__ or ""
    assert "\t" not in doc
    assert "\\text{abstract}" in doc
    assert "\\mathcal{L}" in doc and "\\\\mathcal" not in doc
