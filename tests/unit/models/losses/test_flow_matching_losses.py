"""Tests for the conditional / rectified / OT flow-matching velocity losses."""

from __future__ import annotations

import itertools

import pytest
import torch

from spectramr.models.losses.flow_matching_losses import (
    ConditionalFlowMatchingLoss,
    OTConditionalFlowMatchingLoss,
    RectifiedFlowLoss,
    _hungarian_pair,
)

pytest.importorskip("scipy")


def _pairing_cost(x0: torch.Tensor, x1: torch.Tensor) -> float:
    return float((x0 - x1).flatten(1).pow(2).sum())


def _optimal_cost(x0: torch.Tensor, x1: torch.Tensor) -> float:
    return min(_pairing_cost(x0[list(p)], x1) for p in itertools.permutations(range(x0.shape[0])))


@pytest.mark.parametrize("seed", range(5))
def test_hungarian_permutation_indexes_x0_at_the_assignment_optimum(seed: int) -> None:
    """``x0[perm][j]`` must be the OT partner of ``x1[j]``; the inverse is suboptimal."""
    torch.manual_seed(seed)
    x0 = torch.randn(5, 1, 3, 3)
    x1 = torch.randn(5, 1, 3, 3) + 2.0
    perm = _hungarian_pair(x0, x1)
    assert _pairing_cost(x0[perm], x1) == pytest.approx(_optimal_cost(x0, x1), rel=1e-6)


def test_ot_cfm_is_zero_at_the_ot_paired_velocity() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(5, 1, 4, 4)
    x1 = torch.randn(5, 1, 4, 4) + 1.0
    ctx: dict[str, object] = {"x0": x0, "return_permutation": True}
    best = min(itertools.permutations(range(5)), key=lambda p: _pairing_cost(x0[list(p)], x1))
    velocity = x1 - x0[list(best)]
    loss = OTConditionalFlowMatchingLoss()(velocity, x1, context=ctx)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)
    assert torch.equal(ctx["permutation"], torch.tensor(best))  # type: ignore[arg-type]


@pytest.mark.parametrize("cls", [ConditionalFlowMatchingLoss, RectifiedFlowLoss])
def test_linear_path_target_is_x1_minus_x0(cls: type[torch.nn.Module]) -> None:
    torch.manual_seed(0)
    x0, x1 = torch.randn(3, 1, 4, 4), torch.randn(3, 1, 4, 4)
    assert float(cls()(x1 - x0, x1, context={"x0": x0})) == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(
        cls()(torch.zeros_like(x1), x1, context={"x0": x0}), (x1 - x0).pow(2).mean()
    )


@pytest.mark.parametrize(
    "cls", [ConditionalFlowMatchingLoss, RectifiedFlowLoss, OTConditionalFlowMatchingLoss]
)
@pytest.mark.parametrize("context", [None, {}, {"t": torch.rand(2)}])
def test_missing_x0_raises_instead_of_drawing_an_unrelated_one(
    cls: type[torch.nn.Module], context: dict[str, torch.Tensor] | None
) -> None:
    """A fresh ``x0`` is independent of the ``x_t`` the prediction was computed at."""
    x1 = torch.randn(2, 1, 4, 4)
    with pytest.raises(ValueError, match=r"context\['x0'\]"):
        cls()(torch.zeros_like(x1), x1, context=context)
