"""``lambda_pre_dc_acquired``: the gradient the acquired bins never had.

Under ``dc_method: hard`` the output is ``(1 - M) * prediction + M * measurement``,
so ``d(output)/d(prediction)`` is exactly ``(1 - M)`` and every POST-DC term is a
constant on the acquired bins. The one term that sees the PRE-DC tensor,
``lambda_pre_dc_kspace``, is masked to ``(1 - M)`` as well. Nothing scored the
lines the scanner actually measured.

What is pinned here is the PARTITION, in both directions: the new term must carry
gradient on ``M`` and none off it, the old term the reverse, and the two together
must cover the plane exactly once. A test that only checked the new term were
non-zero would pass on a term that also leaked into the null band and quietly
double-counted it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

_B, _C, _S = 1, 8, 16


def _strategy(lam_null: float = 0.0, lam_acquired: float = 0.0):
    """A strategy stub carrying only what ``_add_pre_dc_fidelity`` reads.

    ``SimpleNamespace``, not ``MagicMock``: a mock's ``__float__`` returns 1.0,
    so an unset weight would read as enabled and the assertions below would be
    measuring the wrong term.
    """
    from unittest.mock import MagicMock

    strategy = MagicMock(spec=DiffusionTrainingStrategy)
    strategy._loss_dict_reuse = {}
    strategy.config = SimpleNamespace(
        losses=SimpleNamespace(
            reconstruction=SimpleNamespace(
                lambda_pre_dc_kspace=lam_null,
                lambda_pre_dc_acquired=lam_acquired,
            )
        )
    )
    strategy._unsampled_weight = DiffusionTrainingStrategy._unsampled_weight
    strategy._acquired_weight = DiffusionTrainingStrategy._acquired_weight
    return strategy


def _half_mask() -> torch.Tensor:
    """Rows 0..7 acquired, rows 8..15 not -- a 1-D phase-encode mask."""
    mask = torch.zeros(_B, 1, _S, _S)
    mask[:, :, : _S // 2, :] = 1.0
    return mask


def _run(strategy, pre_dc, target, mask):
    return DiffusionTrainingStrategy._add_pre_dc_fidelity(
        strategy, torch.zeros(()), (torch.zeros_like(pre_dc), pre_dc), target, mask
    )


class TestTheGradientLandsOnTheAcquiredBins:
    def test_the_new_term_carries_gradient_only_where_the_mask_is_one(self) -> None:
        torch.manual_seed(0)
        target = torch.randn(_B, _C, _S, _S)
        pre_dc = torch.randn(_B, _C, _S, _S, requires_grad=True)
        mask = _half_mask()

        _run(_strategy(lam_acquired=1.0), pre_dc, target, mask).backward()

        acquired = mask.expand_as(pre_dc) > 0
        assert torch.all(pre_dc.grad[~acquired] == 0.0), "leaked into the null band"
        assert torch.any(pre_dc.grad[acquired] != 0.0), "no gradient where it was measured"

    def test_the_old_term_still_carries_gradient_only_off_the_mask(self) -> None:
        """The planted counterpart: the partition has two sides and the existing
        half must not have moved."""
        torch.manual_seed(0)
        target = torch.randn(_B, _C, _S, _S)
        pre_dc = torch.randn(_B, _C, _S, _S, requires_grad=True)
        mask = _half_mask()

        _run(_strategy(lam_null=1.0), pre_dc, target, mask).backward()

        acquired = mask.expand_as(pre_dc) > 0
        assert torch.all(pre_dc.grad[acquired] == 0.0)
        assert torch.any(pre_dc.grad[~acquired] != 0.0)

    def test_together_they_cover_the_plane_exactly_once(self) -> None:
        """Declaring both must not double-count any bin: at equal weights the
        sum of the two masked means is the uniform mean."""
        torch.manual_seed(0)
        target = torch.randn(_B, _C, _S, _S)
        pre_dc = torch.randn(_B, _C, _S, _S)
        mask = _half_mask()

        both = _run(_strategy(lam_null=1.0, lam_acquired=1.0), pre_dc, target, mask)
        uniform = (pre_dc - target).abs().mean()
        # Each term is a mean over its own half, so their sum is twice the
        # grand mean only when the halves are equal in size -- which they are.
        assert float(both) == pytest.approx(float(2 * uniform), rel=1e-5)


class TestItIsOffUntilDeclared:
    def test_both_weights_zero_is_the_same_object_back(self) -> None:
        total = torch.zeros(())
        strategy = _strategy()
        out = DiffusionTrainingStrategy._add_pre_dc_fidelity(
            strategy,
            total,
            (torch.zeros(_B, _C, _S, _S), torch.ones(_B, _C, _S, _S)),
            torch.zeros(_B, _C, _S, _S),
            _half_mask(),
        )
        assert out is total
        assert strategy._loss_dict_reuse == {}

    def test_the_old_term_alone_stamps_only_its_own_key(self) -> None:
        strategy = _strategy(lam_null=0.3)
        _run(strategy, torch.ones(_B, _C, _S, _S), torch.zeros(_B, _C, _S, _S), _half_mask())
        assert "pre_dc_kspace_l1" in strategy._loss_dict_reuse
        assert "pre_dc_acquired_l1" not in strategy._loss_dict_reuse


class TestAbsentSupportIsReportedNotInferred:
    """``None`` means different things for the two weights, deliberately.

    The null-band term falls back to a uniform L1, because a fully-sampled rung
    has no unmeasured bins and that fallback is the only gradient the generator
    gets at ``t = 0``. The acquired-band term must NOT do that -- a uniform L1
    there would silently duplicate the null term over the whole plane.
    """

    @pytest.mark.parametrize(
        ("mask", "label"),
        [(None, "absent"), (torch.zeros(_B, 1, _S, _S), "all-zero")],
        ids=["absent", "all-zero"],
    )
    def test_it_stands_down_and_says_so(self, mask, label) -> None:
        strategy = _strategy(lam_acquired=1.0)
        target = torch.zeros(_B, _C, _S, _S)
        out = _run(strategy, torch.ones(_B, _C, _S, _S), target, mask)
        assert float(out) == 0.0, f"{label} support must not contribute a uniform L1"
        assert float(strategy._loss_dict_reuse["pre_dc_acquired_l1"]) == 0.0

    def test_a_fully_sampled_rung_is_all_acquired(self) -> None:
        """The mirror of ``_unsampled_weight``'s all-ones -> None contract."""
        strategy = _strategy(lam_acquired=1.0)
        ones = torch.ones(_B, 1, _S, _S)
        out = _run(strategy, torch.ones(_B, _C, _S, _S), torch.zeros(_B, _C, _S, _S), ones)
        assert float(out) == pytest.approx(1.0)


class TestTheColumnIsPromisedExactlyWhenItIsWritten:
    @pytest.mark.parametrize(
        ("lam_null", "lam_acquired", "expected"),
        [
            (0.0, 0.0, set()),
            (0.3, 0.0, {"pre_dc_kspace_l1"}),
            (0.0, 0.3, {"pre_dc_acquired_l1"}),
            (0.3, 0.3, {"pre_dc_kspace_l1", "pre_dc_acquired_l1"}),
            (0.0, 1e-12, {"pre_dc_acquired_l1"}),
        ],
    )
    def test_declared_metric_keys(self, lam_null, lam_acquired, expected) -> None:
        strategy = _strategy(lam_null=lam_null, lam_acquired=lam_acquired)
        keys = DiffusionTrainingStrategy.declared_metric_keys(strategy)
        assert set(keys) == expected


class TestTheBuilderLeavesItToTheStrategy:
    def test_it_is_strategy_managed(self) -> None:
        """A lambda with no registry entry must not be built as a module, or the
        LossBuilder raises on an unknown name at construction."""
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )

        assert "pre_dc_acquired" in STRATEGY_MANAGED_LOSSES

    def test_the_schema_default_is_off(self) -> None:
        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        assert ReconstructionLossesConfig().lambda_pre_dc_acquired == 0.0

    def test_the_schema_rejects_a_negative_weight(self) -> None:
        import pydantic

        from spectramr.config.schemas.loss import ReconstructionLossesConfig

        with pytest.raises(pydantic.ValidationError):
            ReconstructionLossesConfig(lambda_pre_dc_acquired=-0.1)
