"""OperatorIdBCHTrainingStrategy reads the framework's input/target roles (#1773)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies import (  # noqa: E402
    operator_id_bch_strategy as bch,
)
from tests.unit.operator_id.test_operator_id_strategy import _make_strategy  # noqa: E402


def test_the_operator_acts_on_the_clean_target_and_is_scored_on_the_degraded_input(
    monkeypatch,
) -> None:
    """The M4Raw route delivers one repetition as ``input`` and the repetition mean
    as ``target``. The strategy read them the other way, so exp(Omega) was applied
    to the single repetition and fitted to the mean: the inverse of the operator
    the arm's hypothesis names."""
    seen: dict[str, torch.Tensor] = {}
    real_expm = bch.expm_multiply

    def _spy_expm(omega, x, **kw):
        seen.setdefault("operand", x.detach().clone())
        return real_expm(omega, x, **kw)

    monkeypatch.setattr(bch, "expm_multiply", _spy_expm)
    strat = _make_strategy()
    real_mmd = strat.mmd_loss

    def _spy_mmd(pushed, observed):
        seen["observed"] = observed.detach().clone()
        return real_mmd(pushed, observed)

    strat.mmd_loss = _spy_mmd

    torch.manual_seed(0)
    degraded = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    clean = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    out = strat._compute_losses_impl(degraded, clean, epoch=0)

    assert torch.isfinite(out["g_total_loss"])
    assert torch.equal(seen["operand"], clean)
    assert torch.equal(seen["observed"], degraded)


def test_the_strategy_declares_the_two_losses_it_builds() -> None:
    """#1956: bch_m4raw listed them under ``losses.complex`` / ``losses.image``,
    keys the schema drops; the strategy is where they are built and weighted."""
    from spectramr.infrastructure.training.strategies.operator_id_bch_strategy import (
        OperatorIdBCHTrainingStrategy,
    )

    assert OperatorIdBCHTrainingStrategy.inline_losses == frozenset(
        {"structured_gaussian_nll", "pushforward_mmd"}
    )
    assert OperatorIdBCHTrainingStrategy.folds_image_losses is False
