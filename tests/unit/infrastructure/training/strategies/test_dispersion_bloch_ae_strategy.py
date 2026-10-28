"""DispersionBlochAEStrategy loss ownership (#1710)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.dispersion_bloch_ae_strategy import (
    DispersionBlochAEStrategy,
)
from spectramr.models.losses.image.multifield_data_consistency import (
    MultiFieldDataConsistency,
)


def _strategy_with_declared(env_losses: dict) -> DispersionBlochAEStrategy:
    st = object.__new__(DispersionBlochAEStrategy)
    st.env = types.SimpleNamespace(losses=env_losses)
    st._declared_loss_weights = lambda: dict.fromkeys(env_losses, 1.0)
    return st


def test_the_inline_physics_terms_are_declared():
    assert DispersionBlochAEStrategy.inline_losses == frozenset(
        {"multifield_data_consistency", "dispersion_monotonicity"}
    )
    assert DispersionBlochAEStrategy.folds_image_losses is True


def test_a_declared_data_consistency_entry_is_not_folded_a_second_time():
    """dlbae_brain_5field declares ``multifield_data_consistency`` in
    ``image_losses`` while the strategy already computes it inline at
    ``data_consistency_weight``; folding it again trained the term at 2x."""
    st = _strategy_with_declared({"multifield_data_consistency": MultiFieldDataConsistency()})
    pred = torch.rand(2, 5, 8, 8)
    target = torch.rand(2, 5, 8, 8)
    assert st._apply_builder_image_losses(pred, target, {}) is None


def test_other_declared_image_losses_are_still_folded():
    st = _strategy_with_declared({"l1_extra": torch.nn.L1Loss()})
    pred = torch.zeros(2, 5, 8, 8)
    target = torch.ones(2, 5, 8, 8)
    folded = st._apply_builder_image_losses(pred, target, {})
    assert folded is not None and float(folded) > 0
