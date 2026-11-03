"""Tests for FieldCocycleGenerator (MICCAI MRIxFields2026, idea 4.2)."""

from __future__ import annotations

import torch

from spectramr.models.generators.field_cocycle_generator import FieldCocycleGenerator


def test_forward_preserves_shape() -> None:
    g = FieldCocycleGenerator()
    x = torch.rand(2, 1, 16, 16)
    y = g(
        x,
        field_strength=torch.tensor([0.1, 1.5]),
        field_strength_target=torch.tensor([3.0, 7.0]),
        contrast_id=torch.tensor([0, 1]),
    )
    assert y.shape == x.shape


def test_single_model_attribute() -> None:
    # The Task-3 anti-ensemble contract read by the field_cocycle_single_model guard.
    assert FieldCocycleGenerator.is_unified_single_model is True


def test_encode_stamps_canonical_repr() -> None:
    g = FieldCocycleGenerator()
    assert g.last_canonical_repr is None
    q = g.encode(torch.rand(2, 1, 16, 16), torch.tensor([0.1, 3.0]))
    assert g.last_canonical_repr is not None
    assert g.last_canonical_repr.shape == q.shape


def test_registered_with_unified_capability() -> None:
    from spectramr.models.registry import get_model_class

    cls = get_model_class("field_cocycle_generator")
    assert getattr(cls, "is_unified_single_model", False) is True
