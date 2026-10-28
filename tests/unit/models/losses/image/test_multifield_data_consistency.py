"""Multi-field data consistency: weighted per-field fidelity and its argument guards."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.image.multifield_data_consistency import MultiFieldDataConsistency


def test_unweighted_is_the_mean_of_per_field_l1() -> None:
    pred, target = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    out = MultiFieldDataConsistency()(pred, target)
    torch.testing.assert_close(out, (pred - target).abs().mean())


def test_weighted_is_the_normalised_weighted_mean() -> None:
    pred, target = torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)
    per_field = (pred - target).pow(2).mean(dim=(0, 2, 3))
    out = MultiFieldDataConsistency(field_weights=(1.0, 3.0), reduction_p=2)(pred, target)
    torch.testing.assert_close(out, (per_field[0] + 3 * per_field[1]) / 4)


def test_all_zero_weights_raise_instead_of_training_on_a_zero_loss() -> None:
    with pytest.raises(ValueError, match="all be zero"):
        MultiFieldDataConsistency(field_weights=(0.0, 0.0))


def test_a_weight_count_that_does_not_match_the_fields_raises() -> None:
    with pytest.raises(ValueError, match="field_weights has 2 entries"):
        MultiFieldDataConsistency(field_weights=(1.0, 1.0))(
            torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
        )
