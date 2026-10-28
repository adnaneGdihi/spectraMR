"""Reading the source-contrast reference off a diffusion batch (#2452)."""

from __future__ import annotations

import pytest
import torch

from spectramr.data.batch_types import TrainingBatch
from spectramr.infrastructure.training.strategies.mixins.reference_batch import (
    reference_from_batch,
    refuse_stacked_cross_contrast,
)


def _batch(**metadata) -> TrainingBatch:
    x = torch.zeros(2, 8, 4, 4)
    return TrainingBatch(input=x, target=x, metadata=metadata)


class TestRefuseStacked:
    def test_stacked_batch_raises(self) -> None:
        """Planted: the stamp the stacked layout publishes, on the batch type the loop builds."""
        with pytest.raises(ValueError, match="source_layout: separate"):
            refuse_stacked_cross_contrast(
                _batch(federated_target_channel_start=torch.tensor([8, 8]))
            )

    def test_stacked_dict_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="#2452"):
            refuse_stacked_cross_contrast({"federated_target_channel_start": torch.tensor(8)})

    @pytest.mark.parametrize("batch", [None, {}, {"kspace_source": torch.zeros(1)}])
    def test_other_batches_pass(self, batch) -> None:
        refuse_stacked_cross_contrast(batch)


class TestReferenceFromBatch:
    def test_absent_is_none(self) -> None:
        assert reference_from_batch(_batch(), "cpu") is None

    def test_torchio_collated_image_is_unwrapped(self) -> None:
        ref = torch.randn(2, 8, 4, 4)
        out = reference_from_batch(
            _batch(kspace_source={"data": ref, "affine": None}, kspace_normalized=True), "cpu"
        )
        assert torch.equal(out, ref)

    def test_5d_is_flattened_volume_major_like_the_target(self) -> None:
        ref = torch.randn(2, 8, 4, 4, 3)
        out = reference_from_batch(
            {"kspace_source": ref, "kspace_normalized": True}, "cpu", flat_batch=6
        )
        expected = ref.permute(0, 4, 1, 2, 3).reshape(6, 8, 4, 4)
        assert torch.equal(out, expected)
        # Row v * D + i is slice i of volume v.
        assert torch.equal(out[1 * 3 + 2], ref[1, ..., 2])

    def test_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="flattened alike"):
            reference_from_batch(
                {"kspace_source": torch.randn(2, 8, 4, 4), "kspace_normalized": True},
                "cpu",
                flat_batch=4,
            )

    def test_non_tensor_raises(self) -> None:
        with pytest.raises(TypeError):
            reference_from_batch({"kspace_source": [1, 2]}, "cpu")

    @pytest.mark.parametrize(
        "marker", [None, False, torch.tensor([True, False])], ids=["absent", "false", "partial"]
    )
    def test_an_unnormalized_reference_raises(self, marker) -> None:
        """Planted: the strategy-side normalization path, which scales the target only."""
        batch = {"kspace_source": torch.randn(2, 8, 4, 4)}
        if marker is not None:
            batch["kspace_normalized"] = marker
        with pytest.raises(ValueError, match="enable_kspace_normalization"):
            reference_from_batch(batch, "cpu")

    def test_a_collated_marker_is_accepted(self) -> None:
        batch = {
            "kspace_source": torch.randn(2, 8, 4, 4),
            "kspace_normalized": torch.tensor([True, True]),
        }
        assert reference_from_batch(batch, "cpu").shape == (2, 8, 4, 4)
