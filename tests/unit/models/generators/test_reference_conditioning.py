"""The source-contrast reference contract of the k-space cold-diffusion generator (#2452)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.reference_conditioning import (
    prepend_reference,
    resolve_reference_channels,
)


class TestResolveReferenceChannels:
    def test_default_is_no_reference(self) -> None:
        kwargs: dict = {}
        assert resolve_reference_channels(kwargs) == 0

    def test_reads_the_declared_width_and_leaves_it_for_the_backbone(self) -> None:
        kwargs = {"reference_channels": 8, "other": 1}
        assert resolve_reference_channels(kwargs) == 8
        assert kwargs == {"reference_channels": 8, "other": 1}

    @pytest.mark.parametrize("value", [-1, 2.5, "8", True])
    def test_rejects_a_non_count(self, value) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            resolve_reference_channels({"reference_channels": value})

    @pytest.mark.parametrize("retired", [{"prior_channel_range": [0, 8]}, {"target_channels": 8}])
    def test_retired_stacked_layout_spellings_raise(self, retired) -> None:
        """Planted: each spelling the stacked layout used, one per shape."""
        with pytest.raises(ValueError, match="source_layout: separate"):
            resolve_reference_channels(dict(retired))


class TestPrependReference:
    def test_concatenates_the_reference_ahead_of_the_state(self) -> None:
        x = torch.randn(2, 8, 4, 4)
        ref = torch.randn(2, 8, 4, 4)
        out = prepend_reference(x, ref, 8)
        assert out.shape == (2, 16, 4, 4)
        assert torch.equal(out[:, :8], ref)
        assert torch.equal(out[:, 8:], x)

    def test_no_reference_is_identity(self) -> None:
        x = torch.randn(2, 8, 4, 4)
        assert prepend_reference(x, None, 0) is x

    def test_model_without_reference_refuses_one(self) -> None:
        with pytest.raises(ValueError, match="reference_channels=0"):
            prepend_reference(torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4), 0)

    def test_model_with_reference_refuses_to_run_without_it(self) -> None:
        with pytest.raises(ValueError, match="kspace_source"):
            prepend_reference(torch.randn(1, 8, 4, 4), None, 8)

    @pytest.mark.parametrize(
        "ref_shape",
        [(1, 4, 4, 4), (2, 8, 4, 4), (1, 8, 5, 4), (1, 8, 4)],
        ids=["channels", "batch", "spatial", "rank"],
    )
    def test_shape_mismatch_raises(self, ref_shape) -> None:
        with pytest.raises(ValueError):
            prepend_reference(torch.randn(1, 8, 4, 4), torch.randn(*ref_shape), 8)

    def test_complex_versus_real_interleaved_raises(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            prepend_reference(
                torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4, dtype=torch.complex64), 8
            )
