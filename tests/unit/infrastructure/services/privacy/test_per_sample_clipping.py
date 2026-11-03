"""Per-sample clipping helpers: each example is clipped on its own, then averaged."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.services.privacy.per_sample_clipping import (
    accumulate_clipped_gradient,
    gradient_surrogate,
    slice_sample,
)


def test_each_example_is_clipped_individually_not_the_batch() -> None:
    w = torch.nn.Parameter(torch.zeros(3))
    acc = [torch.zeros_like(w)]
    # Example 0: gradient (100, 0, 0) -> clipped to (1, 0, 0). Example 1: (0, 0.5, 0) unclipped.
    accumulate_clipped_gradient((w * torch.tensor([100.0, 0.0, 0.0])).sum(), [w], 1.0, acc)
    norm = accumulate_clipped_gradient((w * torch.tensor([0.0, 0.5, 0.0])).sum(), [w], 1.0, acc)
    assert torch.allclose(acc[0], torch.tensor([1.0, 0.5, 0.0]))
    assert torch.isclose(norm, torch.tensor(0.5))
    # A batch-level clip of the summed gradient would have shrunk the small example too.
    batch_clipped = torch.tensor([100.0, 0.5, 0.0]) / torch.tensor([100.0, 0.5, 0.0]).norm()
    assert not torch.allclose(acc[0], batch_clipped)


def test_unused_parameter_gets_a_zero_gradient() -> None:
    used, unused = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    acc = [torch.zeros(2), torch.zeros(2)]
    accumulate_clipped_gradient(used.sum(), [used, unused], 10.0, acc)
    assert torch.equal(acc[1], torch.zeros(2))


def test_non_positive_clip_raises() -> None:
    w = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="max_grad_norm"):
        accumulate_clipped_gradient(w.sum(), [w], 0.0, [torch.zeros(1)])


def test_surrogate_value_and_gradient() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    target = torch.tensor([3.0, -4.0])
    out = gradient_surrogate([w], [target], torch.tensor(7.5))
    assert torch.isclose(out, torch.tensor(7.5))
    out.backward()
    assert torch.equal(w.grad, target)


def test_slice_sample_slices_batch_tensors_and_nested_mappings() -> None:
    batch = {
        "x": torch.arange(6).view(3, 2),
        "meta": {"m": torch.zeros(3, 1)},
        "s": "id",
        "c": torch.ones(2),
    }
    out = slice_sample(batch, 1, 3)
    assert torch.equal(out["x"], torch.tensor([[2, 3]]))
    assert out["meta"]["m"].shape == (1, 1)
    assert out["s"] == "id"
    assert out["c"].shape == (2,)  # leading dim != batch size: passed through
