"""Per-sample draws: sample b must not depend on how many samples follow it."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.per_sample_rng import (
    per_sample_normal_pair,
    per_sample_rand,
    per_sample_randn,
)


@pytest.mark.parametrize("draw", [per_sample_rand, per_sample_randn])
def test_first_sample_is_the_same_for_any_batch_size(draw) -> None:
    one = draw(1, 3, 17, generator=torch.Generator().manual_seed(4))
    five = draw(5, 3, 17, generator=torch.Generator().manual_seed(4))
    assert five.shape == (5, 3, 17)
    assert torch.equal(five[:1], one)


def test_batch_of_one_is_a_plain_draw() -> None:
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    assert torch.equal(per_sample_rand(1, 6, generator=g1)[0], torch.rand(6, generator=g2))


def test_normal_pair_interleaves_within_each_sample() -> None:
    a1, b1 = per_sample_normal_pair(1, 4, 4, generator=torch.Generator().manual_seed(2))
    a3, b3 = per_sample_normal_pair(3, 4, 4, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a3[:1], a1) and torch.equal(b3[:1], b1)
    assert not torch.equal(a1, b1)
