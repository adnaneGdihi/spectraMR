"""Unit tests for the rank-normalize block (MCGI invariance core).

Targets ``spectramr.models.blocks.rank_normalize``.

The per-image empirical-CDF rank transform :math:`R(\\mathbf x)(v)=F_{\\mathbf
x}(\\mathbf x(v))` is *exactly* invariant to the group :math:`G_+` of strictly
increasing intensity remaps, because the empirical CDF transforms covariantly:
:math:`F_{g\\circ\\mathbf x}(g(t))=F_{\\mathbf x}(t)`. Ties take the midrank, and
the same transform runs in training and inference (#2409).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_hard_rank_exactly_invariant_to_monotone_remap() -> None:
    from spectramr.models.blocks.rank_normalize import rank_normalize

    torch.manual_seed(0)
    x = torch.rand(2, 1, 8, 8)

    # A strictly-increasing nonlinear contrast remap g.
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(2.0 * t) + 0.3 * t + 5.0

    r_x = rank_normalize(x)
    r_gx = rank_normalize(g(x))
    assert torch.equal(r_x, r_gx), "hard rank must be bit-exact under monotone remap"


def test_hard_rank_in_unit_interval() -> None:
    from spectramr.models.blocks.rank_normalize import rank_normalize

    x = torch.randn(3, 2, 5, 5)
    r = rank_normalize(x)
    assert r.shape == x.shape
    assert float(r.min()) >= 0.0
    assert float(r.max()) <= 1.0


def test_rank_is_per_image_per_channel() -> None:
    """Ranks are computed within each (batch, channel) map independently."""
    from spectramr.models.blocks.rank_normalize import rank_normalize

    x = torch.zeros(2, 1, 1, 3)
    x[0, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    x[1, 0, 0] = torch.tensor([-5.0, -1.0, 0.0])
    r = rank_normalize(x)
    # Each row maps its own min->0, max->1 regardless of the other row's scale.
    assert torch.allclose(r[0, 0, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(r[1, 0, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_ties_take_the_midrank() -> None:
    from spectramr.models.blocks.rank_normalize import rank_normalize

    x = torch.tensor([0.0, 0.0, 1.0, 2.0]).reshape(1, 1, 1, 4)
    r = rank_normalize(x).flatten()
    assert torch.allclose(r, torch.tensor([0.5, 0.5, 2.0, 3.0]) / 3.0)


def test_order_reversal_is_exact_complement_with_ties() -> None:
    from spectramr.models.blocks.rank_normalize import rank_normalize

    torch.manual_seed(2)
    x = torch.zeros(2, 1, 16, 16)
    x[..., 4:12, 4:12] = torch.rand(2, 1, 8, 8)
    assert torch.allclose(rank_normalize(-x), 1.0 - rank_normalize(x), atol=1e-6)


def test_constant_map_is_one_half() -> None:
    from spectramr.models.blocks.rank_normalize import rank_normalize

    r = rank_normalize(torch.full((1, 1, 5, 5), 3.0))
    assert torch.equal(r, torch.full_like(r, 0.5))


def test_block_is_identical_in_train_and_eval() -> None:
    from spectramr.models.blocks.rank_normalize import RankNormalize2d

    x = torch.randn(2, 1, 7, 7)
    block = RankNormalize2d()
    out_train = block.train()(x)
    assert torch.equal(out_train, block.eval()(x))


def test_arm_patch_size_has_linear_memory() -> None:
    """At the arm's 4 x 192 x 192 patch the old pairwise path needed ~20 GiB per tensor."""
    from spectramr.models.blocks.rank_normalize import rank_normalize

    x = torch.rand(4, 1, 192, 192)
    r = rank_normalize(x)
    assert r.shape == x.shape
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0


def test_block_module_matches_functional() -> None:
    from spectramr.models.blocks.rank_normalize import RankNormalize2d, rank_normalize

    x = torch.randn(2, 1, 7, 7)
    block = RankNormalize2d().eval()
    assert torch.equal(block(x), rank_normalize(x))
