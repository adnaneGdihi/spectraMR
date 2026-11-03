"""BeltramiSFCBlock: the straight-through soft permutation behind #2360."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock
from spectramr.models.generators.conformal_geomamba_ulf import ConformalGeoMambaULF


def _head_params_with_grad(block: BeltramiSFCBlock) -> int:
    return sum(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in block.mu_head.parameters()
    )


def test_soft_perm_is_row_stochastic_and_peaks_on_the_hard_order() -> None:
    torch.manual_seed(0)
    block = BeltramiSFCBlock(grid_size=8, in_channels=1)
    out = block(torch.randn(2, 1, 8, 8))
    soft = out["soft_perm"]
    assert soft.shape == (2, 64, 64)
    torch.testing.assert_close(soft.sum(-1), torch.ones(2, 64))
    # At μ ≈ 0 each slot's heaviest cell is the cell the hard order puts there.
    zero = BeltramiSFCBlock(grid_size=8, in_channels=1)
    for p in zero.mu_head.parameters():
        torch.nn.init.zeros_(p)
    out0 = zero(torch.randn(1, 1, 8, 8))
    assert torch.equal(out0["soft_perm"].argmax(-1), out0["indices"])


def test_reconstruction_loss_alone_reaches_the_mu_head() -> None:
    """#2360 regression: with the regularisers excluded, all μ-head params get gradient."""
    torch.manual_seed(0)
    block = BeltramiSFCBlock(grid_size=8, in_channels=1)
    gen = ConformalGeoMambaULF(in_channels=1, out_channels=1, base_width=4, scales=1, grid_size=8)
    out = block(torch.randn(2, 1, 8, 8))
    y = gen(torch.randn(2, 1, 16, 16), sfc_indices=out["indices"], sfc_soft_perm=out["soft_perm"])
    y.pow(2).mean().backward()
    assert _head_params_with_grad(block) == 4


def test_straight_through_leaves_the_forward_unchanged() -> None:
    torch.manual_seed(0)
    block = BeltramiSFCBlock(grid_size=8, in_channels=1)
    gen = ConformalGeoMambaULF(in_channels=1, out_channels=1, base_width=4, scales=1, grid_size=8)
    x = torch.randn(2, 1, 16, 16)
    out = block(torch.randn(2, 1, 8, 8))
    hard = gen(x, sfc_indices=out["indices"])
    st = gen(x, sfc_indices=out["indices"], sfc_soft_perm=out["soft_perm"])
    torch.testing.assert_close(st, hard)


def test_non_positive_temperature_raises() -> None:
    with pytest.raises(ValueError, match="soft_temperature"):
        BeltramiSFCBlock(grid_size=8, soft_temperature=0.0)
