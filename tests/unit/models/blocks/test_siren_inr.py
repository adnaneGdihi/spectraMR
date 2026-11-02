"""``siren_inr``: the shared k-space coordinate convention and the SIREN block."""

from __future__ import annotations

import torch

from spectramr.models.blocks.siren_inr import SIRENNetwork, kspace_coordinate_grid


def test_the_grid_is_row_major_kx_ky_on_the_unit_square() -> None:
    grid = kspace_coordinate_grid(3, 5, torch.device("cpu"))
    assert grid.shape == (15, 2)
    assert torch.equal(grid[0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(grid[4], torch.tensor([1.0, -1.0]))  # k_x runs along a row
    assert torch.equal(grid[-1], torch.tensor([1.0, 1.0]))


def test_the_network_maps_coordinates_to_re_im() -> None:
    net = SIRENNetwork(in_features=2, hidden_features=8, hidden_layers=1, out_features=2)
    out = net(kspace_coordinate_grid(4, 4, torch.device("cpu")))
    assert out.shape == (16, 2) and torch.isfinite(out).all()
