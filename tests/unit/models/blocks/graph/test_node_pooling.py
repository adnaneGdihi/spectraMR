"""Readout-order pooling of off-grid nodes."""

import pytest
import torch

from spectramr.models.blocks.graph.node_pooling import pool_nodes, readout_groups


def test_no_groups_under_the_cap():
    assert readout_groups(10, 10, torch.device("cpu")) is None


def test_groups_are_consecutive_runs_and_pool_to_their_mean():
    groups = readout_groups(10, 4, torch.device("cpu"))
    assert groups.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    nodes = torch.arange(10.0).view(1, 10, 1)
    pooled = pool_nodes(nodes, groups, 4)
    assert pooled.view(-1).tolist() == [1.0, 4.0, 7.0, 9.0]
    assert pooled[:, groups].shape == nodes.shape


def test_a_non_positive_cap_raises():
    with pytest.raises(ValueError, match="max_nodes"):
        readout_groups(10, 0, torch.device("cpu"))
