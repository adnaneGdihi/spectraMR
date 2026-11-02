"""Readout-order pooling of off-grid k-space nodes, for a backbone with a node cap.

A trajectory is stored arm by arm, so consecutive samples are neighbours along one readout.
Pooling runs of ``stride`` consecutive samples is therefore the off-grid analogue of
average-pooling an image: deterministic, local and free of any interpolation onto a grid.
Unpooling copies each group's output back to its members (``out[:, groups]``), the analogue
of upsampling.
"""

from __future__ import annotations

import math

import torch

__all__ = ["pool_nodes", "readout_groups"]


def readout_groups(num_nodes: int, max_nodes: int, device: torch.device) -> torch.Tensor | None:
    """Group index of every node when ``num_nodes`` exceeds ``max_nodes``, else ``None``.

    Groups are runs of ``ceil(num_nodes / max_nodes)`` consecutive nodes, so at most
    ``max_nodes`` groups remain.
    """
    if max_nodes < 1:
        raise ValueError(f"max_nodes must be a positive integer; got {max_nodes}.")
    if num_nodes <= max_nodes:
        return None
    stride = math.ceil(num_nodes / max_nodes)
    return torch.arange(num_nodes, device=device) // stride


def pool_nodes(nodes: torch.Tensor, groups: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Mean of every feature over each group: [B, N, F] -> [B, G, F].

    The group count is derived from the shapes, not read from ``groups``, so pooling never
    synchronises with the device.
    """
    num_groups = math.ceil(nodes.shape[1] / math.ceil(nodes.shape[1] / max_nodes))
    pooled = nodes.new_zeros(nodes.shape[0], num_groups, nodes.shape[-1])
    pooled.index_add_(1, groups, nodes)
    counts = nodes.new_zeros(num_groups).index_add_(0, groups, nodes.new_ones(groups.shape[0]))
    return pooled / counts.view(1, -1, 1)
