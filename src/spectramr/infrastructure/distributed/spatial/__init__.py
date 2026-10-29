"""Spatial (depth-split) parallelism for 3D convolutional models."""

from spectramr.infrastructure.distributed.spatial.convert import (
    SpatialEquivalenceError,
    SpatialReport,
    check_spatial_equivalence,
    convert_to_spatial,
    make_spatial,
)
from spectramr.infrastructure.distributed.spatial.halo import (
    exchange_halo,
    gather_depth,
    overlap_add,
    reduce_spatial_gradients,
    spatial_scope,
    split_depth,
)
from spectramr.infrastructure.distributed.spatial.wrapper import SpatialModel, wrap_spatial

__all__ = [
    "SpatialEquivalenceError",
    "SpatialModel",
    "SpatialReport",
    "check_spatial_equivalence",
    "convert_to_spatial",
    "exchange_halo",
    "gather_depth",
    "make_spatial",
    "overlap_add",
    "reduce_spatial_gradients",
    "spatial_scope",
    "split_depth",
    "wrap_spatial",
]
