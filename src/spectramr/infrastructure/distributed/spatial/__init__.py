"""Spatial (depth-split) parallelism for 3D convolutional models."""

from spectramr.infrastructure.distributed.spatial.convert import (
    SpatialEquivalenceError,
    check_spatial_equivalence,
    convert_to_spatial,
    make_spatial,
)
from spectramr.infrastructure.distributed.spatial.halo import (
    exchange_halo,
    gather_depth,
    overlap_add,
    reduce_spatial_gradients,
    split_depth,
)

__all__ = [
    "SpatialEquivalenceError",
    "check_spatial_equivalence",
    "convert_to_spatial",
    "exchange_halo",
    "gather_depth",
    "make_spatial",
    "overlap_add",
    "reduce_spatial_gradients",
    "split_depth",
]
