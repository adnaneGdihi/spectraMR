"""The one DeviceMesh of a run, and the data-parallel group read from it.

Every rank builds the same named mesh, once, before any model is wrapped:

    ("pp", "dp_replicate", "dp_shard", "cp", "sp", "tp")

Sizes default to 1, so a plain DDP or FSDP run is a mesh whose only non-trivial axes
are the two data-parallel ones. ``dp_replicate * dp_shard`` is the data-parallel size
``world // (pp * cp * sp * tp)``. Its split reproduces the FSDP ``sharding_strategy``
semantics (full/shard_grad_op: all shard; no_shard: all replicate; hybrid: shard
within a node). FSDP takes its shard mesh from this one (``fsdp_wrap.fsdp_shard_mesh``).

The read side (``dp_group``, ``dp_rank``, ``dp_world_size``, ``is_dp_writer``) lives in
``spectramr.core.parallel_groups`` so core metrics and models can use it; they are
re-exported here as the public surface of the mesh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectramr.core.parallel_groups import (
    DPLayout,
    activate_dp_layout,
    dp_group,
    dp_rank,
    dp_src_rank,
    dp_world_size,
    is_dp_writer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_PARALLEL_DIMS",
    "MESH_DIM_NAMES",
    "ParallelMesh",
    "build_parallel_mesh",
    "dp_group",
    "dp_rank",
    "dp_src_rank",
    "dp_world_size",
    "is_dp_writer",
    "resolve_mesh_shape",
]

#: Dimension order of the one mesh. Row-major: ``pp`` is the slowest-varying axis.
MESH_DIM_NAMES: tuple[str, ...] = ("pp", "dp_replicate", "dp_shard", "cp", "sp", "tp")

#: The two axes whose product is the data-parallel size. Adjacent, so their slice is
#: a contiguous sub-mesh that can be flattened into one group.
DATA_PARALLEL_DIMS: tuple[str, ...] = ("dp_replicate", "dp_shard")

_FSDP_SHARDING = ("full_shard", "shard_grad_op", "no_shard", "hybrid_shard")


def resolve_mesh_shape(
    mesh_cfg: Any,
    *,
    strategy: str,
    sharding_strategy: str | None,
    world_size: int,
    local_world_size: int,
) -> dict[str, int]:
    """Sizes of every mesh dimension, from the config and the world size alone.

    Pure, so every rank computes the same shape and raises the same error before any
    group is created. Raises when the model-parallel product does not divide the
    world size, or when hybrid sharding cannot fit inside one node.
    """
    non_dp = mesh_cfg.non_dp_size
    if world_size % non_dp:
        raise ValueError(
            f"parallel.mesh (tensor={mesh_cfg.tensor}, context={mesh_cfg.context}, "
            f"spatial={mesh_cfg.spatial}, pipeline={mesh_cfg.pipeline}) has a model-parallel "
            f"product of {non_dp}, which does not divide the world size {world_size}. The "
            "data-parallel size would be fractional."
        )
    dp = world_size // non_dp

    if strategy == "fsdp" and sharding_strategy in ("full_shard", "shard_grad_op"):
        replicate, shard = 1, dp
    elif strategy == "fsdp" and sharding_strategy == "hybrid_shard":
        if local_world_size < 1 or local_world_size % non_dp:
            raise ValueError(
                f"fsdp.sharding_strategy='hybrid_shard' needs the ranks per node "
                f"({local_world_size}) to be a multiple of the model-parallel product "
                f"({non_dp})."
            )
        shard = local_world_size // non_dp
        if dp % shard:
            raise ValueError(
                f"hybrid_shard: the data-parallel size {dp} is not a multiple of the "
                f"per-node shard size {shard}."
            )
        replicate = dp // shard
    elif strategy == "fsdp" and sharding_strategy == "no_shard":
        replicate, shard = dp, 1
    elif strategy == "fsdp":
        raise ValueError(
            f"Unknown FSDP sharding_strategy={sharding_strategy!r}. Use one of {_FSDP_SHARDING}."
        )
    else:
        # ddp: every data-parallel rank holds a full replica.
        replicate, shard = dp, 1

    return {
        "pp": mesh_cfg.pipeline,
        "dp_replicate": replicate,
        "dp_shard": shard,
        "cp": mesh_cfg.context,
        "sp": mesh_cfg.spatial,
        "tp": mesh_cfg.tensor,
    }


@dataclass(frozen=True)
class ParallelMesh:
    """The built mesh, plus this rank's data-parallel layout."""

    device_mesh: Any
    layout: DPLayout
    shape: dict[str, int]
    #: The ``sp`` peers of this rank (depth slabs of one volume); ``None`` unless sp > 1.
    spatial_group: Any = None
    #: The ``dp x sp`` group DDP averages gradients over; ``None`` unless sp > 1, so
    #: the data-parallel group is used as before.
    grad_group: Any = None

    def provenance(self) -> dict[str, Any]:
        """Rank-independent record of the mesh. Per-rank coordinates go in the distributed probe."""
        return {
            "shape": dict(self.shape),
            "dim_names": list(MESH_DIM_NAMES),
            "dp_world_size": self.layout.world_size,
        }


def build_parallel_mesh(
    parallel_cfg: Any,
    *,
    device_type: str,
    world_size: int | None = None,
    local_world_size: int | None = None,
) -> ParallelMesh:
    """Create the run's DeviceMesh and activate its data-parallel layout.

    Must be called on every rank, with the process group initialised. Group creation
    is collective, so every rank builds the same mesh in the same order. Activation is
    process-wide: a second call replaces the first.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "parallel.mesh needs an initialised process group. Launch with torchrun "
            "or train-distributed; refusing to build a mesh for a single process."
        )
    mesh_cfg = parallel_cfg.mesh
    strategy = parallel_cfg.strategy
    sharding = parallel_cfg.fsdp.sharding_strategy if strategy == "fsdp" else None
    world = dist.get_world_size() if world_size is None else world_size
    if local_world_size is None:
        from spectramr.core.topology import resolve_run_topology

        local_world_size = resolve_run_topology().local_world_size

    shape = resolve_mesh_shape(
        mesh_cfg,
        strategy=strategy,
        sharding_strategy=sharding,
        world_size=world,
        local_world_size=local_world_size,
    )
    names = MESH_DIM_NAMES
    device_mesh = init_device_mesh(
        device_type, tuple(shape[n] for n in names), mesh_dim_names=names
    )

    coord = device_mesh.get_coordinate()
    index = {name: i for i, name in enumerate(names)}
    dp_size = shape["dp_replicate"] * shape["dp_shard"]
    rank_in_dp = coord[index["dp_replicate"]] * shape["dp_shard"] + coord[index["dp_shard"]]
    model_parallel = [n for n in names if n not in DATA_PARALLEL_DIMS]
    writer = all(coord[index[n]] == 0 for n in model_parallel)
    # Global rank of the data-parallel index-0 peer: the row-major root of this rank's group.
    root_idx = tuple(0 if n in DATA_PARALLEL_DIMS else coord[index[n]] for n in names)
    src_rank = int(device_mesh.mesh[root_idx])

    if all(shape[n] == 1 for n in model_parallel):
        # Trivial mesh: the data-parallel group IS the world group. Keep the default
        # group so nothing new is created and every call matches today's path.
        group = None
    else:
        group = device_mesh[DATA_PARALLEL_DIMS]._flatten("dp").get_group()

    spatial_group = device_mesh["sp"].get_group() if shape["sp"] > 1 else None
    grad_group = None
    if shape["sp"] > 1:
        # cp is size 1 here (refused with sp > 1 by the schema), so the slice is contiguous.
        grad_group = (
            device_mesh[("dp_replicate", "dp_shard", "cp", "sp")]._flatten("dp_sp").get_group()
        )

    layout = DPLayout(
        rank=rank_in_dp,
        world_size=dp_size,
        group=group,
        src_rank=src_rank,
        writer=writer,
        dim_names=names,
        shape=tuple(shape[n] for n in names),
    )
    activate_dp_layout(layout)
    logger.info(
        "[Parallel] DeviceMesh %s dims=%s dp_rank=%d dp_world=%d writer=%s",
        layout.shape,
        names,
        rank_in_dp,
        dp_size,
        writer,
    )
    return ParallelMesh(
        device_mesh=device_mesh,
        layout=layout,
        shape=shape,
        spatial_group=spatial_group,
        grad_group=grad_group,
    )
