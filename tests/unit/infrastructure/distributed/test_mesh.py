"""Mesh shape resolution: pure, so every refusal is checked without a process group.

The multi-rank behaviour (groups, identical batches, reduces) is in
``test_mesh_gloo.py``. Each refusal below is a planted violation: the shape it
names is the one the guard must turn red on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.distributed.mesh import (
    MESH_DIM_NAMES,
    ParallelMesh,
    resolve_mesh_shape,
)

pytestmark = [pytest.mark.unit]


def _mesh(tensor=1, context=1, spatial=1, pipeline=1):
    return SimpleNamespace(
        tensor=tensor,
        context=context,
        spatial=spatial,
        pipeline=pipeline,
        non_dp_size=tensor * context * spatial * pipeline,
    )


def _shape(world, *, strategy="ddp", sharding=None, local=None, **axes):
    return resolve_mesh_shape(
        _mesh(**axes),
        strategy=strategy,
        sharding_strategy=sharding,
        world_size=world,
        local_world_size=world if local is None else local,
    )


def test_trivial_mesh_is_pure_data_parallel() -> None:
    shape = _shape(4)
    assert shape == {
        "pp": 1,
        "dp_replicate": 4,
        "dp_shard": 1,
        "cp": 1,
        "sp": 1,
        "tp": 1,
    }
    assert tuple(shape) == MESH_DIM_NAMES


def test_ddp_splits_the_data_parallel_size_off_the_model_parallel_product() -> None:
    shape = _shape(8, tensor=2, context=2)
    assert shape["dp_replicate"] * shape["dp_shard"] == 2
    assert (shape["tp"], shape["cp"]) == (2, 2)


@pytest.mark.parametrize("sharding", ["full_shard", "shard_grad_op"])
def test_fsdp_full_shard_puts_the_whole_data_parallel_size_on_dp_shard(sharding) -> None:
    shape = _shape(4, strategy="fsdp", sharding=sharding)
    assert (shape["dp_replicate"], shape["dp_shard"]) == (1, 4)


def test_fsdp_no_shard_replicates_the_whole_data_parallel_size() -> None:
    shape = _shape(4, strategy="fsdp", sharding="no_shard")
    assert (shape["dp_replicate"], shape["dp_shard"]) == (4, 1)


def test_fsdp_hybrid_shards_within_one_node() -> None:
    shape = _shape(4, strategy="fsdp", sharding="hybrid_shard", local=2)
    assert (shape["dp_replicate"], shape["dp_shard"]) == (2, 2)


def test_hybrid_shard_uses_the_model_parallel_product_inside_the_node() -> None:
    # 2 ranks per node, tp=2: one data shard per node, replicated across nodes.
    shape = _shape(4, strategy="fsdp", sharding="hybrid_shard", local=2, tensor=2)
    assert (shape["dp_replicate"], shape["dp_shard"]) == (2, 1)


# --- planted violations: each must refuse -----------------------------------------


def test_refuses_a_model_parallel_product_that_does_not_divide_the_world() -> None:
    with pytest.raises(ValueError, match="does not divide the world size"):
        _shape(6, tensor=4)


def test_refuses_hybrid_shard_when_the_model_parallel_product_exceeds_a_node() -> None:
    with pytest.raises(ValueError, match="ranks per node"):
        _shape(4, strategy="fsdp", sharding="hybrid_shard", local=2, tensor=4)


def test_refuses_hybrid_shard_when_the_replica_count_is_not_whole() -> None:
    # 6 ranks, 4 per node: shard size 4 does not divide the data-parallel size 6.
    with pytest.raises(ValueError, match="not a multiple of the per-node shard"):
        _shape(6, strategy="fsdp", sharding="hybrid_shard", local=4)


def test_refuses_an_unknown_fsdp_sharding_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown FSDP sharding_strategy"):
        _shape(4, strategy="fsdp", sharding="zero3")


def test_provenance_records_shape_and_dim_names() -> None:
    layout = SimpleNamespace(world_size=2)
    mesh = ParallelMesh(device_mesh=None, layout=layout, shape=_shape(4, tensor=2))
    record = mesh.provenance()
    assert record["dim_names"] == list(MESH_DIM_NAMES)
    assert record["shape"]["tp"] == 2
    assert record["dp_world_size"] == 2


def test_build_refuses_without_a_process_group() -> None:
    from spectramr.infrastructure.distributed.mesh import build_parallel_mesh

    parallel = SimpleNamespace(strategy="ddp", mesh=_mesh(tensor=2))
    with pytest.raises(RuntimeError, match="initialised process group"):
        build_parallel_mesh(parallel, device_type="cpu", world_size=4, local_world_size=4)
