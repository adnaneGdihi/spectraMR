"""``parallel.mesh``: the schema refuses what no launch could run."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.base import (
    DeepSpeedConfigSchema,
    FSDPConfigSchema,
    ParallelismConfigSchema,
)

pytestmark = [pytest.mark.unit]


def test_every_axis_defaults_to_one() -> None:
    mesh = ParallelismConfigSchema().mesh
    assert (mesh.tensor, mesh.context, mesh.spatial, mesh.pipeline) == (1, 1, 1, 1)
    assert mesh.non_dp_size == 1


def test_ddp_may_split_a_tensor_axis() -> None:
    parallel = ParallelismConfigSchema(strategy="ddp", mesh={"tensor": 2})
    assert parallel.mesh.tensor == 2


def test_fsdp_may_split_a_context_axis() -> None:
    parallel = ParallelismConfigSchema(
        strategy="fsdp", fsdp=FSDPConfigSchema(enabled=True), mesh={"context": 2}
    )
    assert parallel.mesh.non_dp_size == 2


@pytest.mark.parametrize("strategy", ["none", "dp"])
def test_refuses_a_split_mesh_for_strategies_with_no_mesh(strategy) -> None:
    with pytest.raises(ValidationError, match=r"parallel\.mesh"):
        ParallelismConfigSchema(strategy=strategy, mesh={"tensor": 2})


def test_refuses_a_split_mesh_under_deepspeed() -> None:
    with pytest.raises(ValidationError, match=r"parallel\.mesh"):
        ParallelismConfigSchema(
            strategy="deepspeed",
            deepspeed=DeepSpeedConfigSchema(enabled=True),
            mesh={"spatial": 2},
        )


@pytest.mark.parametrize("axis", ["tensor", "context", "spatial", "pipeline"])
def test_refuses_a_non_positive_axis(axis) -> None:
    with pytest.raises(ValidationError):
        ParallelismConfigSchema(strategy="ddp", mesh={axis: 0})


def test_refuses_an_unknown_mesh_key() -> None:
    """A typo'd axis must raise, not vanish into a single-process run (pitfall 15)."""
    with pytest.raises(ValidationError):
        ParallelismConfigSchema(strategy="ddp", mesh={"tensr": 2})


def test_pipeline_microbatches_default_to_one_and_accept_a_pipeline() -> None:
    assert ParallelismConfigSchema().mesh.pipeline_microbatches == 1
    mesh = ParallelismConfigSchema(
        strategy="ddp", mesh={"pipeline": 2, "pipeline_microbatches": 4}
    ).mesh
    assert (mesh.pipeline, mesh.pipeline_microbatches) == (2, 4)


def test_refuses_microbatches_without_a_pipeline() -> None:
    """An inert knob is refused rather than stamped as if it did something (pitfall 15)."""
    with pytest.raises(ValidationError, match="no pipeline to fill"):
        ParallelismConfigSchema(strategy="ddp", mesh={"pipeline_microbatches": 2})


def test_spatial_runs_under_ddp_with_a_probe_shape() -> None:
    parallel = ParallelismConfigSchema(
        strategy="ddp", mesh={"spatial": 2}, spatial_probe_shape=[16, 8, 8]
    )
    assert parallel.spatial_probe_shape == [16, 8, 8]


@pytest.mark.parametrize("sharding", ["full_shard", "shard_grad_op", "hybrid_shard"])
def test_spatial_runs_under_sharding_fsdp(sharding) -> None:
    parallel = ParallelismConfigSchema(
        strategy="fsdp",
        fsdp=FSDPConfigSchema(enabled=True, sharding_strategy=sharding),
        mesh={"spatial": 2},
    )
    assert parallel.mesh.spatial == 2


def test_refuses_spatial_under_fsdp_no_shard() -> None:
    with pytest.raises(ValidationError, match="no_shard"):
        ParallelismConfigSchema(
            strategy="fsdp",
            fsdp=FSDPConfigSchema(enabled=True, sharding_strategy="no_shard"),
            mesh={"spatial": 2},
        )


@pytest.mark.parametrize("strategy", ["none", "dp", "deepspeed"])
def test_refuses_spatial_outside_ddp_and_fsdp(strategy) -> None:
    extra = {"deepspeed": {"enabled": True}} if strategy == "deepspeed" else {}
    with pytest.raises(ValidationError, match="axes of size"):
        ParallelismConfigSchema(strategy=strategy, mesh={"spatial": 2}, **extra)


_REFUSED_PAIRS = {
    ("tensor", "spatial"): "No 3-D model carries TokenAttention",
    ("context", "spatial"): "No 3-D model has a context-parallel plan",
    ("tensor", "pipeline"): "a tensor axis would split nothing",
    ("context", "pipeline"): "a context axis would split nothing",
    ("spatial", "pipeline"): "spatial parallelism needs a 3-D volume",
}


def test_every_pair_but_tensor_context_has_a_refusal() -> None:
    pairs = set(itertools.combinations(["tensor", "context", "spatial", "pipeline"], 2))
    assert pairs - set(_REFUSED_PAIRS) == {("tensor", "context")}


@pytest.mark.parametrize(("pair", "reason"), list(_REFUSED_PAIRS.items()))
def test_refuses_two_model_parallel_axes_with_the_pair_reason(pair, reason) -> None:
    with pytest.raises(ValidationError, match=rf"combines {pair[0]}=2, {pair[1]}=2: .*{reason}"):
        ParallelismConfigSchema(strategy="ddp", mesh=dict.fromkeys(pair, 2))


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_tensor_composes_with_context(strategy) -> None:
    extra = {"fsdp": FSDPConfigSchema(enabled=True)} if strategy == "fsdp" else {}
    parallel = ParallelismConfigSchema(
        strategy=strategy, mesh={"tensor": 2, "context": 2}, **extra
    )
    assert parallel.mesh.non_dp_size == 4


def test_a_third_axis_beside_tensor_context_is_refused() -> None:
    with pytest.raises(ValidationError, match="tensor=2, spatial=2"):
        ParallelismConfigSchema(strategy="ddp", mesh={"tensor": 2, "context": 2, "spatial": 2})


def test_refuses_spatial_with_sync_batch_norm() -> None:
    with pytest.raises(ValidationError, match="sync_batch_norm"):
        ParallelismConfigSchema(strategy="ddp", mesh={"spatial": 2}, sync_batch_norm=True)


@pytest.mark.parametrize("shape", [[16, 8], [15, 8, 8], [16, 0, 8]])
def test_refuses_a_malformed_or_indivisible_probe_shape(shape) -> None:
    with pytest.raises(ValidationError, match=r"spatial_probe_shape"):
        ParallelismConfigSchema(strategy="ddp", mesh={"spatial": 2}, spatial_probe_shape=shape)


def test_refuses_a_probe_shape_without_spatial() -> None:
    with pytest.raises(ValidationError, match="read only when"):
        ParallelismConfigSchema(strategy="ddp", spatial_probe_shape=[16, 8, 8])
