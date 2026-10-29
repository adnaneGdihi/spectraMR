"""``parallel.mesh``: the schema refuses what no launch could run."""

from __future__ import annotations

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


def test_spatial_runs_under_ddp_with_a_probe_shape() -> None:
    parallel = ParallelismConfigSchema(
        strategy="ddp", mesh={"spatial": 2}, spatial_probe_shape=[16, 8, 8]
    )
    assert parallel.spatial_probe_shape == [16, 8, 8]


def test_refuses_spatial_under_fsdp() -> None:
    with pytest.raises(ValidationError, match=r"needs parallel\.strategy='ddp'"):
        ParallelismConfigSchema(
            strategy="fsdp", fsdp=FSDPConfigSchema(enabled=True), mesh={"spatial": 2}
        )


@pytest.mark.parametrize("axes", [{"context": 2}, {"tensor": 2}, {"pipeline": 2}])
def test_refuses_spatial_combined_with_another_model_axis(axes) -> None:
    with pytest.raises(ValidationError, match="cannot be combined"):
        ParallelismConfigSchema(strategy="ddp", mesh={"spatial": 2, **axes})


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
