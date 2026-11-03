"""Tensor parallelism (Megatron-style head and hidden splits) for the transformer blocks."""

from spectramr.infrastructure.distributed.tensor_parallel.layers import (
    tensor_parallel_local_state,
)
from spectramr.infrastructure.distributed.tensor_parallel.plan import (
    MLP_PLANS,
    TP_PLAN_ATTR,
    TensorParallelError,
    TensorParallelPlan,
    parallelize_tensor,
    split_like,
)

__all__ = [
    "MLP_PLANS",
    "TP_PLAN_ATTR",
    "TensorParallelError",
    "TensorParallelPlan",
    "parallelize_tensor",
    "split_like",
    "tensor_parallel_local_state",
]
