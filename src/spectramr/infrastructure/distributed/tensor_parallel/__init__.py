"""Tensor parallelism (Megatron-style head and hidden splits) for the transformer blocks."""

from spectramr.infrastructure.distributed.tensor_parallel.plan import (
    MLP_PLANS,
    TensorParallelError,
    parallelize_tensor,
)

__all__ = ["MLP_PLANS", "TensorParallelError", "parallelize_tensor"]
