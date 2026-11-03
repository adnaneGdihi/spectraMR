"""``clip_grad_norm_global_`` without a tensor-parallel plan is the plain utility.

The tensor-parallel path needs a process group and is covered on four gloo ranks in
``tensor_parallel/test_fsdp_tensor_parallel_gloo.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]


def _with_grads(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    model(torch.randn(3, 4)).square().sum().backward()
    return model


@pytest.mark.parametrize("max_norm", [0.01, float("inf")])
def test_matches_clip_grad_norm_without_a_plan(max_norm: float) -> None:
    from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_

    model = _with_grads(0)
    reference = _with_grads(0)
    norm = clip_grad_norm_global_(model, max_norm)
    expected = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
    torch.testing.assert_close(norm, expected)
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_the_stage_clip_relies_on_the_schema_refusing_pipeline_with_tensor() -> None:
    """The clip sums stage norms over pp only; the schema owns refusing a tp split there."""
    from pydantic import ValidationError

    from spectramr.config.schemas.base import ParallelismConfigSchema

    with pytest.raises(ValidationError, match="pipeline"):
        ParallelismConfigSchema(strategy="ddp", mesh={"pipeline": 2, "tensor": 2})
