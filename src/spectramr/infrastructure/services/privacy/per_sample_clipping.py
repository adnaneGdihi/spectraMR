r"""Per-sample gradient clipping by microbatching, for a strategy-owned loss.

Abadi et al. 2016 bound the sensitivity of each DP-SGD update by clipping
*every example's* gradient to ``C`` before averaging. The RDP accountant
(:mod:`.rdp_accountant`) assumes exactly that mechanism, so a batch-level clip
of the averaged gradient does not deliver the ``(epsilon, delta)`` it reports.

These helpers compute the clipped per-sample gradients one example at a time
(``torch.autograd.grad`` on a batch-of-one loss) and hand the trainer a
*surrogate* loss whose value is the mean per-sample loss and whose gradient is
the mean of the clipped per-sample gradients, so the unchanged
backward/step path applies the privatised update.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def slice_sample(value: Any, index: int, batch_size: int) -> Any:
    """Return example ``index`` of a batch-shaped value, keeping the batch dim.

    Tensors whose leading dim equals ``batch_size`` are sliced, mappings are
    sliced recursively, and anything else (scalars, metadata) passes through.
    """
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, Mapping):
        return {k: slice_sample(v, index, batch_size) for k, v in value.items()}
    return value


def accumulate_clipped_gradient(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    max_grad_norm: float,
    accumulator: list[torch.Tensor],
) -> torch.Tensor:
    """Add one example's gradient, clipped to L2 norm ``max_grad_norm``, in place.

    Args:
        loss: Scalar loss of a single example.
        params: Parameters to differentiate (``requires_grad=True``).
        max_grad_norm: Clip threshold ``C`` (> 0).
        accumulator: One buffer per parameter; receives the clipped gradient.

    Returns:
        The example's pre-clip gradient L2 norm (on device, no host sync).
    """
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0; got {max_grad_norm}")
    grads = torch.autograd.grad(loss, list(params), allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params, strict=True)]
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    scale = (max_grad_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    torch._foreach_add_(accumulator, torch._foreach_mul(grads, scale))
    return norm


def gradient_surrogate(
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    value: torch.Tensor,
) -> torch.Tensor:
    """Scalar whose value is ``value`` and whose gradient w.r.t. ``params`` is ``grads``."""
    inner = torch.stack([(p * g.detach()).sum() for p, g in zip(params, grads, strict=True)]).sum()
    return inner - inner.detach() + value.detach()
