"""Which tensors were already non-finite when the divergence guard fired.

The guard sees only a non-finite loss. A loss that is NaN at step 3 after finite
steps 1 and 2 has two readings that call for different fixes: the forward
produced the NaN from finite weights, or step 2's optimizer update wrote a
non-finite weight from a non-finite gradient. Telling them apart needs the
parameters and gradients, so the scan runs once, on the divergence path, and the
hot loop pays nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

#: Names listed per category; the count is always reported in full.
MAX_NAMES = 5


def _nonfinite_names(named: list[tuple[str, torch.Tensor | None]]) -> tuple[int, list[str]]:
    bad = [
        name
        for name, t in named
        if t is not None and t.numel() > 0 and not bool(torch.isfinite(t).all())
    ]
    return len(bad), bad[:MAX_NAMES]


def describe_nonfinite_state(model: nn.Module) -> str:
    """One line naming the non-finite parameters, gradients and buffers of ``model``."""
    params = list(model.named_parameters())
    n_p, p_names = _nonfinite_names([(n, p.detach()) for n, p in params])
    n_g, g_names = _nonfinite_names([(n, p.grad) for n, p in params])
    n_b, b_names = _nonfinite_names([(n, b.detach()) for n, b in model.named_buffers()])
    if n_p == 0 and n_g == 0 and n_b == 0:
        verdict = "weights, gradients and buffers are all finite: the forward produced the NaN"
    elif n_p == 0:
        verdict = (
            "weights are finite but gradients are not: the update that follows would poison them"
        )
    else:
        verdict = "weights are already non-finite: an earlier optimizer update wrote them"
    return (
        f"{verdict} | non-finite params={n_p} {p_names} grads={n_g} {g_names} "
        f"buffers={n_b} {b_names}"
    )


def log_divergence_state(model: nn.Module, iteration: int) -> None:
    """Log :func:`describe_nonfinite_state`; a failing scan is logged, never raised."""
    try:
        summary = describe_nonfinite_state(model)
    except Exception:
        logger.exception("[Divergence] state scan failed at iteration %d", iteration)
        return
    logger.critical("[Divergence] iteration %d: %s", iteration, summary)


__all__ = ["MAX_NAMES", "describe_nonfinite_state", "log_divergence_state"]
