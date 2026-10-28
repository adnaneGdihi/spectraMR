"""Pre-backward autograd-graph guard.

One owner for a single invariant: **a loss handed to ``backward()`` must be
connected to the autograd graph.**

``loss.grad_fn is None`` means ``loss`` is a *leaf*. ``backward()`` on a leaf
returns without error, every ``param.grad`` stays ``None``, ``optimizer.step()``
moves nothing, and the run completes and checkpoints having learned nothing --
no exception, no warning, a loss curve that looks merely flat (#1952).

Two disjoint producers reach this seam:

* **A computer returned a total with ``requires_grad=False``.** Until 2026-09-09
  ``LossOutput.__post_init__`` silently "repaired" this with
  ``.detach().requires_grad_(True)``, which *manufactures* the leaf rather than
  restoring the graph. With that repair deleted, such a total raises inside
  ``backward()`` on its own -- so it is not what this guard is for.
* **A guard-``return`` handed back ``torch.tensor(0.0, requires_grad=True)``.**
  This shape already carries ``requires_grad=True``, so nothing downstream ever
  questioned it. ``BaseLossComputer._stack_components`` returns it for empty
  ``components`` *and* when every component was skipped as NaN/Inf. This is the
  shape the guard exists to catch.

Hence the predicate is ``grad_fn``, not ``requires_grad``.

**Ordering matters twice.** The check must run *before* ``scaler.scale(loss)``
and before any gradient-accumulation division: both wrap the leaf in a
``MulBackward`` node, giving it a ``grad_fn`` and hiding the defect. It must also
run *after* the non-finite loss guard, so a loss that is both non-finite and
severed still reports the numerically actionable failure first.

The guard is backend-independent by design. ``DeepSpeedStepPolicy.guard_loss``
legitimately no-ops the *non-finite* check (DeepSpeed does its own overflow
detect and step skip), but no backend can make a severed graph correct, so this
check is never delegated to a policy object.
"""

from __future__ import annotations

import torch

__all__ = ["ensure_backward_ready"]


def ensure_backward_ready(loss: torch.Tensor, *, name: str, global_step: int) -> None:
    """Raise if ``loss`` carries no autograd graph.

    Args:
        loss: The scalar loss about to be sent to ``backward()``.
        name: Step-config name (e.g. ``"gen"`` / ``"disc"``) for the message.
        global_step: Step index, for the message.

    Raises:
        RuntimeError: ``loss.grad_fn is None`` -- backward would be a silent
            no-op.
    """
    if loss.grad_fn is not None:
        return
    raise RuntimeError(
        f"Loss for config {name!r} at step {global_step} is not connected to the "
        f"autograd graph (grad_fn is None, requires_grad={loss.requires_grad}); "
        "refusing to run a backward pass that would silently update nothing. "
        "Usual causes: a loss computer returned a guard value such as "
        "torch.tensor(0.0, requires_grad=True) because it had no components to "
        "stack, or every component was skipped as NaN/Inf; or the prediction was "
        "produced under torch.no_grad(). Fix the computer -- do NOT re-attach "
        "with .detach().requires_grad_(True), which manufactures this same leaf."
    )
