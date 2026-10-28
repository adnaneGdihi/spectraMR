"""Whether a model is a declared parameter-free solver, and what such a run may do.

A classical reconstruction (FISTA on a fixed objective) sits where a network sits so
that a strategy grades it on the same validation path, but it has nothing to
optimise and nothing to load. That is a property the model DECLARES with the class
attribute ``parameter_free = True``; it is never inferred from a parameter count,
because a network whose layers failed to register would then be graded as a
"baseline" with no checkpoint (non-negotiable 3). The two disagreements both raise.

One owner (non-negotiable 17): the optimizer builder, the validate-only path and the
run guard all ask :func:`declares_parameter_free`.
"""

from __future__ import annotations

from typing import Any

from spectramr.core.module_utils import unwrap_model

__all__ = ["declares_parameter_free", "require_parameter_free_run"]


def declares_parameter_free(model: Any) -> bool:
    """``True`` iff ``model`` declares ``parameter_free`` and indeed holds no parameters.

    Raises:
        ValueError: a declared model that holds parameters, or an undeclared one
            that holds none.
    """
    import torch.nn as nn

    module = unwrap_model(model)
    if not isinstance(module, nn.Module):
        return False
    # Read from the class, and only a literal True: a mock answers any attribute.
    declared = getattr(type(module), "parameter_free", False) is True
    count = sum(p.numel() for p in module.parameters())
    name = type(module).__name__
    if declared and count:
        raise ValueError(
            f"{name} declares parameter_free but holds {count} parameters; a solver "
            "with weights must be trained or loaded like any network."
        )
    if not declared and count == 0:
        raise ValueError(
            f"{name} holds no parameters and does not declare parameter_free. If it is "
            "a classical solver, set `parameter_free = True` on the class; if it is a "
            "network, its layers did not register."
        )
    return declared


def require_parameter_free_run(model: Any, *, validate_only: bool, resume: Any) -> bool:
    """Check what a run may do with ``model``; returns :func:`declares_parameter_free`.

    A declared parameter-free model is graded by ``--validate-only`` with no
    checkpoint. Training it, or resuming it, has no meaning and raises.
    """
    if model is None or not declares_parameter_free(model):
        return False
    name = type(unwrap_model(model)).__name__
    if not validate_only:
        raise ValueError(
            f"{name} is parameter-free: there is nothing to train. Grade it with "
            "`spectramr train --config <arm>.yaml --validate-only`."
        )
    if resume:
        raise ValueError(
            f"{name} is parameter-free and has no checkpoint to resume; drop --resume."
        )
    return True
