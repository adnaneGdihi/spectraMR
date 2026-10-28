"""Post-construction gradient-checkpointing wrapper, shared by builder and probe.

Split out of :mod:`generator_kwargs` in the Wave 0 exit-criterion work (#1400).
It never belonged there on concern: ``generator_kwargs`` answers *what kwargs
does this generator get*, while this mutates a generator that already exists.

Shared so that every path doing a real backward pass -- training, and the audit
probe -- exercises the same memory/recompute behaviour. An arm that checkpoints
in training but not under the probe is probed on a model training never builds.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_gradient_checkpointing(generator: Any, config: Any) -> None:
    """Apply gradient checkpointing to ``generator`` when the arm enables it.

    Shared so that a path doing a real backward pass -- training, and the audit
    probe -- exercises the same memory/recompute behaviour. An arm that
    checkpoints in training but not under the probe is probed on a model
    training never builds.

    Only the model knows which segments hold activations worth discarding, so a
    model without ``set_grad_checkpointing`` raises: the old generic fallback
    wrapped single Conv2d/Linear layers, saved nothing and missed Conv3d (#1333).

    Raises:
        NotImplementedError: The arm enables checkpointing and the model has no hook.
    """
    try:
        enabled = config.optimization.gradient.enable_checkpointing
    except AttributeError:
        return
    if not enabled:
        return

    if not hasattr(generator, "set_grad_checkpointing"):
        raise NotImplementedError(
            f"optimization.gradient.enable_checkpointing is true but "
            f"{type(generator).__name__} has no set_grad_checkpointing. Add "
            "GradCheckpointingMixin to the model (it decides which blocks to "
            "checkpoint), or set enable_checkpointing: false."
        )
    generator.set_grad_checkpointing(True)
    logger.info("Gradient checkpointing enabled on %s", type(generator).__name__)


__all__ = ["apply_gradient_checkpointing"]
