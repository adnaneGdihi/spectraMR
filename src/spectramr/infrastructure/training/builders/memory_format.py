"""``optimization.compile.channels_last``: convert named models to the NHWC layout.

cuDNN and oneDNN run 2-D convolutions fastest in channels-last, and the layout is
chosen by the weights, so converting the module is enough. The knob names its targets
(the ``apply_to`` vocabulary) because a GAN may pair an image-domain model with one that
works on k-space, where the layout buys nothing and complex tensors are involved.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["apply_channels_last"]


def apply_channels_last(models: dict[str, nn.Module], config: Any) -> dict[str, dict[str, int]]:
    """Convert the models ``optimization.compile.channels_last`` names; return what changed.

    Raises on a name this arm does not build, and on a model with no 4-D real weight to
    convert: declaring the layout while changing nothing would report a configuration
    this run did not use.
    """
    names = getattr(getattr(config.optimization, "compile", None), "channels_last", None)
    if not names:
        return {}
    built = {name for name, model in models.items() if model is not None}
    unknown = sorted(set(names) - built)
    if unknown:
        raise RuntimeError(
            f"optimization.compile.channels_last names {unknown}, which this arm does not "
            f"build (it has {sorted(built)})."
        )
    record: dict[str, dict[str, int]] = {}
    for name in names:
        model = models[name]
        model.to(memory_format=torch.channels_last)
        converted = _count_channels_last(model)
        if converted == 0:
            raise RuntimeError(
                f"optimization.compile.channels_last names {name!r}, but it has no 4-D real "
                "weight to convert (a k-space, 1-D or 3-D model). Drop it from the list."
            )
        record[name] = {"parameters_converted": converted}
        logger.info("channels_last: %s (%d parameters converted)", name, converted)
    return record


def _count_channels_last(model: nn.Module) -> int:
    """4-D real parameters now laid out channels-last."""
    return sum(
        1
        for p in model.parameters()
        if p.ndim == 4 and not p.is_complex() and p.is_contiguous(memory_format=torch.channels_last)
    )
