"""Compile repeated blocks instead of the whole model.

Full compilation hands Inductor one large problem and pays the whole cost at
cold start. A model built from *n* copies of one block class is mostly the same
problem *n* times, so compiling each child of a uniform ``nn.ModuleList``
separately hits the compiler cache after the first and cuts cold-start time
sharply — the idea is HuggingFace Accelerate's ``compile_regions``, ported
rather than copied.

Reach, measured rather than asserted: of 877 model files, 187 mention
``nn.ModuleList`` but only **31** build one from a ``range`` comprehension --
the shape that qualifies -- while **319** stack blocks in ``nn.Sequential``,
which :func:`is_repeated_blocks` never matches by construction. So most
block-stacked models here get the refusal, and ``regional`` is an opt-in for
the minority that fits, not a general accelerator.

**The state-dict consequence is the sharp edge.** Wrapping a child rather than
the root puts the marker mid-path — ``blocks.0._orig_mod.conv.weight`` — which
a leading-prefix strip leaves in the checkpoint. ``core.module_utils`` handles
that now, and regional compilation must not ship without it.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "NoRegionsToCompileError",
    "compile_regions",
    "has_repeated_blocks",
    "is_repeated_blocks",
]


class NoRegionsToCompileError(RuntimeError):
    """``regional`` was set on a model that exposes no qualifying block list.

    Its own type, because the caller must not re-frame it. Wrapped as a generic
    compile failure it acquired the wrong closing instruction -- "set
    compile.enabled: false" -- when the fix is ``regional: false``. A
    ``RuntimeError`` subclass so callers catching the base class still do.
    """

#: Leaf layers that are never worth compiling on their own. A ``ModuleList`` of
#: these is uniform by the class test but shares no meaningful graph between
#: children, so compiling each would pay *n* compilations for no cache reuse --
#: e.g. ``nn.ModuleList([nn.AvgPool2d(2**i) for i in range(3)])``.
_LEAF_TYPES: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose2d,
    nn.Linear,
    nn.LayerNorm,
    nn.BatchNorm2d,
    nn.GroupNorm,
    nn.Embedding,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.Identity,
    nn.Dropout,
)

#: Minimum children for the cache to pay for itself. A one-element list is just
#: the module, and compiling it regionally is strictly worse than compiling it.
_MIN_BLOCKS = 2


def is_repeated_blocks(module: nn.Module) -> bool:
    """Whether *module* is a ``ModuleList`` of two-or-more identical-class blocks.

    The class identity is what matters, not the parameters: dynamo's cache keys
    on the code object, so ``n`` instances of one class compile once and reuse
    ``n-1`` times, while a mixed list would compile each shape separately and
    gain nothing.
    """
    if not isinstance(module, nn.ModuleList) or len(module) < _MIN_BLOCKS:
        return False
    classes = {type(child) for child in module}
    if len(classes) != 1:
        return False
    return not issubclass(next(iter(classes)), _LEAF_TYPES)


def has_repeated_blocks(module: nn.Module) -> bool:
    """Whether *module* contains a qualifying block list anywhere beneath it."""
    return any(is_repeated_blocks(child) for child in module.modules())


def _outermost_block_lists(root: nn.Module) -> list[tuple[str, nn.ModuleList]]:
    """Qualifying lists, without descending into one already claimed.

    ``named_modules()`` yields a parent before its children, so compiling on
    that walk then descends into the ``OptimizedModule`` it has just created
    and compiles the blocks inside it as well. Measured on a two-stage model
    whose stages each stack three layers: **8** compilations where 2 were
    intended, every key doubly wrapped
    (``stages.0._orig_mod.layers.0._orig_mod.conv.weight``). That is nested
    ``torch.compile`` -- the outer graph traces an inner compiled module -- and
    it destroys the cache reuse the whole idea rests on.

    The outermost list wins: its blocks are the largest repeated unit, and the
    inner lists are traced into their graph exactly once each.
    """
    found: list[tuple[str, nn.ModuleList]] = []
    stack: list[tuple[str, nn.Module]] = [("", root)]
    while stack:
        name, module = stack.pop()
        if is_repeated_blocks(module):
            found.append((name, module))  # claimed; its children are the blocks
            continue
        for child_name, child in module.named_children():
            stack.append((f"{name}.{child_name}" if name else child_name, child))
    return found


def compile_regions(module: nn.Module, **compile_kwargs: Any) -> int:
    """Compile each child of every qualifying ``ModuleList``, in place.

    The parent is deliberately left uncompiled: the point is to avoid handing
    the compiler the whole model, and wrapping the root as well would reinstate
    exactly that cost.

    Args:
        module: the model to rewrite in place.
        **compile_kwargs: forwarded verbatim to ``torch.compile``.

    Returns:
        How many blocks were compiled.

    Raises:
        NoRegionsToCompileError: nothing qualified. An arm that asked for regional
            compilation and silently got whole-model compilation -- or none --
            would be reporting a configuration it did not run.
    """
    compiled = 0
    # Resolved BEFORE any mutation: the walk must see the original tree, not
    # the wrappers it is about to install.
    for name, child in _outermost_block_lists(module):
        block_type = type(child[0]).__name__
        for index in range(len(child)):
            child[index] = torch.compile(child[index], **compile_kwargs)
            compiled += 1
        logger.info(
            "Regionally compiled %d block(s) of %s at %r",
            len(child),
            block_type,
            name or "<root>",
        )

    if compiled == 0:
        raise NoRegionsToCompileError(
            "optimization.compile.regional is set, but this model exposes no "
            "nn.ModuleList of two or more identical non-leaf blocks, so there is "
            "nothing to compile regionally. Set regional: false to compile the "
            "whole model instead -- silently doing that for you would report a "
            "configuration this run did not use."
        )
    return compiled
