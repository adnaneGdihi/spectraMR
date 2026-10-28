"""Keep chosen functions out of every ``torch.compile`` graph.

Inductor cannot generate code for complex operators. It does not fail on one —
``torch/_inductor/lowering.py`` routes it to an eager fallback kernel and warns
**once per process** via ``@functools.cache``, which on a cluster is
indistinguishable from silence. So a complex arm that declares compilation gets
a run that reports a compiled configuration while executing its complex regions
eagerly, possibly slower than not compiling at all.

Fencing the physics SSOT makes the compiled graph *provably* free of complex
tensors rather than hopefully so, which is what lets an arm declare
``allow_complex`` honestly. The surface is small and already marked: ``fft2c`` /
``ifft2c`` and their DCT peers are the modules that carry complex dtype.

**Cost, stated plainly.** Each fence is a hard graph break: the compiled region
splits and dynamo re-enters the Python frame. That is Python re-entry and guard
evaluation, not lost fusion — Inductor was already falling back to an eager
kernel at that boundary, so no fusion crossed it to begin with. Whether the
trade pays on a given arm is an empirical question, measured per arm with
``torch._dynamo.explain``, not an assumption this module makes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["dynamo_disable"]


def dynamo_disable[F: Callable[..., Any]](fn: F) -> F:
    """Fence *fn* out of every dynamo graph, or return it unchanged.

    Unchanged when dynamo is absent, or when torch is the ``MagicMock`` that
    ``tests/conftest.py`` installs for the torch-less lane. That case is the
    reason this is a function rather than a bare ``@torch._dynamo.disable``:
    applying the mock's attribute as a decorator replaces the function with
    *another mock*, so the module would import, the symbol would exist, and
    every call to a physics op would return a mock instead of a tensor.

    Never returns a non-callable. A fence that cannot be applied is a
    performance question; a fence that silently swaps out the function is a
    correctness one.
    """
    try:
        import torch

        disable = torch._dynamo.disable
    except Exception:  # torch absent, or built without dynamo
        logger.debug("dynamo unavailable; %s left unfenced", getattr(fn, "__name__", fn))
        return fn

    if not callable(disable):
        return fn

    fenced = disable(fn)

    # `callable(fenced)` is NOT sufficient: a MagicMock is callable, so a
    # shimmed torch would pass that test and this would hand back a mock in
    # place of the physics op. The real `torch._dynamo.disable` uses
    # functools.wraps, so it preserves `__name__` as the original string --
    # a mock yields another mock for that attribute.
    expected = getattr(fn, "__name__", None)
    if not callable(fenced) or getattr(fenced, "__name__", None) != expected:
        logger.debug(
            "torch._dynamo.disable did not return a wrapper of %s (shimmed torch?); left unfenced",
            expected or fn,
        )
        return fn
    return fenced  # type: ignore[return-value]
